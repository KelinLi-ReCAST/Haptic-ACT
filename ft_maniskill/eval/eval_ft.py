"""Evaluate a checkpoint (F/T Haptic-ACT, ACT, Diffusion Policy or BC) on one of the ManiSkill tasks.

Protocol of the reported results: for every seed k in --seeds, the policy's random seed is set to k and 25 episodes
are run with environment seeds k, k+1, ..., k+24. An episode counts as a success if the task reports success at any
step before it ends (termination, truncation or the step cap). The result is the mean and standard deviation of the
per-seed success rates, written to <out>/summary.json (per-episode results in <out>/seed<k>.json).

Observations given to the policy, matching the datasets:
  observation.images.base_camera, observation.images.wrist_camera   480 x 480 RGB, float in [0, 1]
  observation.state, observation.qvel                               Panda joint positions / velocities (9)
  observation.torque                                                7 arm-joint torques [N m]
  observation.environment_state                                     F/T window: the last 5 TCP wrenches
                                                                    [fx fy fz tx ty tz] (N, N m, TCP frame), oldest
                                                                    first, padded with the first reading
A checkpoint uses the inputs listed in its config and ignores the others; only F/T Haptic-ACT reads the F/T window.

The wrench is computed from the joint torques as in the datasets (dataset_processing/augment_dataset.py): the
least-squares solution of tau = J^T F with the TCP Jacobian in the TCP frame.

F/T Haptic-ACT is run with --learned-trigger: the policy predicts a chunk every step and its re-planning head gives
P(re-plan); above the threshold (and outside the refractory window) the remaining chunk is replaced by the new one,
otherwise the queued chunk continues and the new prediction is used only when the queue runs out.

Example:
    python ft_maniskill/eval/eval_ft.py --task peg --policy-path <checkpoint>/pretrained_model \
        --n-action-steps 50 --learned-trigger 0.5 --out outputs/eval/peg_ft
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import lerobot_policy_acthaptic  # noqa: F401  (registers the acthaptic policy type)
import mani_skill.envs  # noqa: F401  (the haptic-act branch of KelinLi-ReCAST/ManiSkill registers the task variants)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.random_utils import set_seed

# Environment id and step cap per task. The caps leave room above the demonstration lengths.
TASKS = {
    "peg": ("PegInsertionSideFT-v1", 450),
    "pickcube": ("PickCubeVisGoalThr3-v1", 100),
    "stackcube": ("StackCube-v1", 150),
    "pushcube": ("PushCube-v1", 100),
}
CAMERAS = {"base_camera": "base_camera", "hand_camera": "wrist_camera"}  # ManiSkill name -> dataset name
IMAGE_SIZE = 480
N_ARM = 7
HISTORY = 5


def make_env(task: str, render_video: bool) -> gym.Env:
    from mani_skill.utils.wrappers import CPUGymWrapper

    env_id, max_steps = TASKS[task]
    env = gym.make(
        env_id,
        obs_mode="rgb",
        control_mode="pd_joint_pos",  # actions are absolute joint targets: 7 arm joints + gripper
        render_mode="rgb_array" if render_video else None,
        sim_backend="physx_cpu",
        num_envs=1,
        robot_uids="panda_wristcam",
        sensor_configs={"width": IMAGE_SIZE, "height": IMAGE_SIZE},
        max_episode_steps=max_steps,
    )
    return CPUGymWrapper(env, record_metrics=False)  # unbatched numpy observations


class ForceObserver:
    """Joint torques and the TCP wrench window of the robot, reset at every episode."""

    def __init__(self):
        self.wrenches: collections.deque = collections.deque(maxlen=HISTORY)
        self._model = None

    def reset(self):
        self.wrenches.clear()
        self._model = None  # the CPU env rebuilds the articulation on reset

    def __call__(self, env: gym.Env) -> dict[str, np.ndarray]:
        unwrapped = env.unwrapped
        robot = unwrapped.agent.robot
        joint_forces = robot.get_link_incoming_joint_forces()[0]
        tau = np.array([float(joint_forces[j.child_link.index, 3]) for j in robot.active_joints[:N_ARM]], dtype=np.float32)
        if self._model is None:
            self._model = robot._objs[0].create_pinocchio_model()
            tcp_index = unwrapped.agent.tcp.index
            self._tcp = int(tcp_index[0]) if hasattr(tcp_index, "__len__") else int(tcp_index)
        self._model.compute_full_jacobian(robot.qpos[0].cpu().numpy().astype(np.float64))
        jacobian = self._model.get_link_jacobian(self._tcp, local=True)[:, :N_ARM]
        wrench = (np.linalg.pinv(jacobian.T) @ tau).astype(np.float32)
        if not self.wrenches:
            self.wrenches.extend([wrench] * HISTORY)
        self.wrenches.append(wrench)
        return {"torque": tau, "wrench": wrench, "environment_state": np.concatenate(list(self.wrenches))}


def to_policy_batch(raw_obs: dict, forces: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    batch = {}
    for cam, name in CAMERAS.items():
        image = torch.from_numpy(np.asarray(raw_obs["sensor_data"][cam]["rgb"], dtype=np.uint8))
        batch[f"observation.images.{name}"] = image.permute(2, 0, 1).unsqueeze(0).float() / 255
    batch["observation.state"] = torch.from_numpy(np.asarray(raw_obs["agent"]["qpos"], dtype=np.float32)).unsqueeze(0)
    batch["observation.qvel"] = torch.from_numpy(np.asarray(raw_obs["agent"]["qvel"], dtype=np.float32).reshape(-1)).unsqueeze(0)
    for key in ("torque", "wrench", "environment_state"):
        batch[f"observation.{key}"] = torch.from_numpy(forces[key]).unsqueeze(0)
    return batch


class LearnedTrigger:
    """Action selection of F/T Haptic-ACT with the learned re-planning trigger (batch size 1)."""

    def __init__(self, policy, threshold: float, refractory: int):
        self.policy, self.threshold, self.refractory = policy, threshold, refractory
        self.queue: collections.deque = collections.deque()
        self.since = 10**9
        self.n_replans = 0

    def reset(self):
        self.queue.clear()
        self.since = 10**9

    @torch.no_grad()
    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        chunk = self.policy.predict_action_chunk(batch)
        p_replan = float(self.policy.replan_probability()[0])
        self.since += 1
        if (p_replan > self.threshold and self.since >= self.refractory) or not self.queue:
            if self.queue:
                self.n_replans += 1
                self.since = 0
            self.queue.clear()
            self.queue.extend(chunk[:, : self.policy.config.n_action_steps].transpose(0, 1))
        return self.queue.popleft()


def load_policy(path: str, device: str, overrides: dict):
    config = PreTrainedConfig.from_pretrained(path)
    config.device = device
    for key, value in overrides.items():
        setattr(config, key, value)
    policy = get_policy_class(config.type).from_pretrained(path, config=config)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}, "normalizer_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, preprocessor, postprocessor


def run_episode(env, policy, preprocessor, postprocessor, select, observer, seed: int, max_steps: int, video: list | None) -> dict:
    policy.reset()
    if isinstance(select, LearnedTrigger):
        select.reset()
    observer.reset()
    raw_obs, _ = env.reset(seed=seed)
    success, steps = False, 0
    while True:
        batch = preprocessor(to_policy_batch(raw_obs, observer(env)))
        action = postprocessor(select(batch))
        raw_obs, _, terminated, truncated, info = env.step(action[0].cpu().numpy())
        steps += 1
        success = success or bool(info["success"])
        if video is not None:
            video.append(env.render())
        if terminated or truncated or steps >= max_steps:
            return {"env_seed": seed, "success": success, "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--policy-path", required=True, help="checkpoint directory (pretrained_model)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--seeds", default="1000,2000,3000,4000,5000,6000,7000,8000,9000,10000")
    parser.add_argument("--episodes-per-seed", type=int, default=25)
    parser.add_argument("--n-action-steps", type=int, default=None, help="actions executed per chunk (ACT 50, BC 1)")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="DDPM inference steps (Diffusion Policy: 50)")
    parser.add_argument("--learned-trigger", type=float, default=None, help="P(re-plan) threshold (F/T Haptic-ACT: 0.5)")
    parser.add_argument("--refractory", type=int, default=25, help="steps after a re-plan during which it cannot fire again")
    parser.add_argument("--videos", type=int, default=0, help="number of episodes per seed to save as mp4")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    overrides = {}
    if args.n_action_steps is not None:
        overrides["n_action_steps"] = args.n_action_steps
    if args.num_inference_steps is not None:
        overrides["num_inference_steps"] = args.num_inference_steps
    policy, preprocessor, postprocessor = load_policy(args.policy_path, args.device, overrides)
    if args.learned_trigger is not None:
        select = LearnedTrigger(policy, args.learned_trigger, args.refractory)
    else:
        select = policy.select_action
    env, observer = make_env(args.task, render_video=args.videos > 0), ForceObserver()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for seed in (int(s) for s in args.seeds.split(",")):
        result_file = out / f"seed{seed}.json"
        if result_file.exists():  # resume
            per_seed[seed] = json.loads(result_file.read_text())["pc_success"]
            continue
        set_seed(seed)
        episodes = []
        for i in range(args.episodes_per_seed):
            video = [] if i < args.videos else None
            episodes.append(run_episode(env, policy, preprocessor, postprocessor, select, observer, seed + i, TASKS[args.task][1], video))
            if video:
                import imageio

                imageio.mimsave(out / f"seed{seed}_ep{i}.mp4", video, fps=20)
        per_seed[seed] = 100.0 * np.mean([e["success"] for e in episodes])
        result_file.write_text(json.dumps({"pc_success": per_seed[seed], "episodes": episodes}, indent=2))
        print(f"seed {seed}: {per_seed[seed]:.1f} % success", flush=True)
    values = [per_seed[k] for k in sorted(per_seed)]
    summary = {
        "task": args.task,
        "policy_path": str(args.policy_path),
        "protocol": f"{len(values)} seeds x {args.episodes_per_seed} trials",
        "per_seed_success": {str(k): per_seed[k] for k in sorted(per_seed)},
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
    }
    if isinstance(select, LearnedTrigger):
        summary["n_replans"] = select.n_replans
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"mean {summary['mean']:.1f} +- {summary['std_population']:.1f} % ({summary['protocol']})")
    env.close()


if __name__ == "__main__":
    main()
