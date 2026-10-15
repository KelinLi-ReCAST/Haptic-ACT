"""Add force-derived columns to a dataset (written as a copy at --out), computed from the columns it already has
(observation.torque, observation.state), so no simulation is replayed:

  observation.torque_hist        (35,) the last 5 joint-torque readings, oldest first, padded at the episode start
                                       with the episode's first reading
  observation.wrench              (6,) least-squares external wrench at the TCP, in the TCP frame
                                       (fx fy fz in N, tx ty tz in N m), from tau = J^T F with SAPIEN's pinocchio
                                       Jacobian at each frame. With link gravity disabled in ManiSkill's Panda, this
                                       is what a wrist F/T sensor at the TCP reads in quasi-static contact.
  observation.environment_state  (35,) copy of torque_hist (replaced by the F/T window in make_ft_dataset.py)
  observation.torque_do           (7,) torque with a fixed 30 % of the frames zeroed (per frame, seed 0)

eval_ft.py computes the same wrench at rollout time.
Usage: python augment_dataset.py --root <dataset> --out <new dataset> [--env PegInsertionSideFT-v1]"""
import argparse
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset

HIST = 5
ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--env", default="PegInsertionSideFT-v1", help="any Panda task; only the robot's kinematic model is used")
args = ap.parse_args()
ds = LeRobotDataset(repo_id="local/src", root=args.root)
hf = ds.hf_dataset
tau = np.asarray(hf["observation.torque"], dtype=np.float32)
q = np.asarray(hf["observation.state"], dtype=np.float32)
ep = np.asarray(hf["episode_index"]).reshape(-1)
n = len(ep)
print("frames", n, "episodes", ep.max() + 1)

# Torque history, per episode.
hist = np.zeros((n, HIST * 7), np.float32)
starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
ends = np.r_[starts[1:], n]
for a, b in zip(starts, ends):
    t = tau[a:b]
    for k in range(HIST):
        idx = np.clip(np.arange(b - a) - (HIST - 1 - k), 0, None)
        hist[a:b, k * 7:(k + 1) * 7] = t[idx]

# TCP wrench from the robot's pinocchio model (one env, only for the kinematic model).
env = gym.make(args.env, obs_mode="none", control_mode="pd_joint_pos", sim_backend="cpu", render_mode=None)
env.reset(seed=0)
unwrapped = env.unwrapped
model = unwrapped.agent.robot._objs[0].create_pinocchio_model()
tcp = unwrapped.agent.tcp.index
tcp = int(tcp[0]) if hasattr(tcp, "__len__") else int(tcp)
wrench = np.zeros((n, 6), np.float32)
for i in range(n):
    model.compute_full_jacobian(q[i].astype(np.float64))
    jacobian = model.get_link_jacobian(tcp, local=True)[:, :7]  # 6 x 7, TCP-frame twist per arm joint
    wrench[i] = np.linalg.pinv(jacobian.T) @ tau[i]  # least-squares F from tau = J^T F
    if i % 50000 == 0:
        print("wrench", i, "/", n)
env.close()

rng = np.random.default_rng(0)
torque_do = tau.copy()
torque_do[rng.random(n) < 0.3] = 0.0


# add_features assigns a 1-D slice per shard for array inputs, which pandas rejects for 2-D values; the callable
# form appends one row at a time and accepts vectors. `row["index"]` is the global frame index.
def per_frame(arr):
    return lambda row, ep, fi: arr[int(row["index"])]


hist_names = [f"t-{HIST - 1 - k}_tau_{j + 1}" for k in range(HIST) for j in range(7)]
features = {
    "observation.torque_hist": (per_frame(hist), {"dtype": "float32", "shape": [HIST * 7], "names": hist_names}),
    "observation.wrench": (per_frame(wrench), {"dtype": "float32", "shape": [6], "names": ["fx", "fy", "fz", "tx", "ty", "tz"]}),
    "observation.environment_state": (per_frame(hist), {"dtype": "float32", "shape": [HIST * 7], "names": hist_names}),
    "observation.torque_do": (per_frame(torque_do), {"dtype": "float32", "shape": [7], "names": [f"tau_{j + 1}" for j in range(7)]}),
}
out = add_features(ds, features, output_dir=args.out, repo_id=f"local/{args.out.name}")
print("wrote", args.out, "features:", sorted(out.meta.features))
print("wrench |F| quantiles (N):", np.round(np.percentile(np.linalg.norm(wrench[:, :3], axis=1), [50, 90, 99, 100]), 1))
