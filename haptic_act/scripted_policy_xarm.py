import os
os.environ['MUJOCO_GL']='egl'

import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_env_xarm import make_ee_sim_env

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.trajectory[0]['t'] == self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)
        next_waypoint = self.trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        xyz, quat, gripper = self.interpolate(self.curr_waypoint, next_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            xyz = xyz + np.random.uniform(-scale, scale, xyz.shape)

        action = np.concatenate([xyz, quat, gripper])

        self.step_count += 1
        return np.concatenate([action])


class PickAndPlacePolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose = ts_first.observation['mocap']

        obj_info = np.array(ts_first.observation['env_state'])
        obj_xyz = obj_info[:3]
        obj_quat = obj_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        # [1.296, 0.28, 1] [1 0 0 0]
        # [1.476  -0.01    0.7575] [0.93629336 -0.28962948  0.18979606 -0.0587108]
        # [1.476  -0.01    1.0575] [0.93629336 -0.28962948  0.18979606 -0.0587108]
        # [1.656  0.55   1.0575] [0.93629336 -0.28962948  0.18979606 -0.0587108]
        self.trajectory = [
            {"t": 0, "xyz": init_mocap_pose[:3], "quat": init_mocap_pose[3:], "gripper": np.array([1.3, 0, 0, 0, 0, 0])}, # sleep
            {"t": 50, "xyz": init_mocap_pose[:3]+np.array([0, 0, 0.2025]), "quat": np.array([1, 0, 0, 0]), "gripper": np.array([1.3, 0, 0, 0, 0, 0])}, # go up
            {"t": 100, "xyz": obj_xyz+np.array([0.101,  0.04,    0.3]), "quat": np.array([0.93629336, -0.28962948,  0.18979606, -0.0587108]), "gripper": np.array([1.3, 0, 0, 0, 0, 0])}, # approach apple
            {"t": 150, "xyz": obj_xyz+np.array([0.101,  0.04,    0.2075]), "quat": np.array([0.93629336, -0.28962948,  0.18979606, -0.0587108]), "gripper": np.array([1.3, 0, 0, 0, 0, 0])}, # go down
            {"t": 200, "xyz": obj_xyz+np.array([0.101,  0.04,    0.2075]), "quat": np.array([0.93629336, -0.28962948,  0.18979606, -0.0587108]), "gripper": np.array([1.3, 1, 1, 1, 1, 1])}, # close gripper
            {"t": 250, "xyz": obj_xyz+np.array([0.101,  0.04,    0.5075]), "quat": np.array([0.93629336, -0.28962948,  0.18979606, -0.0587108]), "gripper": np.array([1.3, 1, 1, 1, 1, 1])}, # go up
            {"t": 300, "xyz": np.array([1.646,  0.49,   1.0575]), "quat": np.array([0.93629336, -0.28962948,  0.18979606, -0.0587108]), "gripper": np.array([1.3, 1, 1, 1, 1, 1])}, # move to meet position
            {"t": 350, "xyz": np.array([1.646,  0.49,   1.0575]), "quat": np.array([0.93629336, -0.28962948,  0.18979606, -0.0587108]), "gripper": np.array([0, 0, 0, 0, 0, 0])}, # open gripper
            {"t": 400, "xyz": np.array([1.646,  0.49,   1.0575]), "quat": np.array([0.93629336, -0.28962948,  0.18979606, -0.0587108]), "gripper": np.array([0, 0, 0, 0, 0, 0])}, # stay
        ]

def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'pick_and_place' in task_name:
        env = make_ee_sim_env('pick_and_place')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['frontview'])
            plt.ion()

        policy = PickAndPlacePolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['frontview'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'pick_and_place'
    test_policy(test_task_name)

