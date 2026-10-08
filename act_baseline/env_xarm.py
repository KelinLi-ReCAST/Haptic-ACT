import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, START_ARM_POSE

import IPython
e = IPython.embed

XML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulation', 'assets', 'manipulation_tasks')
BOX_POSE = [None] # to be changed from outside
START_ARM_POSE = [0,0,0,0,0,0,0,1.3,0,0,0,0,0]

def make_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [robot arm (7),             # absolute joint position
                        hand (6),                  # unnormalized gripper position] 

    Observation space: {"qpos": Concat[ arm_qpos (7),          # absolute joint position
                                        gripper_position (6),  # unnormalized gripper position]
                        "qvel": Concat[ arm_qvel (7),          # absolute joint velocity (rad)
                                        gripper_velocity (6),  # unnormalized gripper velocity]
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'pick_and_place' in task_name:
        xml_path = os.path.join(XML_DIR, 'pick_and_place.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickAndPlaceTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):

        env_action = np.concatenate([action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        arm_qpos = qpos_raw[:7]
        hand_qpos = qpos_raw[7:13]
        return np.concatenate([arm_qpos,hand_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        arm_qvel = qvel_raw[:7]
        hand_qvel = qvel_raw[7:13]
        return np.concatenate([arm_qvel, hand_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    @staticmethod
    def get_contact_force(physics):
        contact_force=[]
        contact_force.append(np.array(physics.data.sensor("thumb_touch").data))
        contact_force.append(np.array(physics.data.sensor("index_touch").data))
        contact_force.append(np.array(physics.data.sensor("middle_touch").data))
        contact_force.append(np.array(physics.data.sensor("ring_touch").data))
        contact_force.append(np.array(physics.data.sensor("pinky_touch").data))
        contact_force = np.array(contact_force).reshape(5)
        # print(contact_force)
        return contact_force

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['contact_force'] = self.get_contact_force(physics)
        obs['images'] = dict()
        obs['images']['birdview'] = physics.render(height=480, width=640, camera_id='birdview')
        obs['images']['sideview'] = physics.render(height=480, width=640, camera_id='sideview')
        obs['images']['frontview'] = physics.render(height=480, width=640, camera_id='frontview')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class PickAndPlaceTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:13] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        #qpos [:7]: arm joints, [7:19]: hand joints, [19:26]: box joint, [26:]: object joint
        env_state = physics.data.qpos.copy()[26:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        soft_pick = False
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_hand = ("thumb1", "camera_contact1") in all_contact_pairs or ("thumb2", "camera_contact1") in all_contact_pairs or ("thumb3", "camera_contact1") in all_contact_pairs or ("thumb4", "camera_contact1") in all_contact_pairs or ("index1", "camera_contact1") in all_contact_pairs or ("index2", "camera_contact1") in all_contact_pairs or ("thumb1", "camera_contact0") in all_contact_pairs or ("thumb2", "camera_contact0") in all_contact_pairs or ("thumb3", "camera_contact0") in all_contact_pairs or ("thumb4", "camera_contact0") in all_contact_pairs or ("index1", "camera_contact0") in all_contact_pairs or ("index2", "camera_contact0") in all_contact_pairs
        touch_box = ("box_side4", "camera_contact1") in all_contact_pairs or ("box_side4", "camera_contact0") in all_contact_pairs
        touch_table = ("tablegeom0", "camera_contact1") in all_contact_pairs
        # print("table:", touch_table, "hand:", touch_hand, "box:", touch_box)

        contact_force = physics.data.sensor("index_touch").data
        if contact_force > 0 and contact_force < 20:
            soft_pick = True

        # print("table:", touch_table, "hand:", touch_hand, "box:", touch_box)
        reward = 0
        if touch_hand:  # pick
            reward = 1
        if touch_hand and not touch_table:  # lifted
            reward = 2    
        if touch_hand and soft_pick: # contact force
            reward = 3
        if touch_box and not touch_hand: # place
            reward = 4
        return reward