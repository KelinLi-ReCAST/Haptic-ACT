import pathlib

### Task parameters
DATA_DIR = str(pathlib.Path(__file__).resolve().parent / 'data') + '/'
SIM_TASK_CONFIGS = {
    'pick_and_place': {
        'dataset_dir': DATA_DIR + 'pick_and_place/raw_data',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['frontview']
    },
}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]