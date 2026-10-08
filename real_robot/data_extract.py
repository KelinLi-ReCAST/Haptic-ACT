import h5py
import os
import cv2
import numpy as np

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        is_sim = root.attrs.get("sim", False)
        compressed = root.attrs.get("compress", False)

        if "/observations/qpos" in root.keys():
            qpos = root["/observations/qpos"][()]
        else:
            qpos = None

        if "/observations/qvel" in root.keys():
            qvel = root["/observations/qvel"][()]
        else:
            qvel = None

        if "/observations/effort" in root.keys():
            effort = root["/observations/effort"][()]
        else:
            effort = None

        if "/observations/c_force" in root.keys():
            c_force = root["/observations/c_force"][()]
        else:
            c_force = None

        action = root["/action"][()]
        image_dict = dict()
        for cam_name in root[f"/observations/images/"].keys():
            image_dict[cam_name] = root[f"/observations/images/{cam_name}"][()]
        if compressed:
            compress_len = root["/compress_len"][()]

    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(
                padded_compressed_image_list
            ):  # [:1000] to save memory
                image_len = int(compress_len[cam_id, frame_id])
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_list.append(image)
            image_dict[cam_name] = image_list

    return qpos, qvel, effort, c_force, action, image_dict

if __name__ == '__main__':
    dataset_dir_list = ['./real_hact','./real_act']
    for dataset_dir in dataset_dir_list:
        force = []
        for i in range(50):
            _,_,_,c_force,_,_ = load_hdf5(dataset_dir, 'episode_'+str(i)+'_compressed')
            if dataset_dir == './real_hact':
                force.append(c_force[:,1:])
            else:
                force.append(c_force[:,1:])
        np.save(dataset_dir+'/force.npy',force)