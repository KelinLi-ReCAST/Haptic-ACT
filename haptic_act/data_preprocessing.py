import h5py
import numpy as np
import time

def downsample(data, sample_size):

    data_length = data.shape[0]
    
    if data_length < sample_size:
        # If original data timesteps is smaller than sample_size, will resample
        indices = np.random.choice(data_length, sample_size, replace=True)
    else:
        # Otherwise, won't resample
        indices = np.random.choice(data_length, sample_size, replace=False)
    
    sampled_data = data[indices]
    
    return sampled_data


def format_convert(qpos, qvel, action, image, sample_size):

    camera_names = ['top']
    # with h5py.File('my_data.hdf5', 'r') as root:
    #     qpos = root['/observations/qpos'][()]
    #     qvel = root['/observations/qvel'][()]
    #     action = root['/action'][()]
    #     image = root['/observations/images/top']
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    for i in range(len(qpos)):
        data_dict['/observations/qpos'].append(qpos[i,:])
        data_dict['/observations/qvel'].append(qvel[i,:])
        data_dict['/action'].append(action[i,:])
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(image[i,:,:,:])

    t0 = time.time()

    for i in range(50):
        with h5py.File(f'data/preprocessed/episode_{i}.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (sample_size, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (sample_size, 13))
            qvel = obs.create_dataset('qvel', (sample_size, 13))
            action = root.create_dataset('action', (sample_size, 13))

            for name, array in data_dict.items():
                root[name][...] = array

        print(f'Saving: {time.time() - t0:.1f} secs\n')

    return 0

if __name__ == '__main__':

    qpos = np.random.random(size=(1326,13))
    qvel = np.random.random(size=(1326,13))
    action = np.random.random(size=(1326,13))
    image = np.random.random(size=(1326,480,640,3))

    # Timesteps for each episode
    sample_size = 400

    qpos = downsample(qpos,sample_size)
    qvel = downsample(qvel,sample_size)
    action = downsample(action,sample_size)
    image = downsample(image,sample_size)

    format_convert(qpos,qvel,action,image,sample_size)