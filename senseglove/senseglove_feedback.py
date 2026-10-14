#Revolute joints: palm_index1_0_joint, index1_0_index1_joint, index1_index2_joint, index2_index3_joint
#                 palm_mid1_0_joint, mid1_0_mid1_joint, mid1_mid2_joint, mid2_mid3_joint
#                 palm_ring1_0_joint, ring1_0_ring1_joint, ring1_ring2_joint, ring2_ring3_joint
#                 palm_pinky1_0_joint, pinky1_0_pinky1_joint, pinky1_pinky2_joint, pinky2_pinky3_joint
#                 palm_thumb1_0_joint, thumb1_0_thumb1_joint, thumb1_thumb2_joint, thumb2_thumb3_joint

import numpy as np
import matplotlib.pyplot as plt
import os
import ctypes
import argparse
import sg

ax = []
ay = []
np.set_printoptions(suppress=True)
joints = np.zeros(21)
dll = ctypes.cdll.LoadLibrary
# load the SDK libraries first so that libsenseglove.so finds them without LD_LIBRARY_PATH
for _lib in ['libSGConnect.so', 'libSGCoreCpp.so']:
    ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', _lib), mode=ctypes.RTLD_GLOBAL)
glove = dll(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', 'libsenseglove.so')) 
glove.get_hand_joints.restype = ctypes.POINTER(ctypes.c_int * 21)

def main(username, apply_force, glove_calibration):
    step = 0
    #calibrate the senseglove (needed for diffrerent user)
    if glove_calibration:
        glove_range, cali_min = sg.hand_calibration(glove,username)
    else:
        calibration_data = np.load(os.path.join(sg.CALIBRATION_DIR, username + ".npy"))
        glove_range = calibration_data[:21]
        cali_min = calibration_data[21:]
    
    #apply force to senseglove
    finger_force_array = []
    finger_force = [40,0,0,0,0]
    finger_force_array.append(finger_force)
    if apply_force:
        sg.apply_force(glove,finger_force)
    
    #map from senseglove to robot hand
    while True:
        ax.append(step)
        ay.append(finger_force)
        sg.plot(ax,ay)
        joints_temp = glove.get_hand_joints().contents
        for i in range(21):
            joints[i] = joints_temp[i]
        # print(joints)
        hand_joints = sg.map_glove2robot(joints, glove_range, cali_min)
        step = step + 1
        print(hand_joints)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, required=True, help='name of the calibration file, senseglove/glove_calibration/<username>.npy')
    parser.add_argument('--no_calibration', action='store_true', help='load an existing calibration instead of calibrating')
    parser.add_argument('--apply_force', action='store_true', help='send a test force to the thumb')
    args = parser.parse_args()

    glove.connect()
    main(args.username, args.apply_force, not args.no_calibration)
    glove.disconnect()    