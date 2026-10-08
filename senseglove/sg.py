import numpy as np
import matplotlib.pyplot as plt
import math
import time
from math import *

def hand_calibration(glove,demonstrator):
  print("\nCalibrate the glove, press any button to start:")
  input()
  print("****************************************************")
  print("Move your fingers")
  cali_steps = 500
  filter_num = 10
  cali_joints = np.zeros((cali_steps,21))
  cali_max = np.zeros(21)
  cali_min = np.zeros(21)
  sort_max = np.zeros(filter_num)
  sort_min = np.zeros(filter_num)
  for i in range(cali_steps):
    for j in range(21):
      cali_joints[i,j] = glove.get_hand_joints().contents[j]
    if i%5 == 0:
      print("\r", end="")
      print("Calibration progress: {}%: ".format(i/5), "▋" * (int(0.2*i) // 2), end="")
  for j in range(21):
    sort_max = np.sort(cali_joints[:,j])[cali_steps-filter_num:]
    sort_min = np.sort(cali_joints[:,j])[:filter_num]
    cali_max[j] = round(np.mean(sort_max))
    cali_min[j] = round(np.mean(sort_min))
  cali = cali_max - cali_min
  file = open('ax.txt','w')
  file.write(str([cali,cali_min]))
  file.close()
  npsave = np.concatenate((cali,cali_min)).copy()
  np.save("./glove_calibration/"+demonstrator,npsave)
  print("\n\nCalibration completed, press any button to continue")
  input()
  return cali, cali_min
    
def map_glove2robot(joints, glove_range, cali_min):
    joints[0] = round(1.4*(joints[0]-cali_min[0])/glove_range[0]-0.1,4) #thumb yaw
    joints[0] = np.clip(joints[0],-0.1,1.3)
    joints[4] = round(0.7*(joints[4]-cali_min[4])/glove_range[4]-0.1,4) #thumb pitch
    joints[4] = np.clip(joints[4],-0.1,0.6)
    joints[8] = round(1.7*(joints[8]-cali_min[8])/glove_range[8],4) #index
    joints[12] = round(1.7*(joints[12]-cali_min[12])/glove_range[12],4) #middle
    joints[16] = round(1.7*(joints[16]-cali_min[16])/glove_range[16],4) #ring
    joints[20] = round(1.7*(joints[20]-cali_min[20])/glove_range[20],4) #pinky
    hand_joints = np.zeros(6)
    for i in range(len(hand_joints)):
        hand_joints[i] = joints[4*i]
        if i > 1:
            hand_joints[i] = np.clip(hand_joints[i],0,1.7)
    return hand_joints

def apply_force(glove,force):
    a = 1.71706106e-03
    b = 2.56987447
    PWM = np.zeros(5)
    for i in range(len(force)):
        PWM[i] = math.sqrt(max((force[i] - b)*8 / a /20, 0))
        PWM[i] = round(np.clip(PWM[i], 0, 100))
    # print(PWM)
    glove.force_feedback(int(PWM[0]),int(PWM[1]),int(PWM[2]),int(PWM[3]),int(PWM[4]))

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def plot(time,force):
  if len(time)>100:
    time.pop(0)
    force.pop(0)
  plt.ion() #开启interactive mode 成功的关键函数
  plt.figure(1)
  plt.clf() #清空画布上的所有内容
  plt.plot(time,np.array(force)[:,0],'-r', label='thumb')
  plt.plot(time,np.array(force)[:,1],'-b', label='index')  
  plt.plot(time,np.array(force)[:,2],'-g', label='middle')
  plt.plot(time,np.array(force)[:,3],'-y', label='ring')  
  plt.plot(time,np.array(force)[:,4],'-k', label='pinky')
  plt.xlabel('time')
  plt.ylabel('force/N')
  plt.legend()
  plt.pause(0.01)