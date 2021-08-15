import h5py
import numpy as np
from hd5_dataset import Hdf5Dataset_grasp
from constants import *

f = h5py.File('grasp_map_480_640.hdf5', "r")
depth_imgs = []
q_imgs = []

data_grasp_map=Hdf5Dataset_grasp(database_filename = 'grasp_map_480640_c.hdf5',access_level=READ_WRITE_ACCESS)

num_d = 0
for key in f:
    data_grasp_map.create_obj(key)
    for pose in f[key]:
        data_grasp_map.create_pose(key,str(pose))
        print(key,pose)
        try:
            depth_img = np.array(f[key][pose]['depth_img'])
            q_img = np.array(f[key][pose]['q_img'])
        except:
            continue
        num_d = num_d+1
        q_img_copy = q_img.copy()
        for i in range(480):
            for j in range(640):
                if q_img[i,j]==1:
                    if q_img[i+2,j]==1:
                        q_img_copy[i+1,j]=1
                    if q_img[i,j+2]==1:
                        q_img_copy[i,j+1]=1
                    if q_img[i+2,j+2]==1:
                        q_img_copy[i+1,j+1]=1
        
        data_grasp_map.write_one_data(key,pose,0,depth_img,q_img_copy)


print('num_d',num_d)
