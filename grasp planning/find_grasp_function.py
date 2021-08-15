#import h5py
import numpy as np
import cv2
from perception import DepthImage
from perception import CameraIntrinsics
from autolab_core import YamlConfig
from samper_2d import AntipodalDepthImageGraspSampler
from visualization import Visualizer2D as vis
from perception import DepthImage
import cv2
import numpy as np
import torch
from cpnet2 import CPNET
from gcmap_dataset import Gcmap
import datetime


class Grasp_finder():
    def __init__(self):
        self.net = CPNET()
        self.net.load_state_dict(torch.load('model_test/model3_lr0001_bs16_d1.pkl',map_location=torch.device('cpu')))
        device = torch.device("cpu")
        self.net = self.net.to(device)
        sample_config = YamlConfig('config.yaml')
        self.sampler=AntipodalDepthImageGraspSampler(sample_config['sampling'])
        self.camera_intrinsics = CameraIntrinsics.load('primesense.intr')
        
        
    def find_maxq_grasp(self,dep,q_map):
        grasps=self.sampler.sample(dep,self.camera_intrinsics,2000,visualize=False)
        max_q_g=-1
        for g in grasps:
            c1=g.contact_points[0]
            c2=g.contact_points[1]
            q_g=q_map[c1[0],c1[1]]+q_map[c2[0],c2[1]]
            if q_g>max_q_g:
                max_q_g=q_g
                g_max=g
        return g_max,max_q_g


    def normalize(self,img):
        img = np.clip((img - img.mean()), -1, 1)
        return img

    def find(self,depth_data):
        device = torch.device("cpu")
        x_data = depth_data#/1000#/1000

        depth_image=DepthImage(x_data,frame='primesense_overhead')
        depth_image=depth_image.crop(300,300)
        #cv2.imshow('4',depth_image.data)
        #cv2.waitKey(0)
        depth_image=depth_image.inpaint()

        depth_image_data=self.normalize(depth_image.data)
        x = torch.from_numpy(depth_image_data)
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        xc = x.to(device)
        start=datetime.datetime.now()
        q_map = self.net(xc).detach().cpu().numpy()
        q_map = q_map.reshape((300, 300))
        mqg,q = self.find_maxq_grasp(depth_image,q_map)
        end=datetime.datetime.now()
        print('time',(end-start))
        #print(mqg.center)
        pose=mqg.pose()
        print(pose.matrix)
        vis.figure()
        vis.imshow(depth_image)
        vis.grasp(mqg)
        vis.show()
        return pose.matrix














