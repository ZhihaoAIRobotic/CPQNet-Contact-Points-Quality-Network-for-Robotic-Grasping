import h5py

from dexnet import Hdf5Dataset,ParallelJawPtGrasp3D
from render import scene_render
import numpy as np
from hd5_dataset import Hdf5Dataset_grasp
from constants import *
from perception import CameraIntrinsics
from autolab_core import YamlConfig,Point,Direction,RigidTransform
from samper_2d import AntipodalDepthImageGraspSampler
from dexnet import GraspQualityConfigFactory, GraspQualityFunctionFactory


def center_grasp_3d(center_2d,depth,camera_intrinsics):
    center_2d = Point(center_2d.data, frame=camera_intrinsics.frame)
    center_3d = camera_intrinsics.deproject_pixel(depth,center_2d)
    return center_3d


f = h5py.File('../dexnet_2_database.hdf5','r')
hd5data = Hdf5Dataset('3dnet',f['datasets']['3dnet'])
obj_names = hd5data.object_keys
data_grasp_map=Hdf5Dataset_grasp(database_filename = 'grasp_map.hdf5',access_level=READ_WRITE_ACCESS)
camera_pose = np.array([
    [1.0, 0.0,   0.0,   0.1],
    [0.0,  1, 0.0, -0.05],
    [0.0,  0.0,   1,   0.55],
   [0.0,  0.0, 0.0, 1.0],
 ])

sr = scene_render()
sr.add_camera(camera_pose)
camera_intrinsics = CameraIntrinsics.load('primesense.intr')
sample_config = YamlConfig('config.yaml')
sampler=AntipodalDepthImageGraspSampler(sample_config['sampling'])
config_3dgrasp=np.zeros(10)


rotate =  np.array([
    [1.0, 0.0,   0.0],
    [0.0,  -1, 0.0],
    [0.0,  0.0,   -1]
 ])


T_camera_world=RigidTransform(rotation=np.dot(rotate, camera_pose[0:3,0:3]),translation=camera_pose[0:3,3],from_frame='primesense_overhead',to_frame='world')

grasp_config=dict(sample_config['metrics']['force_closure'])

quality_config = GraspQualityConfigFactory.create_config(grasp_config)

for obj_name in obj_names:
    obj = hd5data.graspable(obj_name)
    tm = obj.mesh.trimesh_
    poses = obj.mesh.stable_poses()
    sr.add_mesh(tm, pos=np.eye(4))
    num_grasp_a=0
    pose_num=0
    for pose in poses:
        pose_num=1+pose_num
        if(pose_num>1):
            break
        obj_pose = pose
        T_world_obj=obj_pose.T_obj_world.inverse()
        T_obj_world=obj_pose.T_obj_world
        sr.set_obj_pose(T_obj_world.matrix)
        color_img,depth_img=sr.render_depth()
        data_grasp_map.write_one_depth(obj_name,str(pose_num),depth_img=depth_img)        

    sr.remove_obj()






