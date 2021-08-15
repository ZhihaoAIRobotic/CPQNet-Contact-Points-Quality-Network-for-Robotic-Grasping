import h5py

from dexnet import Hdf5Dataset,ParallelJawPtGrasp3D
from render import scene_render
import numpy as np
from hd5_dataset import Hdf5Dataset_grasp
from constants import *
from perception import DepthImage
from perception import CameraIntrinsics
from autolab_core import YamlConfig,Point,Direction,RigidTransform
from samper_2d import AntipodalDepthImageGraspSampler
from visualization import Visualizer2D as vis
from dexnet import GraspQualityConfigFactory, GraspQualityFunctionFactory

def center_grasp_3d(center_2d,depth,camera_intrinsics):
    '''
    Parameters
    ----------
    center_2d : autolab_core.point
    depth : float
    camera_intrinsics : autolab_core.camera_intrinsics
    Reterns
    -------
    center_3d : autolab_core.point
        point in camera frame
    '''
    center_2d = Point(center_2d.data, frame=camera_intrinsics.frame)
    center_3d = camera_intrinsics.deproject_pixel(depth,center_2d)
    #center_3d.data[2]=-center_3d.data[2]
    return center_3d


f = h5py.File('../dexnet_2_database.hdf5','r')
#print(list(f['datasets']['3dnet']))
hd5data = Hdf5Dataset('3dnet',f['datasets']['3dnet'])
obj_names = hd5data.object_keys
#print(obj_names)
data_grasp_map=Hdf5Dataset_grasp(database_filename = 'test.hdf5',access_level=READ_WRITE_ACCESS)
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
    print(obj_name)
    data_grasp_map.create_obj(obj_name)
    obj = hd5data.graspable(obj_name)
    tm = obj.mesh.trimesh_
    poses = obj.mesh.stable_poses()
    
    sr.add_mesh(tm, pos=np.eye(4))

    pose_num=0
    for pose in poses:
        #print('j',pose)
        num_grasp_a=0
        pose_num=1+pose_num
        if(pose_num>3):
            break
        data_grasp_map.create_pose(obj_name,str(pose_num))

        obj_pose = pose
        T_world_obj=obj_pose.T_obj_world.inverse()
        T_obj_world=obj_pose.T_obj_world
        #print('T_obj_world',T_obj_world)
        sr.set_obj_pose(T_obj_world.matrix)
        color_img,depth_img=sr.render_depth(640,480)
        dep=DepthImage(depth_img,frame='primesense_overhead')
        grasps=sampler.sample(dep,camera_intrinsics,3000,visualize=True)
        #print('grasps len',len(grasps))
        map_g = np.zeros((480,640))
        for g in grasps:
            center=g.center
            angle=g.angle
            depth=g.depth
            #print(g.contact_points[0])
            center_3dgrasp_camera=center_grasp_3d(center,depth,camera_intrinsics)

            T_camera_obj = T_world_obj*T_camera_world

            center_3dgrasp_obj=T_camera_obj*center_3dgrasp_camera

            V_camera = Direction(np.array([np.cos(angle),np.sin(angle),0]),'primesense_overhead')
            V_obj=T_camera_obj*V_camera
      
            config_3dgrasp[0:3]=center_3dgrasp_obj.data
            config_3dgrasp[3:6]=V_obj._data.reshape(3,)
            config_3dgrasp[6]=0.05
            config_3dgrasp[7]=0
        

            grasp_3d = ParallelJawPtGrasp3D(config_3dgrasp)

            T_gripper_obj=grasp_3d.T_grasp_obj
            T_gripper_world=T_obj_world*T_gripper_obj

            T_obj_camera=T_camera_obj.inverse()
            grasp_3dto2d=grasp_3d.project_camera(T_obj_camera,camera_intrinsics)
            
            vis.figure()
            vis.imshow(dep)
            vis.grasp(grasp_3dto2d)
            vis.show()
            quality_fn = GraspQualityFunctionFactory.create_quality_function(obj,quality_config)
            q=quality_fn(grasp_3d).quality
            #print('q',q)
            if q==1:
                num_grasp_a=num_grasp_a+1
                map_g[g.contact_points[0][0],g.contact_points[0][1]]=1
                map_g[g.contact_points[1][0],g.contact_points[1][1]]=1
        
        if(num_grasp_a==0):
            print('nothing')
        else:
            print(obj_name,pose_num,num_grasp_a)
            data_grasp_map.write_one_data(obj_name,str(pose_num),T_obj_world.matrix,depth_img,map_g)
    sr.remove_obj()






