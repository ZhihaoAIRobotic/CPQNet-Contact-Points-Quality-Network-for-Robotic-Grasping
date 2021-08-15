import h5py
from dexnet import Hdf5Dataset 
from render import scene_render
import numpy as np
import pyrender
from hd5_dataset import Hdf5Dataset_grasp
from constants import *


f = h5py.File('../dexnet_2_database.hdf5','r')
print(list(f['datasets']['3dnet']))
hd5data = Hdf5Dataset('3dnet',f['datasets']['3dnet'])
obj_names = hd5data.object_keys
print(obj_names)
data_grasp=Hdf5Dataset_grasp(database_filename = 'grasp_map.hdf5',access_level=READ_WRITE_ACCESS)

camera_pose = np.array([
    [1.0, 0.0,   0.0,   0.0],
    [0.0,  1, 0.0, 0.0],
    [0.0,  0.0,   1,   0.5],
   [0.0,  0.0, 0.0, 1.0],
 ])
sr = scene_render()
sr.add_camera(camera_pose)

for i in obj_names:
    obj = hd5data.graspable(i)
    tm = obj.mesh.trimesh_
    poses = obj.mesh.stable_poses()
    sr.add_mesh(tm, pos=np.eye(4))
    for j in range(len(poses)):
        obj_pose = poses[j]
        sr.set_obj_pose(obj_pose.T_obj_world.matrix)
        sr.render_depth()
    sr.remove_obj()






