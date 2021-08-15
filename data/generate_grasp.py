
import numpy as np
from autolab_core import YamlConfig
from samper_2d import AntipodalDepthImageGraspSampler
from visualization import Visualizer2D as vis




sample_config = YamlConfig('config.yaml')
sampler=AntipodalDepthImageGraspSampler(sample_config['sampling'])
config_3dgrasp=np.zeros(10)

grasps=sampler.sample(dep,camera_intrinsics,3000,visualize=False)

for g in grasps:
    if q_g>max_q_g:
        max_q_g=q_g
        g_max=g
            vis.figure()
            vis.imshow(dep)
            vis.grasp(grasp_3dto2d)
            vis.show()


