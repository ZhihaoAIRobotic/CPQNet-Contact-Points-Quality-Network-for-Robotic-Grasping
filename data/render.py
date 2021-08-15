import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
#one image corespond one item name and pose
import cv2

class scene_render():
    def __init__(self):
        self.scene = pyrender.Scene()
        tm_floor = trimesh.load('table.obj')
        #self.add_mesh(tm_floor)

    def add_camera(self, camera_pose):
        camera = pyrender.IntrinsicsCamera(525,525,319.5,239.5)
        self.scene.add(camera, pose=camera_pose)
        light = pyrender.SpotLight(color=np.ones(3), intensity=10.0,
                                   innerConeAngle=np.pi / 16.0,
                                   outerConeAngle=np.pi / 6.0)
        light_pose=np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1, 0.0, 0.0],
            [0.0, 0.0, 1, 2.2],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.scene.add(light, pose=light_pose)

    def add_mesh(self, trimesh, pos=np.eye(4)):
        mesh = pyrender.Mesh.from_trimesh(trimesh)
        self.mn = pyrender.Node(mesh=mesh, matrix=pos)
        self.scene.add_node(self.mn)

    def set_obj_pose(self, pos):
        self.scene.set_pose(self.mn, pose=pos)

    def remove_obj(self):
        self.scene.remove_node(self.mn)

    def render_depth(self,width,height,vis=False):
        r = pyrender.OffscreenRenderer(width, height)
        color, depth = r.render(self.scene)
        if vis:
            cv2.imshow('color',color)
            cv2.imshow('depth',depth)
            cv2.waitKey(5000)
        return color,depth
        

    def view(self):
        pyrender.Viewer(self.scene)







