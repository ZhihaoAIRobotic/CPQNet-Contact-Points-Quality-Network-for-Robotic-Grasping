import torch
import numpy as np
from image import GraspImage
from image import DepthImage
import random
import h5py


class Gcmap(torch.utils.data.Dataset):
    def __init__(self, file_dir, include_depth=True, start=0.0, end=1.0, output_size=300):
        super(Gcmap, self).__init__()
        self.include_depth = include_depth
        self.output_size = output_size
        f = h5py.File(file_dir, "r")
        depth_imgs = []
        q_imgs = []

        for key in f:
            for pose in f[key]:
                try:
                    depth_img = np.array(f[key][pose]['depth_img'])
                    q_img = np.array(f[key][pose]['q_img'])
                except:
                    continue
                depth_imgs.append(depth_img)
                q_imgs.append(q_img)
        l = len(depth_imgs)
        self.depth_imgs = depth_imgs[int(l * start):int(l * end)]
        self.q_imgs = q_imgs[int(l * start):int(l * end)]
        print('grasp_point_f',len(self.depth_imgs))
        print('rgbf',len(self.q_imgs))


    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def _get_crop_attrs_480(self, idx):
        q_img = self.q_imgs[idx]
        point_where = np.where(q_img==1)
        center =  np.array([int(np.mean(point_where[0])),int(np.mean(point_where[1]))])
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def _get_crop_attrs_300(self, idx):
        q_img = self.q_imgs[idx]
        point_where = np.where(q_img==1)
        center = np.array([int(np.mean(point_where[0])),int(np.mean(point_where[1]))])
        left = max(0, min(center[1] - self.output_size // 2, 300 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 300 - self.output_size))
        return center, left, top

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = DepthImage.from_data(self.depth_imgs[idx])
        center, left, top = self._get_crop_attrs_480(idx)
        depth_img.rotate(rot, center.tolist())
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalize()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_grasp(self, idx, rot=0, zoom=1.0):
        grasp_img = GraspImage.from_data(self.q_imgs[idx])
        center, left, top = self._get_crop_attrs_480(idx)
        grasp_img.rotate(rot, center.tolist())
        grasp_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        grasp_img.zoom(zoom)
        grasp_img.resize((self.output_size, self.output_size))
        return grasp_img.img


    def crop(self, img, top_left, bottom_right):
        img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        return img

    def __getitem__(self, idx):
        zoom_factor = np.random.uniform(0.8, 1.0)
        rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
        rot = random.choice(rotations)
        depth_img = self.get_depth(idx, rot, zoom_factor)
        grasp_img = self.get_grasp(idx, rot, zoom_factor)
        x = self.numpy_to_torch(depth_img)
        y = self.numpy_to_torch(grasp_img)
        return x, y, idx, rot, zoom_factor

    def __len__(self):
        return len(self.depth_imgs)


