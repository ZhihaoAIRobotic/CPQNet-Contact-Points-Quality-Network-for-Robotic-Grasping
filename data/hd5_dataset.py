import h5py
from constants import *
import os

class Hdf5Dataset_grasp:
    '''
    write object name, pose, and it's image number
    '''
    def __init__(self,database_filename,access_level=READ_ONLY_ACCESS):
        self.database_filename_ = database_filename
        self.access_level = access_level
        self._load_database()

    def _create_new_db(self):
        """ Creates a new database """
        self.data_ = h5py.File(self.database_filename_, 'w')

    def _load_database(self):
        """ Loads in the HDF5 file """
        if self.access_level == READ_ONLY_ACCESS:
            self.data_ = h5py.File(self.database_filename_, 'r')
        elif self.access_level == READ_WRITE_ACCESS:
            if os.path.exists(self.database_filename_):
                self.data_ = h5py.File(self.database_filename_, 'r+')
            else:
                self._create_new_db()

    def create_obj(self,obj_name):
        for i in self.data_:
            if i == obj_name:
                return 0        
        self.data_.create_group(obj_name)

    def create_pose(self,obj_name,pose_num):
        for i in self.data_[obj_name]:
            if i == pose_num:
                return 0 
        self.data_[obj_name].create_group(pose_num)


    def write_one_data(self,obj_name,pose_num,pose=None,depth_img=None,q_img=None):
        """ writes in the HDF5 file """
        self.data_[obj_name][pose_num].create_dataset('pose',data=pose)
        self.data_[obj_name][pose_num].create_dataset('depth_img',data=depth_img)
        self.data_[obj_name][pose_num].create_dataset('q_img', data=q_img)

    def write_one_depth(self,obj_name,pose_num,depth_img=None):
        try:
            del self.data_[obj_name][pose_num]['depth_img']
            self.data_[obj_name][pose_num]['depth_img'] = depth_img
        except: 
            self.data_[obj_name][pose_num]['depth_img'] = depth_img


    #def generate_dataset(self):
    """ transfer the HDF5 file into trainable data set """

