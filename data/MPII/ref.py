import numpy as np
import h5py
from imageio import imread
import os
import time

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

annot_dir = 'data/MPII/annot'
img_dir = 'data/MPII/images'

assert os.path.exists(img_dir)
mpii, num_examples_train, num_examples_val = None, None, None

import cv2

class MPII:
    def __init__(self):
        print('loading data...')
        tic = time.time()

        train_f = h5py.File(os.path.join(annot_dir, 'train.h5'), 'r')
        val_f = h5py.File(os.path.join(annot_dir, 'valid.h5'), 'r')
        
        self.t_center = train_f['center'][()]
        t_scale = train_f['scale'][()]
        t_part = train_f['part'][()]
        t_visible = train_f['visible'][()]
        t_normalize = train_f['normalize'][()]
        t_imgname = [None] * len(self.t_center)
        for i in range(len(self.t_center)):
            t_imgname[i] = train_f['imgname'][i].decode('UTF-8')
        
        self.v_center = val_f['center'][()]
        v_scale = val_f['scale'][()]
        v_part = val_f['part'][()]
        v_visible = val_f['visible'][()]
        v_normalize = val_f['normalize'][()]
        v_imgname = [None] * len(self.v_center)
        for i in range(len(self.v_center)):
            v_imgname[i] = val_f['imgname'][i].decode('UTF-8')        
        
        self.center = np.append(self.t_center, self.v_center, axis=0)
        self.scale = np.append(t_scale, v_scale)
        self.part = np.append(t_part, v_part, axis=0)
        self.visible = np.append(t_visible, v_visible, axis=0)
        self.normalize = np.append(t_normalize, v_normalize)
        self.imgname = t_imgname + v_imgname
        
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        
    def getAnnots(self, idx):
        '''
        returns h5 file for train or val set
        '''
        return self.imgname[idx], self.part[idx], self.visible[idx], self.center[idx], self.scale[idx], self.normalize[idx]
    
    def getLength(self):
        return len(self.t_center), len(self.v_center)

def init():
    global mpii, num_examples_train, num_examples_val
    mpii = MPII()
    num_examples_train, num_examples_val = mpii.getLength()
    
# Part reference
parts = {'mpii':['rank', 'rkne', 'rhip',
                 'lhip', 'lkne', 'lank',
                 'pelv', 'thrx', 'neck', 'head',
                 'rwri', 'relb', 'rsho',
                 'lsho', 'lelb', 'lwri']}

flipped_parts = {'mpii':[5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]}

part_pairs = {'mpii':[[0, 5], [1, 4], [2, 3], [6], [7], [8], [9], [10, 15], [11, 14], [12, 13]]}

pair_names = {'mpii':['ankle', 'knee', 'hip', 'pelvis', 'thorax', 'neck', 'head', 'wrist', 'elbow', 'shoulder']}

def setup_val_split():
    '''
    returns index for train and validation imgs
    index for validation images starts after that of train images
    so that loadImage can tell them apart
    '''
    valid = [i+num_examples_train for i in range(num_examples_val)]
    train = [i for i in range(num_examples_train)]
    return np.array(train), np.array(valid)
    
def get_img(idx):
    imgname, __, __, __, __, __ = mpii.getAnnots(idx)
    path = os.path.join(img_dir, imgname)
    img = imread(path)
    return img

def get_path(idx):
    imgname, __, __, __, __, __ = mpii.getAnnots(idx)
    path = os.path.join(img_dir, imgname)
    return path
    
def get_kps(idx):
    __, part, visible, __, __, __ = mpii.getAnnots(idx)
    kp2 = np.insert(part, 2, visible, axis=1)
    kps = np.zeros((1, 16, 3))
    kps[0] = kp2
    return kps

def get_normalized(idx):
    __, __, __, __, __, n = mpii.getAnnots(idx)
    return n

def get_center(idx):
    __, __, __, c, __, __ = mpii.getAnnots(idx)
    return c
    
def get_scale(idx):
    __, __, __, __, s, __ = mpii.getAnnots(idx)
    return s