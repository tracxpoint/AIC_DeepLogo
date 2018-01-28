import numpy as np
import os
from collections import defaultdict, deque
from itertools import product
from sklearn.model_selection import train_test_split
import shutil
import glob
import common
import util
import skimage.io
from skimage import transform as sktf
from scipy.misc import imresize
import warnings
import cv2
import imutils
import Augmentor

MAX_DATA_AUG_PER_LINE = 120
MAX_SHIFT_WIDTH = common.CNN_IN_WIDTH * 0.1
MAX_SHIFT_HEIGHT = common.CNN_IN_HEIGHT * 0.1
MAX_ROT_DEG = 10
MIN_ROT_DEG = -10
MAX_SCALE_RATE = 0.95
MIN_SCALE_RATE = 0.85


def resize_img(img, size=(common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH)):
    return imresize(img, size, interp='bicubic')


def make_affine_transform(min_rot, max_rot):
    shift_w = int(np.ceil(np.random.rand() * MAX_SHIFT_WIDTH))
    shift_h = int(np.ceil(np.random.rand() * MAX_SHIFT_HEIGHT))
    rot_deg = np.random.uniform(min_rot, max_rot)
    rot_rad = rot_deg * np.pi / 180.0
    scale_rate = np.random.uniform(MIN_SCALE_RATE, MAX_SCALE_RATE)
    params = {}
    params['shift_w'] = 20
    params['shift_h'] = 20
    params['rot_deg'] = rot_deg
    params['rot_rad'] = rot_rad
    params['scale_rate'] = 1

    mat = sktf.AffineTransform(
        translation=(shift_w, shift_h),
        rotation=rot_rad,
        scale=(scale_rate, scale_rate))

    return mat, params



img = skimage.io.imread("./test.jpg")
cropped_img = img
resized_cropped_img = resize_img(cropped_img)
affine_mat, params = make_affine_transform(0 - 10, 0 + 10)
transformed_img = sktf.warp(cropped_img, affine_mat, mode='edge')
transformed_img = imutils.rotate_bound(transformed_img, 270)
transformed_img = resize_img(transformed_img)

skimage.io.imsave("./rotationTest2.jpg", transformed_img)




#p = Augmentor.Pipeline("./testRotation")
#p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
#p.sample(10)

