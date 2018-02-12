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
from scipy.ndimage import rotate
from os import walk
import math
# import Augmentor

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

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    # if h > sh or w > sw: # shrinking image
    interp = cv2.INTER_AREA
    # else: # stretching image
    #     interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    # if aspect > 1: # horizontal image
    #     new_w = sw
    #     new_h = np.round(new_w/aspect).astype(int)
    #     pad_vert = (sh-new_h)/2
    #     pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
    #     pad_left, pad_right = 0, 0
    # elif aspect < 1: # vertical image
    if True:
        new_h = sh
        new_w = min(64, np.round(new_h*aspect).astype(int))
        # new_w = sw
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad

    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    # if pad_left>=0 and pad_right>=0 and pad_top>=0 and pad_bot>=0:
    scaled_img = cv2.copyMakeBorder(scaled_img, abs(pad_top), abs(pad_bot), abs(pad_left), abs(pad_right), borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

rootDir = '/home/hussam/mldatasets/Logo_TestSet_Cropped/'
for subdir, dirs, files in os.walk(rootDir):
    for filename in files:
        filepath = os.path.join(subdir, filename)
        img = skimage.io.imread(filepath)
# cropped_img = img
# resized_cropped_img = resize_img(cropped_img)
# affine_mat, params = make_affine_transform(0 - 10, 0 + 10)
# transformed_img = sktf.warp(cropped_img, affine_mat, mode='edge')
        for rot_offset in [90, 180, 270]:
            print(filename)
            # transformed_img = imutils.rotate_bound(img, rot_offset)
            # transformed_img = resize_img(transformed_img)
            h, w = img.shape[:2]
            aspect = w/h
            if aspect > 4: # horizontal image
                transformed_img = imutils.rotate(img, rot_offset)
                # transformed_img = rotate(img, rot_offset, reshape=False)
            else: # vertical image
                transformed_img = rotate(img, rot_offset, reshape=True)

# v_img = cv2.imread('./test.jpg') # vertical image
            transformed_img = resizeAndPad(transformed_img, (32,64), 0)
            skimage.io.imsave(os.path.join("/home/hussam/mldatasets/Logos_aug_test",str(rot_offset) +"_"+ filename), transformed_img)


#p = Augmentor.Pipeline("./testRotation")
#p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
#p.sample(10)
