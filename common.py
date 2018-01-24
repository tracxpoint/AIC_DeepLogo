# The MIT License (MIT)
# Copyright (c) 2016 satojkovic

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

__all__ = ('CLASS_NAME', 'CNN_IN_WIDTH', 'CNN_IN_HEIGHT', 'CNN_IN_CH',
           'CNN_SHAPE', 'TRAIN_DIR', 'TRAIN_IMAGE_DIR',
           'CROPPED_AUG_IMAGE_DIR', 'ANNOT_FILE', 'ANNOT_FILE_WITH_BG')

CLASS_NAME = [
    'becks',
    'carlsberg',
    'cocacola',
    'colgate',
    'corona',
    'danone',
    'erdinger',
    'fosters',
    'fritolay',
    'gillette',
    'guiness',
    'heineken',
    'hershey',
    'kelloggs',
    'kraft',
    'milka',
    'millerhighlife',
    'nescafe',
    'nestle',
    'pampers',
    'paulaner',
    'pepsi',
    'rittersport',
    'stellaartois',
    'tsingtao',
    'Background'
    ]

CNN_IN_WIDTH = 64
CNN_IN_HEIGHT = 32
CNN_IN_CH = 3
CNN_SHAPE = (CNN_IN_HEIGHT, CNN_IN_WIDTH, CNN_IN_CH)

TRAIN_DIR = os.getenv('RETAIL25_PATH', '../Retail25')
TRAIN_IMAGE_DIR = os.path.join(TRAIN_DIR, 'Images')
CROPPED_AUG_IMAGE_DIR = '../DeepLogo/cropped_augmented_images'
ANNOT_FILE = os.path.join(
    TRAIN_DIR, 'training_data_flat.txt')
ANNOT_FILE_WITH_BG = os.path.join(TRAIN_DIR, 'train_annot_with_bg_class.txt')

BING_RESULTS_PATH = './bing/Result'
BING_WEIGHTS_PATH = os.path.join(BING_RESULTS_PATH, 'weights.txt')
BING_2nd_WEIGHTS_PATH = os.path.join(BING_RESULTS_PATH, '2nd_stage_weights.json')
BING_SIZES_PATH = os.path.join(BING_RESULTS_PATH, 'sizes.txt')
num_win_psz = 180
num_bbs = 2000
