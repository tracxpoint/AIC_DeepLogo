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

import numpy as np
import selectivesearch
import common
import util
import skimage.io
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_bg_proposals(object_proposals, img_instances_annot):

    temp_proposals = []
    bg_proposals = []
    for obj_proposal in object_proposals:
        global_bg = True
        for annot in img_instances_annot:
            annot_rect = util.get_annot_rect(annot)
            if util.iou(obj_proposal, annot_rect) > 0.1:
                global_bg = False
                break
        if global_bg:
            bg_proposals.append(obj_proposal)
    return bg_proposals


def gen_annot_file_line(img_fn, class_name, train_subset_class, rect):
    rect = ' '.join(map(str, rect))
    line = ' '.join([img_fn, class_name, train_subset_class, rect])
    return line


def gen_annot_file_lines(img_instances_annot):
    lines = []

    img_fn, class_name, train_subset_class = util.parse_annot(img_instances_annot[0])
    # Get original annot line
    for annot in img_instances_annot:
        img_fn, class_name, train_subset_class = util.parse_annot(annot)
        annot_rect = util.get_annot_rect(annot)
        lines.append(
            gen_annot_file_line(img_fn, class_name, train_subset_class,
                                annot_rect))

    # Load image
    img = skimage.io.imread(os.path.join(common.TRAIN_IMAGE_DIR, img_fn))

    # Selective search
    object_proposals = util.get_object_proposals(img)
    if len(object_proposals) == 0:
        return lines

    # Background proposals
    bg_proposals = get_bg_proposals(object_proposals, img_instances_annot)
    if len(bg_proposals) == 0:
        return lines

    if len(bg_proposals) >= len(img_instances_annot):
        randomNumbers = random.sample(range(len(bg_proposals)), len(img_instances_annot))
    else:
        # this introduces a slight class imbalance
        randomNumbers = range(len(bg_proposals))

    for index, rndNum in enumerate(randomNumbers):
        annot = img_instances_annot[index]
        # Select bg proposal

        bg_proposal = bg_proposals[rndNum]
        x1, y1, x2, y2 = bg_proposal[0], bg_proposal[1], bg_proposal[0] + bg_proposal[2], bg_proposal[1] + bg_proposal[3]

        img_fn, class_name, train_subset_class = util.parse_annot(annot)
        annot_rect = util.get_annot_rect(annot)
        # lines.append(
        #     gen_annot_file_line(img_fn, class_name, train_subset_class,
        #                         annot_rect))
        lines.append(
            gen_annot_file_line(img_fn, common.CLASS_NAME[-1], train_subset_class,
                                [x1, y1, x2, y2]))

    return lines


def main():
    # Load an annotation file
    annot_train = np.loadtxt(common.ANNOT_FILE, dtype='a')
    print('train_annotation: {}'.format(annot_train.shape[0]))

    # Multi processing
    results = []
    n_workers = os.cpu_count()

    temp_img, _, _ = util.parse_annot(annot_train[0])
    img_instances_annot = []
    with ProcessPoolExecutor(n_workers) as executer, open(
            common.ANNOT_FILE_WITH_BG, 'w') as fw:
        for idx, annot in enumerate(annot_train):
            # just for testing
            # if (idx > 50):
            #     break

            img_fn, class_name, train_subset_class = util.parse_annot(annot)

            # new image
            if img_fn != temp_img:
                results.append(executer.submit(gen_annot_file_lines, img_instances_annot))
                img_instances_annot = []
                img_instances_annot.append(annot)
            else:
                img_instances_annot.append(annot)

            temp_img = img_fn
            temp_annot = annot
        # last image
        results.append(executer.submit(gen_annot_file_lines, img_instances_annot))
        # for annot in annot_train:
        #     results.append(executer.submit(gen_annot_file_lines, annot))

        for result in as_completed(results):
            print('\n'.join(result.result()))
            fw.writelines('\n'.join(result.result()))
            fw.writelines('\n')


if __name__ == '__main__':
    main()
