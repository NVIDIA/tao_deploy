# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions used for mask visualization."""

import cv2
import numpy as np
import math


def get_color_id(num_classes):
    """Function to return a list of color values for each class."""
    colors = []
    for idx in range(num_classes):
        np.random.seed(idx)
        colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    return colors


def overlay_seg_image(inp_img, seg_img, resize_padding, resize_method):
    """The utility function to overlay mask on original image."""
    resize_methods_mapping = {'BILINEAR': cv2.INTER_LINEAR, 'AREA': cv2.INTER_AREA,
                              'BICUBIC': cv2.INTER_CUBIC,
                              'NEAREST_NEIGHBOR': cv2.INTER_NEAREST}
    rm = resize_methods_mapping[resize_method]
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_h = seg_img.shape[0]
    seg_w = seg_img.shape[1]

    if resize_padding:
        p_height_top, p_height_bottom, p_width_left, p_width_right = \
            resize_with_pad(inp_img, seg_w, seg_h)
        act_seg = seg_img[p_height_top:(seg_h - p_height_bottom), p_width_left:(seg_w - p_width_right)]
        seg_img = cv2.resize(act_seg, (orininal_w, orininal_h), interpolation=rm)
    else:
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h), interpolation=rm)

    fused_img = (inp_img / 2 + seg_img / 2).astype('uint8')
    return fused_img


def resize_with_pad(image, f_target_width=None, f_target_height=None):
    """Function to determine the padding width in all the directions."""
    (im_h, im_w) = image.shape[:2]
    ratio = max(im_w / float(f_target_width), im_h / float(f_target_height))
    resized_height_float = im_h / ratio
    resized_width_float = im_w / ratio
    resized_height = math.floor(resized_height_float)
    resized_width = math.floor(resized_width_float)
    padding_height = (f_target_height - resized_height_float) / 2
    padding_width = (f_target_width - resized_width_float) / 2
    f_padding_height = math.floor(padding_height)
    f_padding_width = math.floor(padding_width)
    p_height_top = max(0, f_padding_height)
    p_width_left = max(0, f_padding_width)
    p_height_bottom = max(0, f_target_height - (resized_height + p_height_top))
    p_width_right = max(0, f_target_width - (resized_width + p_width_left))

    return p_height_top, p_height_bottom, p_width_left, p_width_right
