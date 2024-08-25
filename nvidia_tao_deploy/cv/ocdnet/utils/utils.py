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

"""Utils module."""

import os
import cv2
import glob
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt


def show_img(imgs: np.ndarray, title='img'):
    """Show img."""
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    imgs = np.expand_dims(imgs, axis=0)
    for i, img in enumerate(imgs):
        plt.figure()
        plt.title(f'{title}_{i}')
        plt.imshow(img, cmap=None if color else 'gray')
    plt.show()


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    """Draw bbox."""
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path


def save_result(result_path, box_list, score_list, is_output_polygon):
    """Save result."""
    if is_output_polygon:
        with open(result_path, 'wt', encoding="utf-8") as res:
            for i, box in enumerate(box_list):
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                score = score_list[i]
                res.write(result + ',' + str(score) + "\n")
    else:
        with open(result_path, 'wt', encoding="utf-8") as res:
            for i, box in enumerate(box_list):
                score = score_list[i]
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                res.write(result + ',' + str(score) + "\n")


def get_file_list(folder_path: str, p_postfix: list = None, sub_dir: bool = True) -> list:
    """Get file list

    Args:
        folder_path: The path to the folder
        p_postfix: postfix
        sub_dir: check the subfolder

    Returns:
        Return file list
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if p_postfix is None:
        p_postfix = ['.jpg']
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    file_list = [x for x in glob.glob(folder_path + '/*.*') if
                 os.path.splitext(x)[-1] in p_postfix or '.*' in p_postfix]
    return natsorted(file_list)
