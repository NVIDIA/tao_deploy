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

"""D-DETR loader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import cv2

from nvidia_tao_deploy.dataloader.coco import COCOLoader
from nvidia_tao_deploy.inferencer.preprocess_input import preprocess_input


def resize(image, target, size, max_size=None):
    """resize."""
    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        """get_size_with_aspect_ratio."""
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        """get_size."""
        # Size needs to be (width, height)
        if isinstance(size, (list, tuple)):
            return_size = size[::-1]
        else:
            return_size = get_size_with_aspect_ratio(image_size, size, max_size)
        return return_size

    size = get_size(image.size, size, max_size)

    # PILLOW bilinear is not same as F.resize from torchvision
    # PyTorch mimics OpenCV's behavior.
    # Ref: https://tcapelle.github.io/pytorch/fastai/2021/02/26/image_resizing.html
    rescaled_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * [ratio_width, ratio_height, ratio_width, ratio_height]
        target["boxes"] = scaled_boxes

    h, w = size
    target["size"] = np.array([h, w])

    return rescaled_image, target


class DDETRCOCOLoader(COCOLoader):
    """D-DETR DataLoader."""

    def __init__(
        self,
        image_std=None,
        **kwargs
    ):
        """Init.

        Args:
            image_std (list): image standard deviation.
        """
        super().__init__(**kwargs)
        self.image_std = image_std

    def _get_single_processed_item(self, idx):
        """Load and process single image and its label."""
        gt_image_info, image_id = self._load_gt_image(idx)
        gt_image, gt_scale = gt_image_info
        gt_label = self._load_gt_label(idx)
        return gt_image, gt_scale, image_id, gt_label

    def preprocess_image(self, image_path):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.

        This includes padding, resizing, normalization, data type casting, and transposing.
        This Image Batcher implements one algorithm for now:
        * DDETR: Resizes and pads the image to fit the input size.

        Args:
            image_path(str): The path to the image on disk to load.

        Returns:
            image (np.array): A numpy array holding the image sample, ready to be concatenated
                              into the rest of the batch
            scale (list): the resize scale used, if any.
        """
        scale = None
        image = Image.open(image_path)
        image = image.convert(mode='RGB')

        image = np.asarray(image, dtype=self.dtype)
        image, _ = resize(image, None, size=(self.height, self.width))

        if self.data_format == "channels_first":
            image = np.transpose(image, (2, 0, 1))
        image = preprocess_input(image,
                                 data_format=self.data_format,
                                 img_std=self.image_std,
                                 mode='torch')
        return image, scale
