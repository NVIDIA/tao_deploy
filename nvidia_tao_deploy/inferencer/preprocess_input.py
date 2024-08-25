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

"""Utilities for ImageNet data preprocessing & prediction decoding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
logger = logging.getLogger(__name__)


def _preprocess_numpy_input(x, data_format, mode, color_mode, img_mean, img_std, img_depth, **kwargs):
    """Preprocesses a Numpy array encoding a batch of images.

    # Arguments
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.

    # Returns
        Preprocessed Numpy array.
    """
    assert img_depth in [8, 16], f"Unsupported image depth: {img_depth}, should be 8 or 16."

    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(np.float32, copy=False)

    if mode == 'tf':
        if img_mean and len(img_mean) > 0:
            logger.debug("image_mean is ignored in tf mode.")
        if img_depth == 8:
            x /= 127.5
        else:
            x /= 32767.5
        x -= 1.
        return x

    if mode == 'torch':
        override_mean = False
        if (isinstance(img_mean, list) and (np.array(img_mean) > 1).any()) or (img_mean is None):
            override_mean = True
            logger.debug("image_mean is ignored if larger than 1 in torch mode and overwritten with [0.485, 0.456, 0.406].")
        if img_depth == 8:
            x /= 255.
        else:
            x /= 65535.

        if color_mode == "rgb":
            assert img_depth == 8, f"RGB images only support 8-bit depth, got {img_depth}, "

            if override_mean:
                mean = [0.485, 0.456, 0.406]
                std = [0.224, 0.224, 0.224]
            else:
                mean = img_mean
                std = img_std
        elif color_mode == "grayscale":
            if not img_mean:
                mean = [0.449]
                std = [0.224]
            else:
                assert len(img_mean) == 1, "image_mean must be a list of a single value \
                    for gray image input."
                mean = img_mean
                if img_std is not None:
                    assert len(img_std) == 1, "img_std must be a list of a single value \
                        for gray image input."
                    std = img_std
                else:
                    std = None
        else:
            raise NotImplementedError(f"Invalid color mode: {color_mode}")
    else:
        if color_mode == "rgb":
            assert img_depth == 8, f"RGB images only support 8-bit depth, got {img_depth}, "
            if data_format == 'channels_first':
                # 'RGB'->'BGR'
                if x.ndim == 3:
                    x = x[::-1, ...]
                else:
                    x = x[:, ::-1, ...]
            else:
                # 'RGB'->'BGR'
                x = x[..., ::-1]
            if not img_mean:
                mean = [103.939, 116.779, 123.68]
            else:
                assert len(img_mean) == 3, "image_mean must be a list of 3 values \
                    for RGB input."
                mean = img_mean
            std = None
        else:
            if not img_mean:
                if img_depth == 8:
                    mean = [117.3786]
                else:
                    # 117.3786 * 256
                    mean = [30048.9216]
            else:
                assert len(img_mean) == 1, "image_mean must be a list of a single value \
                    for gray image input."
                mean = img_mean
            std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        for idx in range(len(mean)):
            if x.ndim == 3:
                x[idx, :, :] -= mean[idx]
                if std is not None:
                    x[idx, :, :] /= std[idx]
            else:
                x[:, idx, :, :] -= mean[idx]
                if std is not None:
                    x[:, idx, :, :] /= std[idx]
    else:
        for idx in range(len(mean)):
            x[..., idx] -= mean[idx]
            if std is not None:
                x[..., idx] /= std[idx]
    return x


def preprocess_input(x, data_format="channels_first", mode='caffe', color_mode="rgb", img_mean=None, img_std=None, img_depth=8, **kwargs):
    """Preprocesses a tensor or Numpy array encoding a batch of images.

    # Arguments
        x: Input Numpy or symbolic tensor, 3D or 4D.
            The preprocessed data is written over the input data
            if the data types are compatible. To avoid this
            behaviour, `numpy.copy(x)` can be used.
        data_format: Data format of the image tensor/array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.

    # Returns
        Preprocessed tensor or Numpy array.

    # Raises
        ValueError: In case of unknown `data_format` argument.
    """
    return _preprocess_numpy_input(x, data_format=data_format,
                                   mode=mode, color_mode=color_mode,
                                   img_mean=img_mean,
                                   img_std=img_std, img_depth=img_depth,
                                   **kwargs)
