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

"""Classification loader."""

import os
from pathlib import Path
from abc import ABC

import numpy as np
from PIL import Image
from nvidia_tao_deploy.cv.common.constants import VALID_IMAGE_EXTENSIONS
from nvidia_tao_deploy.inferencer.preprocess_input import preprocess_input


# padding size.
# We firstly resize to (target_width + CROP_PADDING, target_height + CROP_PADDING)
# , then crop to (target_width, target_height).
# for standard ImageNet size: 224x224 the ratio is 0.875(224 / (224 + 32)).
# but for EfficientNet B1-B7, larger resolution is used, hence this ratio
# is no longer 0.875
# ref:
# https://github.com/tensorflow/tpu/blob/r1.15/models/official/efficientnet/preprocessing.py#L110
CROP_PADDING = 32

_PIL_INTERPOLATION_METHODS = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
}


class ClassificationLoader(ABC):
    """Classification Dataloader."""

    def __init__(self,
                 shape,
                 image_dirs,
                 class_mapping,
                 is_inference=False,
                 batch_size=10,
                 data_format="channels_first",
                 interpolation_method="bicubic",
                 mode="caffe",
                 crop="center",
                 image_mean=None,
                 image_std=None,
                 image_depth=8,
                 dtype=None):
        """Init.

        Args:
            shape (list): list of input dimension that is either (c, h, w) or (h, w, c) format.
            image_dirs (list): list of image directories.
            label_dirs (list): list of label directories.
            class_mapping (dict): class mapping. e.g. {'aeroplane': 0, 'car': 1}
            is_inference (bool): If set true, we do not load labels (Default: False)
            interpolation_method (str): Bilinear / Bicubic.
            mode (str): caffe / torch
            crop (str): random / center
            batch_size (int): size of the batch.
            image_mean (list): image mean used for preprocessing.
            image_std (list): image std used for preprocessing.
            image_depth(int): Bit depth of images(8 or 16).
            dtype (str): data type to cast to
        """
        self.image_paths = []
        self.is_inference = is_inference
        self._add_source(image_dirs[0])  # WARNING(@yuw): hardcoded 0
        self.image_paths = np.array(self.image_paths)
        self.data_inds = np.arange(len(self.image_paths))
        self.class_mapping = class_mapping

        self.resample = _PIL_INTERPOLATION_METHODS[interpolation_method]
        self.mode = mode
        self.crop = crop

        self.data_format = data_format
        if data_format == "channels_first":
            self.num_channels, self.height, self.width = shape
        else:
            self.height, self.width, self.num_channels = shape

        self.image_depth = image_depth
        self.batch_size = batch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.n_samples = len(self.data_inds)
        self.dtype = dtype
        self.n_batches = int(len(self.image_paths) // self.batch_size)
        assert self.n_batches > 0, "empty image dir or batch size too large!"
        self.model_img_mode = 'rgb' if self.num_channels == 3 else 'grayscale'

    def _add_source(self, image_folder):
        """Add classification sources."""
        images = [p.resolve() for p in Path(image_folder).glob("**/*") if p.suffix in VALID_IMAGE_EXTENSIONS]
        images = sorted(images)
        self.image_paths = images

    def __len__(self):
        """Get length of Sequence."""
        return self.n_batches

    def _load_gt_image(self, image_path):
        """Load GT image from file."""
        img = Image.open(image_path)
        if self.num_channels == 3:
            img = img.convert('RGB')  # Color Image
        else:
            if self.image_depth == 16:
                img = img.convert('I')  # PIL int32 mode for 16-bit images
            else:
                img = img.convert('L')  # Grayscale Image
        return img

    def __iter__(self):
        """Iterate."""
        self.n = 0
        return self

    def __next__(self):
        """Load a full batch."""
        images = []
        labels = []
        if self.n < self.n_batches:
            for idx in range(self.n * self.batch_size,
                             (self.n + 1) * self.batch_size):
                image, label = self._get_single_processed_item(idx)
                images.append(image)
                labels.append(label)
            self.n += 1

            return self._batch_post_processing(images, labels)
        raise StopIteration

    def _batch_post_processing(self, images, labels):
        """Post processing for a batch."""
        images = np.array(images)

        # try to make labels a numpy array
        is_make_array = True
        x_shape = None
        for x in labels:
            if not isinstance(x, np.ndarray):
                is_make_array = False
                break
            if x_shape is None:
                x_shape = x.shape
            elif x_shape != x.shape:
                is_make_array = False
                break

        if is_make_array:
            labels = np.array(labels)

        return images, labels

    def _get_single_processed_item(self, idx):
        """Load and process single image and its label."""
        image, label = self._get_single_item_raw(idx)
        image = self.preprocessing(image)
        return image, label

    def _get_single_item_raw(self, idx):
        """Load single image and its label.

        Returns:
            image (PIL.image): image object in original resolution
            label (int): one-hot encoded class label
        """
        image = self._load_gt_image(self.image_paths[self.data_inds[idx]])
        img_dir = os.path.dirname(self.image_paths[self.data_inds[idx]])
        if self.is_inference:
            label = -1
        else:
            label = self.class_mapping[os.path.basename(img_dir)]
        return image, label

    def preprocessing(self, image):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.

        This includes padding, resizing, normalization, data type casting, and transposing.

        Args:
            image (PIL.image): The Pillow image on disk to load.

        Returns:
            image (np.array): A numpy array holding the image sample, ready to be concatenated
                              into the rest of the batch
        """
        width, height = image.size

        if self.crop == 'center':
            # Resize keeping aspect ratio
            # result should be no smaller than the targer size, include crop fraction overhead
            target_size_before_crop = (
                self.width + CROP_PADDING,
                self.height + CROP_PADDING
            )
            ratio = max(
                target_size_before_crop[0] / width,
                target_size_before_crop[1] / height
            )
            target_size_before_crop_keep_ratio = int(width * ratio), int(height * ratio)
            image = image.resize(target_size_before_crop_keep_ratio, resample=self.resample)
            width, height = image.size
            left_corner = int(round(width / 2)) - int(round(self.width / 2))
            top_corner = int(round(height / 2)) - int(round(self.height / 2))
            image = image.crop(
                (left_corner,
                    top_corner,
                    left_corner + self.width,
                    top_corner + self.height))
        else:
            image = image.resize((self.width, self.height), self.resample)
        image = np.asarray(image, dtype=self.dtype)

        if self.data_format == "channels_first":
            if image.ndim == 2 and self.model_img_mode == 'grayscale':
                image = np.expand_dims(image, axis=2)
            image = np.transpose(image, (2, 0, 1))

        # Normalize and apply imag mean and std
        image = preprocess_input(image,
                                 data_format=self.data_format,
                                 img_mean=self.image_mean,
                                 img_std=self.image_std,
                                 img_depth=self.image_depth,
                                 mode=self.mode,
                                 color_mode=self.model_img_mode)
        return image
