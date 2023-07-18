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

"""LPRNet loader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from abc import ABC
import numpy as np
from PIL import Image

from nvidia_tao_deploy.cv.common.constants import VALID_IMAGE_EXTENSIONS
from nvidia_tao_deploy.utils.path_utils import expand_path


class OCRNetLoader(ABC):
    """OCRNet Dataloader."""

    def __init__(self,
                 shape,
                 image_dirs,
                 batch_size=10,
                 label_txts=[],
                 dtype=None):
        """Init.

        Args:
            image_dirs (list): list of image directories.
            classes (list): list of classes
            batch_size (int): size of the batch.
            label_txt (list): list of ground truth file.
            dtype (str): data type to cast to
        """
        self.image_paths = []
        self.labels = []

        if len(label_txts) != 0:
            self.is_inference = False
            assert len(label_txts) == len(image_dirs)
            for image_dir, label_txt in zip(image_dirs, label_txts):
                self._add_source(image_dir, label_txt)
        else:
            self.is_inference = True
            for image_dir in image_dirs:
                self._add_source(image_dir)

        self.image_paths = np.array(self.image_paths)
        self.data_inds = np.arange(len(self.image_paths))

        self.num_channels, self.height, self.width = shape
        self.batch_size = batch_size
        self.n_samples = len(self.data_inds)
        self.dtype = dtype
        self.n_batches = int(len(self.image_paths) // self.batch_size)
        assert self.n_batches > 0, "empty image dir or batch size too large!"

    def _add_source(self, image_folder, label_txt=None):
        """Add image/label paths."""
        if not self.is_inference:
            # label_list = open(label_txt, "r", encoding="utf-8").readlines()
            with open(label_txt, "r", encoding="utf-8") as f:
                label_list = f.readlines()
                for label_meta in label_list:
                    img_file, label = label_meta.split()
                    label = label.strip()
                    img_path = expand_path(f"{image_folder}/{img_file}")
                    if img_file.lower().endswith(VALID_IMAGE_EXTENSIONS) and os.path.exists(expand_path(img_path)):
                        self.image_paths.append(img_path)
                        self.labels.append(label)
        else:
            img_files = os.listdir(image_folder)
            for img_file in img_files:
                if img_file.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    self.image_paths.append(os.path.join(image_folder,
                                                         img_file))

    def __len__(self):
        """Get length of Sequence."""
        return self.n_batches

    def _load_gt_image(self, image_path):
        """Load GT image from file."""
        img = Image.open(image_path).convert("L")
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

        return images, labels

    def _get_single_processed_item(self, idx):
        """Load and process single image and its label."""
        image, label = self._get_single_item_raw(idx)
        image = self.preprocessing(image)
        return image, label

    def _get_single_item_raw(self, idx):
        """Load single image and its label.

        Returns:
            image (np.array): image object in original resolution
            label (int): one-hot encoded class label
        """
        image = self._load_gt_image(self.image_paths[self.data_inds[idx]])
        if self.is_inference:
            label = "NO-LABEL"  # Fake label
        else:
            label = self.labels[self.data_inds[idx]]
        return image, label

    def preprocessing(self, image):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.

        This includes padding, resizing, normalization, data type casting, and transposing.

        Args:
            image (np.array): The opencv image on disk to load.

        Returns:
            image (np.array): A numpy array holding the image sample, ready to be concatenated
                              into the rest of the batch
        """
        image = image.resize((self.width, self.height), resample=Image.BICUBIC)
        image = (np.array(image, dtype=self.dtype) / 255.0 - 0.5) / 0.5

        # @TODO(tylerz): No need to transpose for gray input
        # # transpose image from HWC (0, 1, 2) to CHW (2, 0, 1)
        # image = image.transpose(2, 0, 1)
        return image
