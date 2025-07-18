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

import os
import cv2
from abc import ABC
import numpy as np

from nvidia_tao_deploy.cv.common.constants import VALID_IMAGE_EXTENSIONS


class LPRNetLoader(ABC):
    """LPRNet Dataloader."""

    def __init__(self,
                 shape,
                 image_dirs,
                 label_dirs,
                 classes,
                 is_inference=False,
                 batch_size=10,
                 max_label_length=8,
                 dtype=None):
        """Init.

        Args:
            image_dirs (list): list of image directories.
            label_dirs (list): list of label directories.
            classes (list): list of classes
            batch_size (int): size of the batch.
            is_inference (bool): If True, no labels will be returned
            image_mean (list): image mean used for preprocessing.
            max_label_length (int): The maximum length of license plates in the dataset
            dtype (str): data type to cast to
        """
        assert len(image_dirs) == len(label_dirs), "Mismatch!"

        self.image_paths = []
        self.label_paths = []

        self.is_inference = is_inference

        for image_dir, label_dir in zip(image_dirs, label_dirs):
            self._add_source(image_dir, label_dir)

        self.image_paths = np.array(self.image_paths)
        self.data_inds = np.arange(len(self.image_paths))

        self.class_dict = {classes[index]: index for index in range(len(classes))}
        self.classes = classes
        self.max_label_length = max_label_length

        self.num_channels, self.height, self.width = shape
        self.batch_size = batch_size
        self.n_samples = len(self.data_inds)
        self.dtype = dtype
        self.n_batches = int(len(self.image_paths) // self.batch_size)
        assert self.n_batches > 0, "empty image dir or batch size too large!"

    def _add_source(self, image_folder, label_folder):
        """Add image/label paths."""
        img_files = os.listdir(image_folder)
        if not self.is_inference:
            label_files = set(os.listdir(label_folder))
        else:
            label_files = []

        for img_file in img_files:
            file_name, _ = os.path.splitext(img_file)
            if img_file.lower().endswith(VALID_IMAGE_EXTENSIONS) and file_name + ".txt" in label_files:
                self.image_paths.append(os.path.join(image_folder,
                                                     img_file))
                self.label_paths.append(os.path.join(label_folder,
                                                     file_name + ".txt"))
            elif img_file.lower().endswith(VALID_IMAGE_EXTENSIONS) and self.is_inference:
                self.image_paths.append(os.path.join(image_folder,
                                                     img_file))

    def __len__(self):
        """Get length of Sequence."""
        return self.n_batches

    def _load_gt_image(self, image_path):
        """Load GT image from file."""
        read_flag = cv2.IMREAD_COLOR
        if self.num_channels == 1:
            read_flag = cv2.IMREAD_GRAYSCALE
        img = cv2.imread(image_path, read_flag)
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
            image (np.array): image object in original resolution
            label (int): one-hot encoded class label
        """
        image = self._load_gt_image(self.image_paths[self.data_inds[idx]])
        if self.is_inference:
            label = np.zeros((1, 7))  # Random array to label
        else:
            with open(self.label_paths[self.data_inds[idx]], "r", encoding="utf-8") as f:
                label = f.readline().strip()
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
        image = cv2.resize(image, (self.width, self.height)) / 255.0
        image = np.array(image, dtype=self.dtype)
        if self.num_channels == 1:
            image = image[..., np.newaxis]

        # transpose image from HWC (0, 1, 2) to CHW (2, 0, 1)
        image = image.transpose(2, 0, 1)
        return image
