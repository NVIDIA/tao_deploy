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

"""Multitask Classification loader."""

import os
from abc import ABC

import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
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


class MClassificationLoader(ABC):
    """Multitask Classification Dataloader."""

    def __init__(self,
                 shape,
                 image_dirs,
                 label_csv,
                 batch_size=10,
                 data_format="channels_first",
                 interpolation_method="bicubic",
                 dtype=None):
        """Init.

        Args:
            image_dirs (list): list of image directories.
            label_dirs (list): list of label directories.
            interpolation_method (str): Bilinear / Bicubic.
            mode (str): caffe / torch
            crop (str): random / center
            batch_size (int): size of the batch.
            image_mean (list): image mean used for preprocessing.
            dtype (str): data type to cast to
        """
        self.image_paths = []
        self.data_df = pd.read_csv(label_csv)
        self._generate_class_mapping(self.data_df)
        self._add_source(image_dirs[0])

        self.image_paths = np.array(self.image_paths)
        self.data_inds = np.arange(len(self.image_paths))

        self.resample = Image.BILINEAR if interpolation_method == "bilinear" else Image.BICUBIC

        self.data_format = data_format
        if self.data_format == "channels_first":
            self.height, self.width = shape[1], shape[2]
        else:
            self.height, self.width = shape[0], shape[1]
        self.batch_size = batch_size
        self.n_samples = len(self.data_inds)
        self.dtype = dtype
        self.n_batches = int(len(self.image_paths) // self.batch_size)
        assert self.n_batches > 0, "empty image dir or batch size too large!"

    def _add_source(self, image_folder):
        """Add classification sources."""
        supported_img_format = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

        for filename in self.filenames:
            # Only add valid items to paths
            img_path = os.path.join(image_folder, filename)
            if filename.lower().endswith(supported_img_format) and os.path.exists(img_path):
                self.image_paths.append(os.path.join(image_folder, img_path))

    def __len__(self):
        """Get length of Sequence."""
        return self.n_batches

    def _generate_class_mapping(self, class_table):
        """Prepare task dictionary and class mapping."""
        self.filenames = class_table.iloc[:, 0].values
        self.samples = len(self.filenames)
        self.tasks_header = sorted(class_table.columns.tolist()[1:])
        self.class_dict = {}
        self.class_mapping = {}
        for task in self.tasks_header:
            unique_vals = sorted(class_table.loc[:, task].unique())
            self.class_dict[task] = len(unique_vals)
            self.class_mapping[task] = dict(zip(unique_vals, range(len(unique_vals))))

        # convert class dictionary to a sorted tolist
        self.class_dict_list_sorted = sorted(self.class_dict.items(), key=lambda x: x[0])
        self.class_mapping_inv = {}
        for key, val in self.class_mapping.items():
            self.class_mapping_inv[key] = {v: k for k, v in val.items()}

    def _load_gt_image(self, image_path):
        """Load GT image from file."""
        img = Image.open(image_path).convert('RGB')
        return img

    def __iter__(self):
        """Iterate."""
        self.n = 0
        return self

    def __next__(self):
        """Load a full batch."""
        images = []
        labels = defaultdict(list)
        if self.n < self.n_batches:
            for idx in range(self.n * self.batch_size,
                             (self.n + 1) * self.batch_size):
                image, label = self._get_single_processed_item(idx)
                images.append(image)
                for k, v in label.items():
                    labels[k].append(v)
            self.n += 1

            return self._batch_post_processing(images, labels)
        raise StopIteration

    def _batch_post_processing(self, images, labels):
        """Post processing for a batch."""
        images = np.array(images)
        final_labels = []
        for _, v in labels.items():
            final_labels.append(np.array(v))
        return images, final_labels

    def _get_single_processed_item(self, idx):
        """Load and process single image and its label."""
        image, label = self._get_single_item_raw(idx)
        image, label = self.preprocessing(image, label)
        return image, label

    def _get_single_item_raw(self, idx):
        """Load single image and its label.

        Returns:
            image (PIL.image): image object in original resolution
            label (int): one-hot encoded class label
        """
        image = self._load_gt_image(self.image_paths[self.data_inds[idx]])
        label = self.data_df.iloc[idx, :]
        return image, label

    def preprocessing(self, image, label):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.

        This includes padding, resizing, normalization, data type casting, and transposing.

        Args:
            image (PIL.image): The Pillow image on disk to load.

        Returns:
            image (np.array): A numpy array holding the image sample, ready to be concatenated
                              into the rest of the batch
        """
        image = image.resize((self.width, self.height), self.resample)
        image = np.asarray(image, dtype=self.dtype)

        if self.data_format == "channels_first":
            image = np.transpose(image, (2, 0, 1))

        # Normalize and apply imag mean and std
        image = preprocess_input(image, data_format=self.data_format)

        # one-hot-encoding
        batch_y = {}
        for c, cls_cnt in self.class_dict_list_sorted:
            batch_y[c] = np.zeros(cls_cnt, dtype=self.dtype)

        for c, _ in self.class_dict_list_sorted:
            batch_y[c][self.class_mapping[c][label[c]]] = 1
        return image, batch_y
