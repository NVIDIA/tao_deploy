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

"""Visual ChangeNet-Segmentation loader."""

import os
import cv2
import numpy as np
from abc import ABC
from PIL import Image

from nvidia_tao_deploy.inferencer.preprocess_input import preprocess_input


class ChangeNetDataLoader(ABC):
    """Visual ChangeNet Dataloader."""

    def __init__(
            self,
            dataset_config=None,
            dtype=np.float32,
            mode='test',
            split=None):
        """Initialize the Visual ChangeNet dataloader."""
        self.dataset_config = dataset_config
        self.batch_size = dataset_config["batch_size"]
        self.num_workers = dataset_config["workers"]
        self.root_dir = dataset_config["root_dir"]
        self.label_transform = dataset_config["label_transform"]
        self.img_size = dataset_config["img_size"]
        self.dataset = dataset_config["dataset"]
        self.image_folder_name = dataset_config["image_folder_name"]
        self.change_image_folder_name = dataset_config["change_image_folder_name"]
        self.list_folder_name = dataset_config["list_folder_name"]
        self.annotation_folder_name = dataset_config["annotation_folder_name"]
        self.augmentation = dataset_config["augmentation"]
        self.label_suffix = dataset_config["label_suffix"]
        self.split = split
        self.mean = self.augmentation['mean']
        self.std = self.augmentation['std']
        self.mode = mode

        self.list_path = os.path.join(self.root_dir, self.list_folder_name, self.split + '.txt')
        self.img_name_list = self.load_img_name_list(self.list_path)
        self.n_batches = len(self.img_name_list) // self.batch_size
        self.dtype = dtype
        self.A_size = len(self.img_name_list)
        assert self.n_batches > 0, "empty image dir or batch size too large!"

    def get_img_path(self, root_dir, img_name, folder_name):
        """Get the full path of an image given its filename and folder name."""
        return os.path.join(root_dir, folder_name, img_name)

    def get_label_path(self, root_dir, img_name, folder_name, label_suffix):
        """Get the full path of a label image given its filename and folder name."""
        return os.path.join(root_dir, folder_name, img_name.replace('.jpg', label_suffix))

    def load_img_name_list(self, dataset_path):
        """Load the list of image filenames from a given file."""
        img_name_list = np.loadtxt(dataset_path, dtype=str)
        if img_name_list.ndim == 2:
            return img_name_list[:, 0]
        return img_name_list

    def __iter__(self):
        """Initialize iterator."""
        self.n = 0
        return self

    def __next__(self):
        """Get the next image."""
        if self.n < self.n_batches:
            start_idx = self.batch_size * self.n
            input_1 = []
            input_2 = []
            label = []
            end_idx = (self.n + 1) * self.batch_size
            for idx in range(start_idx, end_idx):
                if self.mode != 'predict':
                    array1, array2, target = self.__getitem__(idx)
                    label.append(target)
                else:
                    array1, array2 = self.__getitem__(idx)
                input_1.append(array1)
                input_2.append(array2)
            self.n += 1
            if self.mode != 'predict':
                return np.asarray(input_1, dtype=array1.dtype), np.asarray(input_2, dtype=array2.dtype), np.asarray(label, dtype=array1.dtype)
            return np.asarray(input_1, dtype=array1.dtype), np.asarray(input_2, dtype=array2.dtype)

        raise StopIteration

    def __getitem__(self, index):
        """Yield a single image."""
        A_path = self.get_img_path(self.root_dir, self.img_name_list[index % self.A_size], self.image_folder_name)
        B_path = self.get_img_path(self.root_dir, self.img_name_list[index % self.A_size], self.change_image_folder_name)
        # Load with PIL and then resize and convert to numpy
        img_pil = Image.open(A_path).convert('RGB')
        img_B_pil = Image.open(B_path).convert('RGB')

        preprocessed_image_0 = self.preprocess_single_sample(
            img_pil, self.img_size,
            mean=self.mean,
            std=self.std,
            dtype=self.dtype
        )
        preprocessed_image_1 = self.preprocess_single_sample(
            img_B_pil, self.img_size,
            mean=self.mean,
            std=self.std,
            dtype=self.dtype
        )
        preprocessed_image_0 = np.asarray(preprocessed_image_0)
        preprocessed_image_1 = np.asarray(preprocessed_image_1)

        if self.mode != 'predict':
            L_path = self.get_label_path(self.root_dir, self.img_name_list[index % self.A_size], self.annotation_folder_name, self.label_suffix)
            label = Image.open(L_path)
            label = self.preprocess_single_sample(
                label, self.img_size,
                mean=None,
                std=None,
                dtype=np.uint8,
                norm=False  # No normalize using mean/std
            )
            if self.label_transform == 'norm':
                label = label // 255
            return preprocessed_image_0, preprocessed_image_1, label
        return preprocessed_image_0, preprocessed_image_1

    @staticmethod
    def preprocess_single_sample(image_array, output_shape, mean, std, dtype, norm=True):
        """Apply pre-processing to a single image."""
        # assert isinstance(output_shape, tuple), "Output shape must be a tuple."
        resized_image = cv2.resize(
            np.asarray(image_array, dtype),
            (output_shape, output_shape), interpolation=cv2.INTER_LINEAR)
        if norm:
            resized_image = np.transpose(resized_image, (2, 0, 1))
            resized_image = preprocess_input(
                resized_image,
                data_format="channels_first",
                img_mean=mean,
                img_std=std,
                mode="torch"
            )
        else:
            resized_image = np.expand_dims(resized_image, axis=2)
        return resized_image

    def __len__(self):
        """Length of the dataloader."""
        return self.n_batches
