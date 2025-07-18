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

"""Optical Inspection loader."""

import math
import os
from abc import ABC
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from nvidia_tao_deploy.inferencer.preprocess_input import preprocess_input


class OpticalInspectionDataLoader(ABC):
    """Optical Inpsection Dataloader."""

    def __init__(
            self,
            csv_file=None,
            transform=None,
            input_data_path=None,
            train=False,
            data_config=None,
            dtype=np.float32,
            split='inference',
            batch_size=1):
        """Initialize the Optical Inspection dataloader."""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Inference data csv file wasn't found at {csv_file}")
        self.merged = pd.read_csv(csv_file)
        self.transform = transform
        self.input_image_root = input_data_path
        self.train = train
        self.num_inputs = data_config.num_input
        self.concat_type = data_config.concat_type
        self.input_map = data_config.input_map
        self.grid_map = data_config.grid_map
        self.image_width = data_config.image_width
        self.image_height = data_config.image_height
        self.data_config = data_config
        self.ext = data_config.image_ext
        self.batch_size = batch_size
        self.dtype = dtype
        self.n_batches = math.ceil(float(len(self.merged)) / self.batch_size)
        self.split = split
        assert self.n_batches > 0, (
            f"There should atleast be 1 batch to load. {self.n_batches}"
        )
        self.n = 0
        if self.concat_type == "grid":
            print(
                f"Using {self.num_inputs} input and {self.concat_type} type "
                f"{self.grid_map['x']} X {self.grid_map['y']} for comparison."
            )
        else:
            print(
                f"Using {self.num_inputs} input and {self.concat_type} type "
                f"1 X {self.num_inputs} for comparison."
            )

    def __iter__(self):
        """Initialize iterator."""
        self.n = 0
        return self

    def __next__(self):
        """Get the next image."""
        if self.n < self.n_batches:
            start_idx = self.batch_size * self.n
            unit_batch = []
            golden_batch = []
            label_batch = []
            end_idx = min(start_idx + self.batch_size, len(self.merged))
            for idx in range(start_idx, end_idx):
                if self.split == 'evaluate':
                    unit_array, golden_array, label = self.__getitem__(idx)
                    label_batch.append(label)
                else:
                    unit_array, golden_array = self.__getitem__(idx)
                unit_batch.append(unit_array)
                golden_batch.append(golden_array)
            self.n += 1
            if self.split == 'evaluate':
                return np.asarray(unit_batch, dtype=unit_array.dtype), np.asarray(golden_batch, dtype=golden_array.dtype), np.asarray(label_batch, dtype=label.dtype)

            return np.asarray(unit_batch, dtype=unit_array.dtype), np.asarray(golden_batch, dtype=golden_array.dtype)
        raise StopIteration

    def get_absolute_image_path(self, prefix, input_map=None):
        """Get absolute image path."""
        image_path = prefix
        if input_map:
            image_path += f"_{input_map}"
        image_path += self.ext
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file wasn't found at {image_path}")
        return image_path

    def __getitem__(self, index):
        """Yield a single image."""
        image_tuple = self.merged.iloc[index, :]
        image_0, image_1 = [], []

        if self.input_map:
            for input_map in self.input_map:
                image_0.append(
                    Image.open(
                        self.get_absolute_image_path(
                            self.get_unit_path(image_tuple),
                            input_map=input_map
                        )
                    )
                )
                image_1.append(
                    Image.open(
                        self.get_absolute_image_path(
                            self.get_golden_sample_path(image_tuple),
                            input_map=input_map
                        )
                    )
                )
        else:
            image_0.append(
                Image.open(
                    self.get_absolute_image_path(
                        self.get_unit_path(image_tuple)
                    )
                )
            )
            image_1.append(
                Image.open(
                    self.get_absolute_image_path(
                        self.get_golden_sample_path(image_tuple)
                    )
                )
            )

        size = (self.image_height, self.image_width)
        preprocessed_image_0 = self.preprocess_single_sample(
            image_0, size,
            self.data_config.augmentation_config.rgb_input_mean,
            self.data_config.augmentation_config.rgb_input_std,
            self.dtype
        )
        preprocessed_image_1 = self.preprocess_single_sample(
            image_1, size,
            self.data_config.augmentation_config.rgb_input_mean,
            self.data_config.augmentation_config.rgb_input_std,
            self.dtype
        )
        concatenated_unit_sample = self.concatenate_image(preprocessed_image_0)
        concatenated_golden_sample = self.concatenate_image(preprocessed_image_1)

        if self.split == 'evaluate':
            label = np.array([int(image_tuple['label'] != 'PASS')])
            return concatenated_unit_sample, concatenated_golden_sample, label

        return concatenated_unit_sample, concatenated_golden_sample

    def concatenate_image(self, preprocessed_image_array):
        """Concatenated image array from processed input.

        Args:
            preprocessed_image_array (list(PIL.Image)): List of image inputs.

        Returns:
            concatenated_image (np.ndarray): Concatenated image input.
        """
        if self.concat_type == "grid" and int(self.num_inputs) % 2 == 0:
            x, y = int(self.grid_map["x"]), int(self.grid_map["y"])
            concatenated_image = np.zeros((3, x * self.image_height, y * self.image_width))
            for idx in range(x):
                for idy in range(y):
                    concatenated_image[
                        :,
                        idx * self.image_height: (idx + 1) * self.image_height,
                        idy * self.image_width: (idy + 1) * self.image_width] = preprocessed_image_array[idx * x + idy]
        else:
            concatenated_image = np.zeros((
                3,
                self.num_inputs * self.image_height,
                self.image_width))
            for idx in range(self.num_inputs):
                concatenated_image[
                    :,
                    idx * self.image_height: self.image_height * idx + self.image_height,
                    :] = preprocessed_image_array[idx]
        return concatenated_image

    @staticmethod
    def preprocess_single_sample(image_array, output_shape, mean, std, dtype):
        """Apply pre-processing to a single image."""
        assert isinstance(output_shape, tuple), "Output shape must be a tuple."
        image_output = []
        for image in image_array:
            resized_image = cv2.resize(
                np.asarray(image, dtype),
                output_shape, interpolation=cv2.INTER_LINEAR)
            resized_image = np.transpose(resized_image, (2, 0, 1))
            resized_image = preprocess_input(
                resized_image,
                data_format="channels_first",
                img_mean=mean,
                img_std=std,
                mode="torch"
            )
            image_output.append(resized_image)
        return image_output

    def __len__(self):
        """Length of the dataloader."""
        return self.n_batches

    def get_unit_path(self, image_tuple):
        """Get path to the image file from csv."""
        image_path = os.path.join(
            self.input_image_root,
            image_tuple["input_path"],
            image_tuple["object_name"]
        )
        return image_path

    def get_golden_sample_path(self, image_tuple):
        """Get path to the corresponding golden sample."""
        image_path = os.path.join(
            self.input_image_root,
            image_tuple["golden_path"],
            image_tuple["object_name"]
        )
        return image_path
