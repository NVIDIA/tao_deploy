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

"""Classification and inference datasets loaders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
from nvidia_tao_deploy.cv.common.constants import VALID_IMAGE_EXTENSIONS


def center_crop(img, new_width, new_height):
    """Center crops the image with given width and height.

    Args:
        img (Pillow.Image): input Pillow opened image.
        new_width (int): the target width to resize to.
        new_height (int): the target height to resize to.

    Returns:
        img (Pillow.Image): output Pillow image.
    """
    width, height = img.size   # Get dimensions

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img


class MLRecogDataLoader(ABC):
    """A base class for MLRecog classification and inference data loader."""

    def __init__(self,
                 shape,
                 batch_size=10,
                 image_mean=None,
                 image_std=None,
                 image_depth=8,
                 dtype=None):
        """Init.

        Args:
            shape (list): list of input dimension that is either (c, h, w) or (h, w, c) format.
            batch_size (int): size of the batch.
            image_mean (list): image mean used for preprocessing.
            image_std (list): image std used for preprocessing.
            image_depth(int): Bit depth of images(8 or 16).
            dtype (str): data type to cast to
        """
        self.image_paths = []
        self.labels = []
        self.num_channels, self.height, self.width = shape
        self.image_depth = image_depth
        self.batch_size = batch_size
        self.n_batches = 0
        self.image_mean = image_mean if image_mean is not None else [0.0] * self.num_channels
        self.image_std = image_std if image_std is not None else [1.0] * self.num_channels
        self.model_img_mode = 'rgb' if self.num_channels == 3 else 'grayscale'
        self.dtype = dtype

    def __len__(self):
        """Get length of Sequence."""
        return self.n_batches

    def __iter__(self):
        """Iterate."""
        self.n = 0
        return self

    @abstractmethod
    def __next__(self):
        """Next object in loader"""
        pass

    def preprocessing(self, image):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.

        This includes resizing, centercrop, normalization, data type casting, and transposing.

        Args:
            image (PIL.image): The Pillow image on disk to load.

        Returns:
            image (np.array): A numpy array holding the image sample, ready to be concatenated
                              into the rest of the batch
        """
        init_size = (int(self.width * 1.14), int(self.height * 1.14))
        image = image.resize(init_size, resample=Image.BILINEAR)
        image = center_crop(image, self.width, self.height)
        image = np.asarray(image, dtype=self.dtype)

        image /= 255.
        rgb_mean = np.array(self.image_mean)
        image -= rgb_mean
        image /= np.array(self.image_std)

        # fixed to channel first
        if image.ndim == 2 and self.model_img_mode == 'grayscale':
            image = np.expand_dims(image, axis=2)
        image = np.transpose(image, (2, 0, 1))

        return image


class MLRecogClassificationLoader(MLRecogDataLoader):
    """Classification Dataloader."""

    def __init__(self,
                 shape,
                 image_dir,
                 class_mapping,
                 batch_size=10,
                 image_mean=None,
                 image_std=None,
                 image_depth=8,
                 dtype=None):
        """Initiates the classification dataloader.

        Args:
            shape (list): list of input dimension that is either (c, h, w) or (h, w, c) format.
            image_dir (str): path of input directory.
            class_mapping (dict): class mapping. e.g. {'aeroplane': 0, 'car': 1}
            batch_size (int): size of the batch.
            image_mean (list): image mean used for preprocessing.
            image_std (list): image std used for preprocessing.
            image_depth(int): Bit depth of images(8 or 16).
            dtype (str): data type to cast to
        """
        super().__init__(shape, batch_size, image_mean, image_std, image_depth, dtype)
        self.class_mapping = class_mapping
        self._add_source(image_dir)  # add both image paths and labels
        self.image_paths = np.array(self.image_paths)
        self.data_inds = np.arange(len(self.image_paths))
        self.n_samples = len(self.data_inds)
        self.n_batches = int(len(self.image_paths) // self.batch_size)
        assert self.n_batches > 0, "empty image dir or batch size too large!"

    def _add_source(self, image_folder):
        """Adds classification sources."""
        images = [p.resolve() for p in Path(image_folder).glob("**/*") if p.suffix in VALID_IMAGE_EXTENSIONS]
        images = sorted(images)
        labels = [self.class_mapping[p.parent.name] for p in images]
        self.image_paths = images
        self.labels = labels

    def _load_gt_image(self, image_path):
        """Loads GT image from file."""
        img = Image.open(image_path)
        if self.num_channels == 3:
            img = img.convert('RGB')  # Color Image
        else:
            if self.image_depth == 16:
                img = img.convert('I')  # PIL int32 mode for 16-bit images
            else:
                img = img.convert('L')  # Grayscale Image
        return img

    def __next__(self):
        """Loads a full batch."""
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
        """Posts processing for a batch."""
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
        label = self.labels[idx]
        return image, label


class MLRecogInferenceLoader(MLRecogDataLoader):
    """Inference Dataloader."""

    def __init__(self,
                 shape,
                 image_dir,
                 input_type,
                 batch_size=10,
                 image_mean=None,
                 image_std=None,
                 image_depth=8,
                 dtype=None):
        """Initiates the inference loader.

        Args:
            shape (list): list of input dimension that is either (c, h, w) or (h, w, c) format.
            image_dirs (list): list of image directories.
            input_type (str): input type of the `image_dir`. Can be [`image_folder`, `classification_folder`]
            batch_size (int): size of the batch.
            image_mean (list): image mean used for preprocessing.
            image_std (list): image std used for preprocessing.
            image_depth(int): Bit depth of images(8 or 16).
            dtype (str): data type to cast to
        """
        super().__init__(shape, batch_size, image_mean, image_std, image_depth, dtype)
        self._add_source(image_dir, input_type)
        self.image_paths = np.array(self.image_paths)
        self.data_inds = np.arange(len(self.image_paths))
        self.n_samples = len(self.data_inds)
        self.n_batches = int(len(self.image_paths) // self.batch_size)
        assert self.n_batches > 0, "empty image dir or batch size too large!"

    def _add_source(self, image_folder, input_type):
        """Add classification sources."""
        if input_type == 'classification_folder':
            images = [p.resolve() for p in Path(image_folder).glob("**/*") if p.suffix in VALID_IMAGE_EXTENSIONS]
        elif input_type == 'image_folder':
            images = [p.resolve() for p in Path(image_folder).glob("*") if p.suffix in VALID_IMAGE_EXTENSIONS]
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        images = sorted(images)
        self.image_paths = images

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

    def __next__(self):
        """Load a full batch."""
        images = []
        if self.n < self.n_batches:
            for idx in range(self.n * self.batch_size,
                             (self.n + 1) * self.batch_size):
                image = self._get_single_processed_item(idx)
                images.append(image)
            self.n += 1

            return np.array(images)
        raise StopIteration

    def _get_single_processed_item(self, idx):
        """Load and process single image and its label."""
        image = self._load_gt_image(self.image_paths[self.data_inds[idx]])
        image = self.preprocessing(image)
        return image
