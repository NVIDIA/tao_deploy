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

"""Segformer loader."""

from PIL import Image
import logging
import numpy as np

from nvidia_tao_deploy.cv.segformer.utils import imrescale, impad
from nvidia_tao_deploy.cv.unet.dataloader import UNetLoader


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    level="DEBUG")
logger = logging.getLogger(__name__)


class SegformerLoader(UNetLoader):
    """Segformer Dataloader."""

    def __init__(self,
                 keep_ratio=True,
                 pad_val=0,
                 image_mean=None,
                 image_std=None,
                 target_classes=None,
                 label_transform=None,
                 **kwargs):
        """Init.

        Args:
            keep_ratio (bool): To keep the aspect ratio of image (padding will be used).
            pad_val (int): Per-channel pixel value to pad for input image.
            image_mean (list): image mean.
            image_std (list): image standard deviation.
            target_classes (list): List of TargetClass instances.
            label_transform (str): Label transform type, "norm" or others, norm means divide by 255.
        """
        super().__init__(**kwargs)
        self.pad_val = pad_val
        self.keep_ratio = keep_ratio
        self.image_mean = image_mean
        self.image_std = image_std
        self.target_classes = target_classes
        self.label_transform = label_transform

    def preprocessing(self, image, label):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.

        This includes padding, resizing, normalization, data type casting, and transposing.

        Args:
            image (PIL.image): The Pillow image on disk to load.

        Returns:
            image (np.array): A numpy array holding the image sample, ready to be concatenated
                              into the rest of the batch
        """
        if self.keep_ratio:
            # mmcv style resize for image
            image = np.asarray(image)
            image = imrescale(image, (self.width, self.height))
            image, _ = impad(image, shape=(self.height, self.width), pad_val=self.pad_val)
            image = image.astype(self.dtype)
        else:
            image = image.resize((self.width, self.height), Image.BILINEAR)
            image = np.asarray(image).astype(self.dtype)

        # Segformer does not follow regular PyT preprocessing. No divide by 255
        for i in range(len(self.image_mean)):
            image[..., i] -= self.image_mean[i]
            image[..., i] /= self.image_std[i]
        image = np.transpose(image, (2, 0, 1))

        if self.keep_ratio:
            label = np.asarray(label)
            label = imrescale(label, (self.width, self.height), interpolation='nearest')
            # We always pad with 0 for labels
            label, _ = impad(label, shape=(self.height, self.width), pad_val=0)
        else:
            label = label.resize((self.width, self.height), Image.BILINEAR)
            label = np.asarray(label)

        # Convert label to train id
        if self.label_transform == "norm":
            label = label / 255

        if self.target_classes:
            for target_class in self.target_classes:
                label[label == target_class.label_id] = target_class.train_id

        label = label.astype(np.uint8)
        return image, label
