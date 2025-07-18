
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

"""Evaluation dataloader for RT-DETR."""

import numpy as np
from PIL import Image

from nvidia_tao_deploy.cv.deformable_detr.dataloader import DDETRCOCOLoader, resize
from nvidia_tao_deploy.inferencer.preprocess_input import preprocess_input


class RTDETRCOCOLoader(DDETRCOCOLoader):
    """D-DETR DataLoader."""

    def __init__(
        self,
        image_std=None,
        img_mean=None,
        **kwargs
    ):
        """Init.

        Args:
            image_std (list): image standard deviation.
        """
        super().__init__(image_std=image_std, img_mean=img_mean, **kwargs)

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
                                 img_mean=self.img_mean,
                                 img_std=self.image_std,
                                 color_mode="rgb",
                                 mode="torch")
        return image, scale
