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

"""RetinaNet loader."""

import numpy as np
from PIL import Image

from nvidia_tao_deploy.dataloader.kitti import KITTILoader


class RetinaNetKITTILoader(KITTILoader):
    """RetinaNet Dataloader."""

    def __init__(self,
                 keep_aspect_ratio=False,
                 **kwargs):
        """Init.

        Args:
            keep_aspect_ratio (bool): keep aspect ratio of the image.
        """
        super().__init__(**kwargs)
        self.keep_aspect_ratio = keep_aspect_ratio

    def preprocessing(self, image, label):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.

        This includes padding, resizing, normalization, data type casting, and transposing.

        Args:
            image (PIL.image): The Pillow image on disk to load.
            label (np.array): labels

        Returns:
            image (np.array): A numpy array holding the image sample, ready to be concatenated
                              into the rest of the batch
            label (np.array): labels
        """

        def resize_pad(image, pad_color=(0, 0, 0)):
            """Resize and Pad.

            A subroutine to implement padding and resizing. This will resize the image to fit
            fully within the input size, and pads the remaining bottom-right portions with
            the value provided.

            Args:
                image (PIL.Image): The PIL image object
                pad_color (list): The RGB values to use for the padded area. Default: Black/Zeros.

            Returns:
                pad (PIL.Image): The PIL image object already padded and cropped,
                scale (list): the resize scale used.
            """
            width, height = image.size
            width_scale = width / self.width
            height_scale = height / self.height
            if not self.keep_aspect_ratio:
                image = image.resize(
                    (self.width, self.height),
                    resample=Image.BILINEAR)
                return image, [height_scale, width_scale]

            scale = 1.0 / max(width_scale, height_scale)
            image = image.resize(
                (round(width * scale), round(height * scale)),
                resample=Image.BILINEAR)
            if self.num_channels == 1:
                pad = Image.new("L", (self.width, self.height))
                pad.paste(0, [0, 0, self.width, self.height])
            else:
                pad = Image.new("RGB", (self.width, self.height))
                pad.paste(pad_color, [0, 0, self.width, self.height])
            pad.paste(image)
            return pad, [scale, scale]

        image, scale = resize_pad(image, (124, 116, 104))
        image = np.asarray(image, dtype=self.dtype)

        # Handle Grayscale
        if self.num_channels == 1:
            image = np.expand_dims(image, axis=2)

        label[:, 2] /= scale[1]
        label[:, 3] /= scale[0]
        label[:, 4] /= scale[1]
        label[:, 5] /= scale[0]
        # Round
        label = np.round(label, decimals=0)

        # Filter out invalid labels
        label = self._filter_invalid_labels(label)

        return image, label
