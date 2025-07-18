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

"""EfficientDet loader."""

import numpy as np
from PIL import Image

from nvidia_tao_deploy.dataloader.coco import COCOLoader


class EfficientDetCOCOLoader(COCOLoader):
    """EfficientDet DataLoader."""

    def preprocess_image(self, image_path):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.

        This includes padding, resizing, normalization, data type casting, and transposing.
        This Image Batcher implements one algorithm for now:
        * EfficientDet: Resizes and pads the image to fit the input size.

        Args:
            image_path(str): The path to the image on disk to load.

        Returns:
            image (np.array): A numpy array holding the image sample, ready to be concatenated
                              into the rest of the batch
            scale (list): the resize scale used, if any.
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
            scale = 1.0 / max(width_scale, height_scale)
            image = image.resize(
                (round(width * scale), round(height * scale)),
                resample=Image.BILINEAR)
            pad = Image.new("RGB", (self.width, self.height))
            pad.paste(pad_color, [0, 0, self.width, self.height])
            pad.paste(image)
            return pad, scale

        scale = None
        image = Image.open(image_path)
        image = image.convert(mode='RGB')
        # For EfficientNet V2: Resize & Pad with ImageNet mean values
        # and keep as [0,255] Normalization
        image, scale = resize_pad(image, (124, 116, 104))
        image = np.asarray(image, dtype=self.dtype)
        # [0-1] Normalization, Mean subtraction and Std Dev scaling are
        # part of the EfficientDet graph, so no need to do it during preprocessing here

        if self.data_format == "channels_first":
            image = np.transpose(image, (2, 0, 1))
        return image, scale
