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

"""UNet loader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
from PIL import Image
from nvidia_tao_deploy.cv.common.constants import VALID_IMAGE_EXTENSIONS
from nvidia_tao_deploy.utils.path_utils import expand_path
import numpy as np

from abc import ABC


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    level="DEBUG")
logger = logging.getLogger(__name__)


class UNetLoader(ABC):
    """UNet Dataloader."""

    def __init__(self,
                 shape,
                 image_data_source,
                 label_data_source,
                 num_classes,
                 batch_size=10,
                 is_inference=False,
                 resize_method='bilinear',
                 preprocess="min_max_-1_1",
                 model_arch="shufflenet",
                 resize_padding=False,
                 input_image_type="color",
                 dtype=None):
        """Init.

        Args:
            image_data_source (list): list of image directories.
            label_data_source (list): list of label directories.
            num_classes (int): number of classes
            batch_size (int): size of the batch.
            is_inference (bool): If True, no labels will be returned
            resize_method (str): Bilinear / Bicubic.
            preprocess (str): A way to normalize the image tensor. (Default: min_max_-1_1)
            model_arch (str): Model architecture (Default: shufflenet).
            resize_padding (bool): Whether to resize the image with padding.
            input_image_type (str): color / grayscale.
            dtype (str): data type to cast to
        """
        self.image_paths, self.label_paths = [], []
        self.is_inference = is_inference
        self._add_source(image_data_source, label_data_source)
        self.image_paths = np.array(self.image_paths)
        self.data_inds = np.arange(len(self.image_paths))
        self.num_classes = num_classes

        self.resize_method = Image.BILINEAR if resize_method.lower() == "bilinear" else Image.BICUBIC
        self.preprocess = preprocess
        self.model_arch = model_arch
        self.resize_padding = resize_padding
        self.input_image_type = input_image_type

        # Always assume channel first
        self.num_channels, self.height, self.width = shape[0], shape[1], shape[2]

        if self.num_channels == 1 and self.input_image_type != "grayscale":
            raise ValueError("A network with channel size 1 expects grayscale input image type")

        self.batch_size = batch_size
        self.n_samples = len(self.data_inds)
        self.dtype = dtype
        self.n_batches = int(len(self.image_paths) // self.batch_size)
        assert self.n_batches > 0, "empty image dir or batch size too large!"

    def _add_source(self, image_data_source, label_data_source):
        """Add Image and Mask sources."""
        if image_data_source[0].endswith(".txt"):
            if self.is_inference:
                for imgs in image_data_source:
                    logger.debug("Reading Imgs : %s", imgs)

                    # Read image files
                    with open(imgs, encoding="utf-8") as f:
                        x_set = f.readlines()
                    for f_im in x_set:
                        # Ensuring all image files are present
                        f_im = f_im.strip()
                        if not os.path.exists(expand_path(f_im)):
                            raise FileNotFoundError(f"{f_im} does not exist!")
                        if f_im.lower().endswith(VALID_IMAGE_EXTENSIONS):
                            self.image_paths.append(f_im)
            else:
                for imgs, lbls in zip(image_data_source, label_data_source):
                    logger.debug("Reading Imgs : %s, Reading Lbls : %s", imgs, lbls)

                    # Read image files
                    with open(imgs, encoding="utf-8") as f:
                        x_set = f.readlines()

                    # Read label files
                    with open(lbls, encoding="utf-8") as f:
                        y_set = f.readlines()
                    for f_im, f_label in zip(x_set, y_set):
                        # Ensuring all image files are present
                        f_im = f_im.strip()
                        f_label = f_label.strip()
                        if not os.path.exists(expand_path(f_im)):
                            raise FileNotFoundError(f"{f_im} does not exist!")
                        if not os.path.exists(expand_path(f_label)):
                            raise FileNotFoundError(f"{f_label} does not exist!")
                        if f_im.lower().endswith(VALID_IMAGE_EXTENSIONS):
                            self.image_paths.append(f_im)
                        if f_label.lower().endswith(VALID_IMAGE_EXTENSIONS):
                            self.label_paths.append(f_label)
        else:
            self.image_paths = [os.path.join(image_data_source[0], f) for f in os.listdir(image_data_source[0])
                                if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
            if self.is_inference:
                self.label_paths = []
            else:
                self.label_paths = [os.path.join(label_data_source[0], f) for f in os.listdir(label_data_source[0])
                                    if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]

    def __len__(self):
        """Get length of Sequence."""
        return self.n_batches

    def _load_gt_image(self, image_path):
        """Load GT image from file."""
        if self.num_channels == 1:  # Set to grayscale only when channel size is 1
            img = Image.open(image_path).convert('L')
        else:
            img = Image.open(image_path).convert('RGB')

        return img

    def _load_gt_label(self, label_path):
        """Load mask labels."""
        mask = Image.open(label_path).convert("L")
        return mask

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
        image, label = self.preprocessing(image, label)
        return image, label

    def _get_single_item_raw(self, idx):
        """Load single image and its label.

        Returns:
            image (PIL.image): image object in original resolution
            label (PIL.image): Mask
        """
        image = self._load_gt_image(self.image_paths[self.data_inds[idx]])
        if self.is_inference:
            label = Image.fromarray(np.zeros(image.size)).convert("L")  # Random image to label
        else:
            label = self._load_gt_label(self.label_paths[self.data_inds[idx]])
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
        # resize based on different configs
        if self.model_arch in ["vanilla_unet"]:
            image = self.resize_image_with_crop_or_pad(image, self.height, self.width)
        else:
            if self.resize_padding:
                image = self.resize_pad(image, self.height, self.width, pad_color=(0, 0, 0))
            else:
                image = image.resize((self.width, self.height), resample=self.resize_method)
        image = np.asarray(image, dtype=self.dtype)

        # Labels should be always nearest neighbour, as they are integers.
        label = label.resize((self.width, self.height), resample=Image.NEAREST)
        label = np.asarray(label, dtype=self.dtype)

        # Grayscale can either have num_channels 1 or 3
        if self.num_channels == 1:
            image = np.expand_dims(image, axis=2)

        if self.input_image_type == "grayscale":
            label /= 255
            image = image / 127.5 - 1

        # rgb to bgr
        if self.input_image_type != "grayscale":
            image = image[..., ::-1]

        # TF1: normalize_img_tf
        if self.input_image_type != "grayscale":
            if self.preprocess == "div_by_255":
                # A way to normalize an image tensor by dividing them by 255.
                # This assumes images with max pixel value of
                # 255. It gives normalized image with pixel values in range of >=0 to <=1.
                image /= 255.0
            elif self.preprocess == "min_max_0_1":
                image /= 255.0
            elif self.preprocess == "min_max_-1_1":
                image = image / 127.5 - 1
            else:
                raise NotImplementedError(f"{self.preprocess} is not a defined method.")

        # convert to channel first
        image = np.transpose(image, (2, 0, 1))

        label = label.astype(np.uint8)
        return image, label

    def resize_image_with_crop_or_pad(self, img, target_height, target_width):
        """tf.image.resize_image_with_crop_or_pad() equivalent in Pillow.

        TF1 center crops if desired size is smaller than image size and pad with 0 if larger than image size
        Ref: https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/python/ops/image_ops_impl.py#L1251-L1405
        """
        img = self.resize_pad(img, target_height, target_width)
        img = self.center_crop(img, target_height, target_width)

        return img

    def center_crop(self, img, target_height, target_width):
        """Center Crop."""
        width, height = img.size   # Get dimensions

        # process crop width and height for max available dimension
        crop_width = target_width if target_width < width else width
        crop_height = target_height if target_height < height else height
        mid_x, mid_y = int(width / 2), int(height / 2)
        cw2, ch2 = int(crop_width / 2), int(crop_height / 2)

        left = mid_x - ch2
        top = mid_y - ch2
        right = mid_x + cw2
        bottom = mid_y + cw2

        # Crop the center of the image
        img = img.crop((left, top, right, bottom))
        return img

    def resize_pad(self, image, target_height, target_width, pad_color=(0, 0, 0)):
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
        width_scale = width / target_width
        height_scale = height / target_height

        scale = 1.0 / max(width_scale, height_scale)
        image = image.resize(
            (round(width * scale), round(height * scale)),
            resample=Image.BILINEAR)
        pad = Image.new("RGB", (target_width, target_height))
        pad.paste(pad_color, [0, 0, target_width, target_height])
        pad.paste(image)
        return pad
