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

"""Util class for creating image batches."""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import random
from omegaconf import ListConfig

from nvidia_tao_deploy.cv.common.constants import VALID_IMAGE_EXTENSIONS, VALID_GT_EXTENSIONS
from nvidia_tao_deploy.inferencer.preprocess_input import preprocess_input
from nvidia_tao_deploy.cv.depth_net.utils import read_gt_depth
from nvidia_tao_deploy.cv.deformable_detr.dataloader import resize
from nvidia_tao_deploy.cv.ml_recog.dataloader import center_crop


class ImageBatcher:
    """Creates batches of pre-processed images."""

    def __init__(self, input, shape, dtype,  # noqa pylint: disable=W0622
                 right_input=None,
                 gt_input=None,
                 max_num_images=None, exact_batches=False, preprocessor="EfficientDet",
                 img_std=[0.229, 0.224, 0.225],
                 img_mean=[0.485, 0.456, 0.406],
                 evaluation=False):
        """Initialize.

        Args:
            input: The input directory to read images from. (list or str)
            shape: The tensor shape of the batch to prepare, either in channels_first or channels_last format.
            dtype: The (numpy) datatype to cast the batched data to.
            max_num_images: The maximum number of images to read from the directory.
            exact_batches: This defines how to handle a number of images that is not an exact
                multiple of the batch size. If false, it will pad the final batch with zeros to reach
                the batch size. If true, it will *remove* the last few images in excess of a batch size
                multiple, to guarantee batches are exact (useful for calibration).
            preprocessor: Set the preprocessor to use, depending on which network is being used.
            img_std: Set img std for DDETR use case
            img_mean: Set img mean for DDETR use case
            evaluation: Set to True for evaluation mode.
        """
        self.images = []
        self.right_images = []
        self.gt_images = []
        self.evaluation = evaluation

        def is_image(path):
            return os.path.isfile(path) and path.lower().endswith(VALID_IMAGE_EXTENSIONS)

        if isinstance(input, (ListConfig, list)):
            # Multiple directories
            for image_dir in input:
                self.images.extend(str(p.resolve()) for p in Path(image_dir).glob("**/*") if p.suffix in VALID_IMAGE_EXTENSIONS)
            if right_input is None and gt_input is None and not self.evaluation:
                # Shuffle so that we sample uniformly from the sequence
                random.shuffle(self.images)
        else:
            if os.path.isdir(input):
                self.images = [str(p.resolve()) for p in Path(input).glob("**/*") if p.suffix in VALID_IMAGE_EXTENSIONS]
                self.images.sort()
            elif os.path.isfile(input):
                if is_image(input):
                    self.images.append(input)

        if right_input is not None:
            if isinstance(right_input, (ListConfig, list)):
                # Multiple directories
                for image_dir in right_input:
                    self.right_images.extend(str(p.resolve()) for p in Path(image_dir).glob("**/*") if p.suffix in VALID_IMAGE_EXTENSIONS)
            else:
                if os.path.isdir(right_input):
                    self.right_images = [str(p.resolve()) for p in Path(right_input).glob("**/*") if p.suffix in VALID_IMAGE_EXTENSIONS]
                    self.right_images.sort()
                elif os.path.isfile(right_input):
                    if is_image(right_input):
                        self.right_images.append(right_input)
            # raise error for different number of images and right images
            if len(self.right_images) != len(self.images):
                raise ValueError(f"Number of right images ({len(self.right_images)}) does not match number of images ({len(self.images)})")

        if gt_input is not None and self.evaluation:
            if isinstance(gt_input, (ListConfig, list)):
                for image_dir in gt_input:
                    self.gt_images.extend(str(p.resolve()) for p in Path(image_dir).glob("**/*") if p.suffix in VALID_GT_EXTENSIONS)
            else:
                if os.path.isdir(gt_input):
                    self.gt_images = [str(p.resolve()) for p in Path(gt_input).glob("**/*") if p.suffix in VALID_GT_EXTENSIONS]
                    self.gt_images.sort()
                elif os.path.isfile(gt_input):
                    if is_image(gt_input):
                        self.gt_images.append(gt_input)

            # raise error for different number of images and ground truth images
            if len(self.gt_images) != len(self.images):
                raise ValueError(f"Number of ground truth images ({len(self.gt_images)}) does not match number of images ({len(self.images)})")

            if len(self.gt_images) == 0:
                raise ValueError(f"No ground truth images found at the given path {gt_input}")

        self.num_images = len(self.images)
        if self.num_images < 1:
            print(f"No valid {'/'.join(VALID_IMAGE_EXTENSIONS)} images found in {input}")
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        self.channel = -1
        if self.shape[1] in [3, 1]:
            self.format = "channels_first"
            self.height = self.shape[2]
            self.width = self.shape[3]
            self.channel = self.shape[1]
        elif self.shape[3] in [3, 1]:
            self.format = "channels_last"
            self.height = self.shape[1]
            self.width = self.shape[2]
            self.channel = self.shape[3]
        assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            raise ValueError("Not enough images to create batches")
        self.images = self.images[0:self.num_images]

        if right_input is not None:
            self.right_images = self.right_images[0:self.num_images]

        if gt_input is not None and self.evaluation:
            self.gt_images = self.gt_images[0:self.num_images]

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        self.gt_batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            if right_input is not None:
                self.batches.append((self.images[start:end], self.right_images[start:end]))
            else:
                self.batches.append(self.images[start:end])

            if self.evaluation:
                self.gt_batches.append(self.gt_images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

        self.preprocessor = preprocessor
        self.img_std = img_std
        self.img_mean = img_mean

    def preprocess_image(self, image_path):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.

        This includes padding, resizing, normalization, data type casting, and transposing.
        This Image Batcher implements few algorithms for now:
        * EfficientDet: Resizes and pads the image to fit the input size.
        * MRCNN: Resizes, pads, and normalizes the image to fit the input size.
        * DetectNetv2: Resizes and normalizes the image to fit the input size.

        Args:
            image_path (str): The path to the image on disk to load.

        Returns:
            Two values: A numpy array holding the image sample, ready to be contacatenated
                into the rest of the batch, and the resize scale used, if any.
        """

        def resize_pad(image, pad_color=(0, 0, 0)):
            """Resize and Pad.

            A subroutine to implement padding and resizing. This will resize the image to fit
            fully within the input size, and pads the remaining bottom-right portions with
            the value provided.

            Args:
                image: The PIL image object
                pad_color: The RGB values to use for the padded area. Default: Black/Zeros.

            Returns
                Two values: The PIL image object already padded and cropped,
                    and the resize scale used.
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
        if self.preprocessor == "EfficientDet":
            # For EfficientNet V2: Resize & Pad with ImageNet mean values
            # and keep as [0,255] Normalization
            image, scale = resize_pad(image, (124, 116, 104))
            image = np.asarray(image, dtype=self.dtype)
            # [0-1] Normalization, Mean subtraction and Std Dev scaling are
            # part of the EfficientDet graph, so no need to do it during preprocessing here
        elif self.preprocessor == "Mask2former":
            orig_w, orig_h = image.size
            image = image.resize((self.width, self.height), Image.BICUBIC)
            image = np.asarray(image, dtype=self.dtype)
            image = preprocess_input(
                image,
                data_format='channels_last',
                img_mean=self.img_mean,
                img_std=self.img_std,
                mode='torch')
            new_h, new_w, _ = image.shape
            scale = (orig_h / new_h, orig_w / new_w)
        elif self.preprocessor == "DetectNetv2":
            image = image.resize((self.width, self.height), Image.LANCZOS)
            image = np.asarray(image, dtype=self.dtype)

            image = image / 255.0
            scale = 1.0
        elif self.preprocessor == "MRCNN":
            image, scale = resize_pad(image, (124, 116, 104))
            image = np.asarray(image, dtype=self.dtype)
            image = preprocess_input(image, data_format="channels_last", mode="torch")
        elif self.preprocessor == 'DDETR':
            image = np.asarray(image, dtype=self.dtype)
            orig_h, orig_w, _ = image.shape
            image, _ = resize(image, None, size=(self.height, self.width))

            image = preprocess_input(image,
                                     data_format='channels_last',
                                     img_std=self.img_std,
                                     mode='torch')
            new_h, new_w, _ = image.shape
            scale = (orig_h / new_h, orig_w / new_w)
        elif self.preprocessor == 'RTDETR':
            image = np.asarray(image, dtype=self.dtype)
            orig_h, orig_w, _ = image.shape
            image, _ = resize(image, None, size=(self.height, self.width))

            image = preprocess_input(image,
                                     data_format='channels_last',
                                     img_std=self.img_std,
                                     img_mean=self.img_mean,
                                     mode='torch')
            new_h, new_w, _ = image.shape
            scale = (orig_h / new_h, orig_w / new_w)
        elif self.preprocessor == "OCDNet":
            image = image.resize((self.width, self.height), Image.LANCZOS)
            rgb_mean = np.array([122.67891434, 116.66876762, 104.00698793])
            image -= rgb_mean
            image /= 255.
        elif self.preprocessor == "MLRecog":
            init_size = (int(self.width * 1.14), int(self.height * 1.14))
            image = image.resize(init_size, Image.BILINEAR)
            image = center_crop(image, self.width, self.height)
            image = np.asarray(image, dtype=self.dtype)
            image = preprocess_input(image,
                                     data_format='channels_first',
                                     img_mean=self.img_mean,
                                     img_std=self.img_std,
                                     mode='torch')
        elif self.preprocessor == "OCRNet":
            image = image.convert("L")
            image = image.resize((self.width, self.height), resample=Image.BICUBIC)
            image = (np.array(image, dtype=self.dtype) / 255.0 - 0.5) / 0.5
        elif self.preprocessor == "DepthNet":
            image = np.asarray(image, dtype=self.dtype)
            orig_h, orig_w, _ = image.shape
            image, _ = resize(image, None, size=(self.height, self.width))
            image = preprocess_input(
                image,
                data_format='channels_first',
                img_mean=self.img_mean,
                img_std=self.img_std,
                mode='torch')
            new_h, new_w, _ = image.shape
            scale = (orig_h / new_h, orig_w / new_w)
        else:
            raise NotImplementedError(f"Preprocessing method {self.preprocessor} not supported")
        if self.format == "channels_first":
            if self.channel != 1:
                image = np.transpose(image, (2, 0, 1))
        return image, scale

    def preprocess_depth(self, depth_path):
        """Preprocess the depth image."""
        depth = read_gt_depth(depth_path)
        return depth

    def get_batch(self):
        """Retrieve the batches.

        This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.

        Returns:
            A generator yielding three items per iteration: a numpy array holding
            a batch of images, the list of paths to the images loaded within this batch,
            and the list of resize scales for each image in the batch.
        """
        for index, batch_images in enumerate(self.batches):
            # stereo input - input image is tuple (left, right)
            if isinstance(batch_images, tuple):
                left_images, right_images = batch_images
                batch_data = {"left_image": np.zeros(self.shape, dtype=self.dtype), "right_image": np.zeros(self.shape, dtype=self.dtype)}
                batch_scales = [None] * len(left_images)
                batch_gt = []
                for i, (left_image, right_image) in enumerate(zip(left_images, right_images)):
                    self.image_index += 1
                    batch_data["left_image"][i], batch_scales[i] = self.preprocess_image(left_image)
                    batch_data["right_image"][i], _ = self.preprocess_image(right_image)
                    if self.evaluation:
                        batch_gt.append(read_gt_depth(self.gt_batches[index][i]))
                self.batch_index += 1
                if self.evaluation:
                    yield batch_data, left_images, batch_scales, np.array(batch_gt)
                else:
                    yield batch_data, left_images, batch_scales
            else:
                # mono input
                batch_data = np.zeros(self.shape, dtype=self.dtype)
                batch_gt = []
                batch_scales = [None] * len(batch_images)
                for i, image in enumerate(batch_images):
                    self.image_index += 1
                    batch_data[i], batch_scales[i] = self.preprocess_image(image)
                    if self.evaluation:
                        batch_gt.append(read_gt_depth(self.gt_batches[index][i]))
                self.batch_index += 1
                if self.evaluation:
                    yield batch_data, batch_images, batch_scales, np.array(batch_gt)
                else:
                    yield batch_data, batch_images, batch_scales
