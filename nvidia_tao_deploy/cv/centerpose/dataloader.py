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

"""CenterPose Inference Dataloader."""


import cv2
import glob
import logging
import os
import sys
import json
from os.path import exists
import numpy as np
from abc import ABC

from nvidia_tao_deploy.cv.centerpose.utils import get_affine_transform


# List of valid image extensions
VALID_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "JPEG", "JPG", "PNG"]
logger = logging.getLogger(__name__)


class CPPredictDataset(ABC):
    """Base CenterPose Predict Dataset Class."""

    def __init__(self, dataset_config, inference_data, shape, dtype, # noqa pylint: disable=W0622
                 max_num_images=None, exact_batches=False, evaluate=False):
        """Initialize the CenterPose Dataset Class for TRT inference.

        Args:
            dataset_config: The CenterPose dataset default settings.
            inference_data: The inference data folder.
            shape: The input size of the TRT engine.
            dtype: The data type of the inputs.
            max_num_images: The maximum number of images to read from the directory.
            exact_batches: This defines how to handle a number of images that is not an exact
                multiple of the batch size. If false, it will pad the final batch with zeros to reach
                the batch size. If true, it will *remove* the last few images in excess of a batch size
                multiple, to guarantee batches are exact (useful for calibration).
            evaluate: This defines switching to evaluation mode, which output the json file and intrinsic matrix.

        Raises:
            FileNotFoundErorr: If provided sequence or image extension does not exist.
        """
        self.inference_data = inference_data
        self.opt = dataset_config
        self.batch_size = self.opt.batch_size
        self.evaluation = evaluate

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape

        self.mean = np.array(self.opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.ids = []
        self.images = []
        self.images += self._load_data(self.inference_data)

        self.num_images = len(self.images)
        if self.num_images < 1:
            logger.info("No valid {} images found in {}".format(VALID_IMAGE_EXTENSIONS, self.inference_data))
            sys.exit(1)

        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            raise ValueError("Not enough images to create batches")
        self.images = self.images[0:self.num_images]
        logger.info('Initializing {} inference images.'.format(len(self.images)))

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

    def _load_inference_images(self, root, extensions=VALID_IMAGE_EXTENSIONS):
        """Load the inference files inside the folder"""
        imgs = []

        def add_img_files(path, ):
            for ext in extensions:
                for imgpath in glob.glob(path + "/*.{}".format(ext.replace('.', ''))):
                    if exists(imgpath):
                        imgs.append(imgpath)

        def explore(path):
            if not os.path.isdir(path):
                return
            folders = [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
            if len(folders) > 0:
                for path_entry in folders:
                    explore(path_entry)
            else:
                add_img_files(path)

        explore(root)
        return imgs

    def _load_evaluation_images(self, root, extensions=VALID_IMAGE_EXTENSIONS):
        """Load the evaluation images inside the folder, and the related camera calibration info"""
        imgs = []

        def add_img_files(path, ):
            for ext in extensions:
                for imgpath in glob.glob(path + "/*.{}".format(ext.replace('.', ''))):
                    if exists(imgpath) and exists(imgpath.replace(ext, "json")):
                        imgs.append((imgpath, imgpath.replace(ext, "json")))

        def explore(path):
            if not os.path.isdir(path):
                return
            folders = [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
            if len(folders) > 0:
                for path_entry in folders:
                    explore(path_entry)
            else:
                add_img_files(path)

        explore(root)
        return imgs

    def _load_data(self, path):
        """Load the inference or evaluation images according to the extensions"""
        if self.evaluation:
            imgs = self._load_evaluation_images(path)
        else:
            imgs = self._load_inference_images(path)
        return imgs

    def _preprocess_img(self, img, trans_input):
        """Processing the inference image"""
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def get_batch(self):
        """Retrieve the batches.

        Returns:
            A generator yielding three items per iteration: a numpy array holding
            a batch of images, the list of paths to the images loaded within this batch,
            and the list of camera calibration info for each image in the batch.
        """
        for batch_images in self.batches:
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            batch_paths = [None] * len(batch_images)
            batch_c = np.zeros((self.batch_size, 2), dtype=self.dtype)
            batch_s = np.zeros((self.batch_size, 1), dtype=self.dtype)
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i], batch_paths[i], batch_c[i], batch_s[i] = self._get_single_processed_item(image)
            self.batch_index += 1
            batch_parameters = [batch_c, batch_s]
            yield batch_data, batch_paths, batch_parameters

    def get_evaluation_batch(self):
        """Retrieve the batches that used for evaluation.

        Returns:
            A generator yielding three items per iteration: a numpy array holding
            a batch of images, the list of paths to the images and its related json files
            loaded within this batch, and the list of camera calibration for each image in the batch.
        """
        for batch_images in self.batches:

            batch_data = np.zeros(self.shape, dtype=self.dtype)
            batch_paths = [None] * len(batch_images)
            batch_jsons = [None] * len(batch_images)
            batch_c = np.zeros((self.batch_size, 2), dtype=self.dtype)
            batch_s = np.zeros((self.batch_size, 1), dtype=self.dtype)
            batch_intrinsic = np.zeros((self.batch_size, 3, 3), dtype=self.dtype)

            for i, image_json in enumerate(batch_images):
                path_img, path_json = image_json

                # Load the camera intrinsic matrix
                with open(path_json, 'r', encoding='utf-8') as f:
                    anns = json.load(f)
                intrinsic = np.identity(3)
                intrinsic[0, 0] = anns['camera_data']['intrinsics']['fx']
                intrinsic[0, 2] = anns['camera_data']['intrinsics']['cx']
                intrinsic[1, 1] = anns['camera_data']['intrinsics']['fy']
                intrinsic[1, 2] = anns['camera_data']['intrinsics']['cy']

                batch_intrinsic[i] = intrinsic
                batch_jsons[i] = path_json
                batch_data[i], batch_paths[i], batch_c[i], batch_s[i] = self._get_single_processed_item(path_img)
                self.image_index += 1

            self.batch_index += 1
            batch_parameters = [batch_c, batch_s, batch_jsons, batch_intrinsic]
            yield batch_data, batch_paths, batch_parameters

    def _get_single_processed_item(self, img_path):
        """Yield a single image.

        Args:
            img_path: path of the image to load.
        Returns:
            inp: pre-processed image for the model.
            img_path: the loaded image path.
            c: the principle points.
            s: the maximum axis of the image.
        """
        img = cv2.imread(img_path)
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0
        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_res, self.opt.input_res])
        inp = self._preprocess_img(img, trans_input)
        return inp, img_path, c, s
