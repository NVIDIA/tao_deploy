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

"""Base TensorRT engine calibrator."""

import logging
import os

import numpy as np
import pycuda.autoinit # noqa pylint: disable=unused-import
import pycuda.driver as cuda
import tensorrt as trt

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """Implements the INT8 Entropy Calibrator2."""

    def __init__(self, cache_file):
        """Init.

        Args:
            cache_file (str): The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher):
        """Define the image batcher to use, if any.

        If using only the cache file, an image batcher doesn't need to be defined.

        Args:
            image_batcher (obj): The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
        self.batch_allocation = cuda.mem_alloc(size)
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """Overrides from trt.IInt8EntropyCalibrator2.

        Get the batch size to use for calibration.

        Returns:
            Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """Overrides from trt.IInt8EntropyCalibrator2.

        Get the next batch to use for calibration, as a list of device memory pointers.

        Args:
            names (list): The names of the inputs, if useful to define the order of inputs.

        Returns:
            A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, _ = next(self.batch_generator)
            logger.info("Calibrating image %d / %d",
                        self.image_batcher.image_index, self.image_batcher.num_images)
            cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            logger.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """Overrides from trt.IInt8EntropyCalibrator2.

        Read the calibration cache file stored on disk, if it exists.

        Returns:
            The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache file: %s", self.cache_file)
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Overrides from trt.IInt8EntropyCalibrator2.

        Store the calibration cache to a file on disk.

        Args
            cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            logger.info("Writing calibration cache data to: %s", self.cache_file)
            f.write(cache)
