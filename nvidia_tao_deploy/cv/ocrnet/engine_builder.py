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

"""OCRNet TensorRT engine builder."""

import logging
import os

import tensorrt as trt

from nvidia_tao_deploy.engine.builder import EngineBuilder
from nvidia_tao_deploy.engine.calibrator import EngineCalibrator
from nvidia_tao_deploy.utils.image_batcher import ImageBatcher

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


class OCRNetEngineBuilder(EngineBuilder):
    """Parses an ONNX graph and builds a TensorRT engine from it."""

    def __init__(
        self,
        data_format="channels_first",
        **kwargs
    ):
        """Init.

        Args:
            data_format (str): data_format.
        """
        super().__init__(**kwargs)
        self._data_format = data_format

    def set_calibrator(self,
                       inputs=None,
                       calib_cache=None,
                       calib_input=None,
                       calib_num_images=5000,
                       calib_batch_size=8,
                       calib_data_file=None,
                       image_mean=None):
        """Simple function to set an int8 calibrator. (Default is ImageBatcher based)

        Args:
            inputs (list): Inputs to the network
            calib_input (str): The path to a directory holding the calibration images.
            calib_cache (str): The path where to write the calibration cache to,
                         or if it already exists, load it from.
            calib_num_images (int): The maximum number of images to use for calibration.
            calib_batch_size (int): The batch size to use for the calibration process.

        Returns:
            No explicit returns.
        """
        logger.info("Calibrating using ImageBatcher")

        self.config.int8_calibrator = EngineCalibrator(calib_cache)
        if not os.path.exists(calib_cache):
            calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
            calib_dtype = trt.nptype(inputs[0].dtype)
            self.config.int8_calibrator.set_image_batcher(
                ImageBatcher(calib_input, calib_shape, calib_dtype,
                             max_num_images=calib_num_images,
                             exact_batches=True,
                             preprocessor="OCRNet"))
