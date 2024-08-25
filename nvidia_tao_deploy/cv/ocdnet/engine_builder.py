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

"""OCDNet TensorRT engine builder."""

import logging
import os
import sys
import onnx

import tensorrt as trt

from nvidia_tao_deploy.engine.builder import EngineBuilder
from nvidia_tao_deploy.engine.calibrator import EngineCalibrator
from nvidia_tao_deploy.utils.image_batcher import ImageBatcher

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


class OCDNetEngineBuilder(EngineBuilder):
    """Parses an UFF/ONNX graph and builds a TensorRT engine from it."""

    def __init__(
        self,
        width,
        height,
        img_mode,
        batch_size=None,
        data_format="channels_first",
        **kwargs
    ):
        """Init.

        Args:
            data_format (str): data_format.
        """
        super().__init__(batch_size=batch_size, **kwargs)
        self._data_format = data_format
        self.width = width
        self.height = height
        self.img_mode = img_mode

    def get_onnx_input_dims(self, model_path):
        """Get input dimension of ONNX model."""
        onnx_model = onnx.load(model_path)
        onnx_inputs = onnx_model.graph.input
        for i, inputs in enumerate(onnx_inputs):
            return [i.dim_value for i in inputs.type.tensor_type.shape.dim][:]

    def create_network(self, model_path, file_format="onnx"):
        """Parse the UFF/ONNX graph and create the corresponding TensorRT network definition.

        Args:
            model_path: The path to the UFF/ONNX graph to load.
            file_format: The file format of the decrypted etlt file (default: onnx).
        """
        if file_format == "onnx":
            logger.info("Parsing ONNX model")
            self._input_dims = self.get_onnx_input_dims(model_path)
            self.batch_size = self._input_dims[0]

            network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

            self.network = self.builder.create_network(network_flags)
            self.parser = trt.OnnxParser(self.network, self.trt_logger)

            model_path = os.path.realpath(model_path)
            if not self.parser.parse_from_file(model_path):
                logger.error("Failed to load ONNX file: %s", model_path)
                for error in range(self.parser.num_errors):
                    logger.error(self.parser.get_error(error))
                sys.exit(1)

            inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
            outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

            logger.info("Network Description")
            for input in inputs: # noqa pylint: disable=W0622
                logger.info("Input '%s' with shape %s and dtype %s", input.name, input.shape, input.dtype)
            for output in outputs:
                logger.info("Output '%s' with shape %s and dtype %s", output.name, output.shape, output.dtype)

            if self.batch_size <= 0:  # dynamic batch size
                logger.info("dynamic batch size handling")
                opt_profile = self.builder.create_optimization_profile()
                model_input = self.network.get_input(0)
                input_shape = model_input.shape
                input_name = model_input.name
                real_shape_min = (self.min_batch_size, input_shape[1],
                                  self.height, self.width)
                real_shape_opt = (self.opt_batch_size, input_shape[1],
                                  self.height, self.width)
                real_shape_max = (self.max_batch_size, input_shape[1],
                                  self.height, self.width)
                opt_profile.set_shape(input=input_name,
                                      min=real_shape_min,
                                      opt=real_shape_opt,
                                      max=real_shape_max)
                self.config.add_optimization_profile(opt_profile)
        else:
            logger.info("Parsing UFF model")
            raise NotImplementedError("UFF for OCDNet is not supported")

    def set_calibrator(self,
                       inputs=None,
                       calib_cache=None,
                       calib_input=None,
                       calib_num_images=1,
                       calib_batch_size=8,
                       calib_data_file=None,
                       image_mean=None):
        """Simple function to set an Tensorfile based int8 calibrator.

        Args:
            calib_input: The path to a directory holding the calibration images.
            calib_cache: The path where to write the calibration cache to,
                         or if it already exists, load it from.
            calib_num_images: The maximum number of images to use for calibration.
            calib_batch_size: The batch size to use for the calibration process.
            image_mean: Image mean per channel.

        Returns:
            No explicit returns.
        """
        if not calib_data_file:
            logger.info("Calibrating using ImageBatcher")
            self.config.int8_calibrator = EngineCalibrator(calib_cache)
            if not os.path.exists(calib_cache):
                calib_shape = [calib_batch_size] + [self.network.get_input(0).shape[1]] + [self.height] + [self.width]
                calib_dtype = trt.nptype(inputs[0].dtype)
                logger.info(calib_shape)
                logger.info(calib_dtype)
                self.config.int8_calibrator.set_image_batcher(
                    ImageBatcher(calib_input, calib_shape, calib_dtype,
                                 max_num_images=calib_num_images,
                                 exact_batches=True,
                                 preprocessor="OCDNet"))
