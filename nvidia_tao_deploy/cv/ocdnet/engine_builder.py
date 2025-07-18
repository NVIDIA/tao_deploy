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

import tensorrt as trt

from nvidia_tao_deploy.engine.builder import TRT_8_API, EngineBuilder
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

    def create_network(self, model_path, file_format="onnx"):
        """Parse the UFF/ONNX graph and create the corresponding TensorRT network definition.

        Args:
            model_path: The path to the UFF/ONNX graph to load.
            file_format: The file format of the decrypted etlt file (default: onnx).
        """
        if file_format == "onnx":
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            if TRT_8_API:
                network_flags = network_flags | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION))

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

            logger.info("Parsing ONNX model")
            # input_dims are a dict {name: shape}
            input_dims = self.get_onnx_input_dims(inputs)
            batch_sizes = {v[0] for v in input_dims.values()}
            assert len(batch_sizes) == 1, (
                "All tensors should have the same batch size."
            )
            self.batch_size = list(batch_sizes)[0]
            # self._input_dims = {}
            # for k, v in input_dims.items():
            #     self._input_dims[k] = v[1:]

            logger.info("Network Description")
            opt_profile = self.builder.create_optimization_profile()
            for model_input in inputs: # noqa pylint: disable=W0622
                logger.info("Input '%s' with shape %s and dtype %s", model_input.name, model_input.shape, model_input.dtype)
                input_shape = model_input.shape
                input_name = model_input.name
                if self.batch_size <= 0:
                    # @seanf: for some reason, the .onnx model in scratch space has -1 for width and height,
                    # which is why we take in these vals upon construction
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
                else:
                    shape = (self.batch_size, input_shape[1], self.height, self.width)
                    opt_profile.set_shape(
                        input=input_name,
                        min=shape,
                        opt=shape,
                        max=shape
                    )

            self.config.add_optimization_profile(opt_profile)
            self.config.set_calibration_profile(opt_profile)

            for output in outputs:
                logger.info("Output '%s' with shape %s and dtype %s", output.name, output.shape, output.dtype)

        else:
            raise NotImplementedError(f"{file_format.capitalize()} backend is not supported for D-DETR.")

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
