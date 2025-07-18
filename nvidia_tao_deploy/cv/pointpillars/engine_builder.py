# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""PointPillars TensorRT engine builder."""

import logging
import os
import sys

import tensorrt as trt

from nvidia_tao_deploy.engine.builder import TRT_8_API, EngineBuilder

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


class PointPillarsEngineBuilder(EngineBuilder):
    """Parses an UFF/ONNX graph and builds a TensorRT engine from it."""

    def __init__(
        self,
        batch_size=None,
        **kwargs
    ):
        """Init.

        Args:
            data_format (str): data_format.
        """
        super().__init__(batch_size=batch_size, **kwargs)

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
            logger.info("Network Description")
            opt_profile = self.builder.create_optimization_profile()
            for model_input in inputs: # noqa pylint: disable=W0622
                logger.info("Input '%s' with shape %s and dtype %s", model_input.name, model_input.shape, model_input.dtype)
                input_shape = model_input.shape
                input_name = model_input.name
                if input_name == "points":
                    shape = (self.batch_size, input_shape[1], input_shape[2])
                elif input_name == "num_points":
                    shape = (self.batch_size,)
                else:
                    raise ValueError(f"Unsupported input name: {input_name}")
                print(f"Input profile: {input_name}:{shape}")
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
