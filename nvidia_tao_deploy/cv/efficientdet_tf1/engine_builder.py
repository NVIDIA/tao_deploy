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

"""EfficientDet TensorRT engine builder."""

import logging
import os
import sys

import tensorrt as trt

from nvidia_tao_deploy.engine.builder import TRT_8_API, EngineBuilder

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


class EfficientDetEngineBuilder(EngineBuilder):
    """Parses an ONNX graph and builds a TensorRT engine from it."""

    def create_network(self, model_path, file_format="onnx"):
        """Parse the ONNX graph and create the corresponding TensorRT network definition.

        Args:
            model_path: The path to the ONNX graph to load.
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
            self._input_dims = {}
            for k, v in input_dims.items():
                self._input_dims[k] = v[1:]

            logger.info("Network Description")
            for input in inputs: # noqa pylint: disable=W0622
                self.batch_size = input.shape[0]
                logger.info("Input '%s' with shape %s and dtype %s", input.name, input.shape, input.dtype)
            for output in outputs:
                logger.info("Output '%s' with shape %s and dtype %s", output.name, output.shape, output.dtype)

            # TF1 EfficientDet only support static batch size
            assert self.batch_size > 0
        else:
            logger.info("Parsing UFF model")
            raise NotImplementedError("UFF for EfficientDet is not supported")
