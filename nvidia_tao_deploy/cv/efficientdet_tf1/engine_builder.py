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
import onnx

import tensorrt as trt

from nvidia_tao_deploy.engine.builder import EngineBuilder

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


class EfficientDetEngineBuilder(EngineBuilder):
    """Parses an ONNX graph and builds a TensorRT engine from it."""

    def get_input_dims(self, model_path):
        """Get input dimension of UFF model."""
        onnx_model = onnx.load(model_path)
        onnx_inputs = onnx_model.graph.input
        logger.info('List inputs:')
        for i, inputs in enumerate(onnx_inputs):
            logger.info('Input %s -> %s.', i, inputs.name)
            logger.info('%s.', [i.dim_value for i in inputs.type.tensor_type.shape.dim][1:])
            logger.info('%s.', [i.dim_value for i in inputs.type.tensor_type.shape.dim][0])

    def create_network(self, model_path, file_format="onnx"):
        """Parse the ONNX graph and create the corresponding TensorRT network definition.

        Args:
            model_path: The path to the ONNX graph to load.
        """
        if file_format == "onnx":
            self.get_input_dims(model_path)
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
                self.batch_size = input.shape[0]
                logger.info("Input '%s' with shape %s and dtype %s", input.name, input.shape, input.dtype)
            for output in outputs:
                logger.info("Output '%s' with shape %s and dtype %s", output.name, output.shape, output.dtype)

            # TF1 EfficientDet only support static batch size
            assert self.batch_size > 0
        else:
            logger.info("Parsing UFF model")
            raise NotImplementedError("UFF for EfficientDet is not supported")
