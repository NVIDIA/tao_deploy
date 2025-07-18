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

"""G-DINO TensorRT engine builder."""

import logging
import os
import sys

import tensorrt as trt

from nvidia_tao_deploy.engine.builder import EngineBuilder, TRT_8_API


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


class GDINODetEngineBuilder(EngineBuilder):
    """Parses an ONNX graph and builds a TensorRT engine from it."""

    def __init__(
        self,
        max_text_len=256,
        img_std=[0.229, 0.224, 0.225],
        **kwargs
    ):
        """Init.

        Args:
            data_format (str): data_format.
        """
        super().__init__(**kwargs)
        self.max_text_len = max_text_len
        self._img_std = img_std

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
            for model_input in inputs:
                logger.info("Input '%s' with shape %s and dtype %s", model_input.name, model_input.shape, model_input.dtype)
                input_shape = model_input.shape
                input_name = model_input.name
                if input_name == 'inputs':
                    real_shape_min = (self.min_batch_size, *input_shape[1:])
                    real_shape_opt = (self.opt_batch_size, *input_shape[1:])
                    real_shape_max = (self.max_batch_size, *input_shape[1:])
                elif input_name == 'text_token_mask':
                    real_shape_min = (self.min_batch_size, self.max_text_len, self.max_text_len)
                    real_shape_opt = (self.opt_batch_size, self.max_text_len, self.max_text_len)
                    real_shape_max = (self.max_batch_size, self.max_text_len, self.max_text_len)
                else:
                    real_shape_min = (self.min_batch_size, self.max_text_len)
                    real_shape_opt = (self.opt_batch_size, self.max_text_len)
                    real_shape_max = (self.max_batch_size, self.max_text_len)

                opt_profile.set_shape(input=input_name,
                                      min=real_shape_min,
                                      opt=real_shape_opt,
                                      max=real_shape_max)

            self.config.add_optimization_profile(opt_profile)
            self.config.set_calibration_profile(opt_profile)

            for output in outputs:
                logger.info("Output '%s' with shape %s and dtype %s", output.name, output.shape, output.dtype)

        else:
            raise NotImplementedError(f"{file_format.capitalize()} backend is not supported for D-DETR.")
