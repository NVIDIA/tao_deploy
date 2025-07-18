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

"""Mask2former TensorRT engine builder."""

import logging

import tensorrt as trt

from nvidia_tao_deploy.engine.builder import EngineBuilder

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
precision_mapping = {
    'fp32': trt.float32,
    'fp16': trt.float16,
    'int32': trt.int32,
    'int8': trt.int8,
}


class Mask2formerEngineBuilder(EngineBuilder):
    """Parses an ONNX graph and builds a TensorRT engine from it."""

    def __init__(
        self,
        data_format="channels_first",
        img_std=[0.229, 0.224, 0.225],
        **kwargs
    ):
        """Init.

        Args:
            data_format (str): data_format.
        """
        super().__init__(**kwargs)
        self._data_format = data_format
        self._img_std = img_std

    def set_layer_precisions(self, layer_precisions):
        """Set the layer precision for specified layers.

        This function control per-layer precision constraints. Effective only when
        "OBEY_PRECISION_CONSTRAINTS" or "PREFER_PRECISION_CONSTRAINTS" is set by builder config.
        the layer name is identical to the node name from your ONNX model.

        Returns:
            No explicit returns.
        """
        for layer in self.network:
            layer_name = layer.name
            for k, v in layer_precisions.items():
                if k in layer_name:
                    if any(x in layer_name.lower() for x in ['/concat', '/gather', '/transpose', '/split', 'squeeze', '/expand', 'nonzero', '/tile']):
                        continue
                    if layer.precision in (trt.int32, trt.bool):
                        logger.info("Skipped setting precision for layer {} because the \
                                    default layer precision is INT32 or Bool.".format(layer_name))
                        continue

                    #  We should not set the constant layer precision if its weights are in INT32.
                    if layer.type == trt.LayerType.CONSTANT:
                        logger.info("Skipped setting precision for layer {} because this \
                                    constant layer has INT32 weights.".format(layer_name))
                        continue

                    #  We should not set the layer precision if the layer operates on a shape tensor.
                    if layer.num_inputs >= 1 and (layer.get_input(0).is_shape_tensor or layer.get_output(0).is_shape_tensor):

                        logger.info("Skipped setting precision for layer {} because this layer \
                                    operates on a shape tensor.".format(layer_name))
                        continue

                    if (layer.num_inputs >= 1 and layer.get_input(0).dtype == trt.int32 and layer.num_outputs >= 1 and layer.get_output(0).dtype == trt.int32):

                        logger.info("Skipped setting precision for layer {} because this \
                                    layer has INT32 input and output.".format(layer_name))
                        continue

                    #  All heuristics passed. Set the layer precision.
                    layer.precision = precision_mapping[v]
                    logger.info("Setting precision for layer {} to {}.".format(layer_name, layer.precision))
