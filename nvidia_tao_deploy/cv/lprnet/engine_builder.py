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

"""LPRNet TensorRT engine builder."""

import logging

from nvidia_tao_deploy.engine.builder import EngineBuilder

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


class LPRNetEngineBuilder(EngineBuilder):
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

    def set_input_output_node_names(self):
        """Set input output node names."""
        self._output_node_names = ["tf_op_layer_ArgMax", "tf_op_layer_Max"]
        self._input_node_names = ["image_input"]

    def create_engine(self, engine_path, precision,
                      calib_input=None, calib_cache=None, calib_num_images=5000,
                      calib_batch_size=8, calib_data_file=None):
        """Build the TensorRT engine and serialize it to disk.

        Args:
            engine_path: The path where to serialize the engine to.
            precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
            calib_input: The path to a directory holding the calibration images.
            calib_cache: The path where to write the calibration cache to,
                         or if it already exists, load it from.
            calib_num_images: The maximum number of images to use for calibration.
            calib_batch_size: The batch size to use for the calibration process.
        """
        if precision == "int8":
            raise NotImplementedError("INT8 is not supported for LPRNet!")
        super().create_engine(engine_path, precision, calib_input,
                              calib_cache, calib_num_images, calib_batch_size,
                              calib_data_file)

    def set_data_preprocessing_parameters(self, input_dims, image_mean=None):
        """Set data pre-processing parameters for the int8 calibration."""
        num_channels = input_dims[0]
        if num_channels == 3:
            means = [0, 0, 0]
        elif num_channels == 1:
            means = [0]
        else:
            raise NotImplementedError(f"Invalid number of dimensions {num_channels}.")
        self.preprocessing_arguments = {"scale": 1.0 / 255.0,
                                        "means": means,
                                        "flip_channel": True}
