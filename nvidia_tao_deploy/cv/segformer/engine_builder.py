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

"""Segformer TensorRT engine builder."""

import logging
import tensorrt as trt

from nvidia_tao_deploy.engine.builder import EngineBuilder


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


class SegformerEngineBuilder(EngineBuilder):
    """Parses an UFF/ONNX graph and builds a TensorRT engine from it."""

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

    def create_engine(self, engine_path, precision,
                      calib_input=None, calib_cache=None, calib_num_images=5000,
                      calib_batch_size=8, calib_data_file=None, calib_json_file=None,
                      layers_precision=None, profilingVerbosity="detailed"):
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
        if precision == "fp32" and self.builder.platform_has_tf32:
            self.config.set_flag(trt.BuilderFlag.TF32)

        super().create_engine(engine_path, precision, calib_input,
                              calib_cache, calib_num_images, calib_batch_size,
                              calib_data_file, calib_json_file, layers_precision,
                              profilingVerbosity)
