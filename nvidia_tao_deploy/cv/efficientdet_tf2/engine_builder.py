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

    def create_network(self, model_path, dynamic_batch_size=None, file_format="onnx"):
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
            with open(model_path, "rb") as f:
                if not self.parser.parse(f.read()):
                    logger.error("Failed to load ONNX file: %s", model_path)
                    for error in range(self.parser.num_errors):
                        logger.error(self.parser.get_error(error))
                    sys.exit(1)

            inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
            # self.batch_size = self.max_batch_size

            logger.info("Network Description")
            profile = self.builder.create_optimization_profile()
            dynamic_inputs = False
            for inp in inputs:
                logger.info("Input '{}' with shape {} and dtype {}".format(inp.name, inp.shape, inp.dtype))  # noqa pylint: disable=C0209
                if inp.shape[0] == -1:
                    dynamic_inputs = True
                    if dynamic_batch_size:
                        if type(dynamic_batch_size) is str:
                            dynamic_batch_size = [int(v) for v in dynamic_batch_size.split(",")]
                        assert len(dynamic_batch_size) == 3
                        min_shape = [dynamic_batch_size[0]] + list(inp.shape[1:])
                        opt_shape = [dynamic_batch_size[1]] + list(inp.shape[1:])
                        max_shape = [dynamic_batch_size[2]] + list(inp.shape[1:])
                        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
                        logger.info("Input '{}' Optimization Profile with shape MIN {} / OPT {} / MAX {}".format(  # noqa pylint: disable=C0209
                            inp.name, min_shape, opt_shape, max_shape))
                    else:
                        shape = [self.batch_size] + list(inp.shape[1:])
                        profile.set_shape(inp.name, shape, shape, shape)
                        logger.info("Input '{}' Optimization Profile with shape {}".format(inp.name, shape))  # noqa pylint: disable=C0209
            if dynamic_inputs:
                self.config.add_optimization_profile(profile)
        else:
            logger.info("Parsing UFF model")
            raise NotImplementedError("UFF for EfficientDet is not supported")

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
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        logger.debug("Building %s Engine in %s", precision, engine_path)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        if self.batch_size is None:
            self.batch_size = calib_batch_size
            self.builder.max_batch_size = self.batch_size

        if self.batch_size != calib_batch_size:
            warning_msg = "For ONNX models with static batch size, " \
                          "calibration is done using the original batch size " \
                          f"of the ONNX model which is {self.batch_size}. " \
                          f"Overriding the provided calibration batch size {calib_batch_size}" \
                          f" to {self.batch_size}"
            logger.warning(warning_msg)
            calib_batch_size = self.batch_size

        if self._is_qat and precision != "int8":
            raise ValueError(f"QAT model only supports data_type int8 but {precision} was provided.")

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                logger.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                logger.warning("INT8 is not supported natively on this platform/device")
            elif self._is_qat:
                # TF2 embeds QAT scales into the ONNX directly.
                # Hence, no need to set dynamic range of tensors.
                self.config.set_flag(trt.BuilderFlag.INT8)
            else:
                if self.builder.platform_has_fast_fp16:
                    # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                    self.config.set_flag(trt.BuilderFlag.FP16)
                self.config.set_flag(trt.BuilderFlag.INT8)
                # Set ImageBatcher based calibrator
                self.set_calibrator(inputs=inputs,
                                    calib_cache=calib_cache,
                                    calib_input=calib_input,
                                    calib_num_images=calib_num_images,
                                    calib_batch_size=calib_batch_size,
                                    calib_data_file=calib_data_file)

        with self.builder.build_engine(self.network, self.config) as engine, \
                open(engine_path, "wb") as f:
            logger.debug("Serializing engine to file: %s", engine_path)
            f.write(engine.serialize())
