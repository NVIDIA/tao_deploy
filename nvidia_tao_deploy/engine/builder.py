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

"""Base TensorRT engine builder."""

from abc import ABC, abstractmethod
import json
import logging
import os
import struct
import tensorrt as trt

from nvidia_tao_deploy.engine.calibrator import EngineCalibrator
from nvidia_tao_deploy.utils.image_batcher import ImageBatcher


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKSPACE_SIZE = 8
DEFAULT_MAX_BATCH_SIZE = 1
DEFAULT_MIN_BATCH_SIZE = 1
DEFAULT_OPT_BATCH_SIZE = 1
NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH = [
    int(item) for item
    in trt.__version__.split(".")
][:3]


class EngineBuilder(ABC):
    """Parses an ONNX graph and builds a TensorRT engine from it."""

    def __init__(self,
                 batch_size=None,
                 verbose=False,
                 max_batch_size=DEFAULT_MAX_BATCH_SIZE,
                 opt_batch_size=DEFAULT_OPT_BATCH_SIZE,
                 min_batch_size=DEFAULT_MIN_BATCH_SIZE,
                 workspace=DEFAULT_MAX_WORKSPACE_SIZE,
                 strict_type_constraints=False,
                 force_ptq=False,
                 is_qat=False):
        """Create a TensorRT engine.

        Args:
            batch_size (int): batch_size used for calibration
            verbose (bool): If enabled, a higher verbosity level will be set on the TensorRT logger.
            max_batch_size (int): Maximum batch size.
            opt_batch_size (int): Optimal batch size.
            min_batch_size (int): Minimum batch size.
            workspace (int): Max memory workspace to allow, in Gb.
            strict_type (bool): Whether or not to apply strict_type_constraints for INT8 mode.
            force_ptq (bool): Flag to force post training quantization using TensorRT
                for a QAT trained model. This is required if the inference platform is
                a Jetson with a DLA.
            is_qat (bool): Wheter or not the model is a QAT.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = workspace * (2 ** 30)

        self.batch_size = batch_size
        self.max_batch_size, self.opt_batch_size, self.min_batch_size = max_batch_size, opt_batch_size, min_batch_size
        self.network = None
        self.parser = None

        # Disable QAT regardless of is_qat flag if force_ptq is True
        self._is_qat = is_qat if not force_ptq else False
        self._strict_type = strict_type_constraints

        self._trt_version_number = NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + \
            NV_TENSORRT_PATCH
        # if self._trt_version_number < 8600:
        #     if self._trt_version_number >= 8500:
        #         logger.info("TRT version is lower than 8.6. Setting PreviewFeature.FASTER_DYNAMIC_SHAPES_0805 for better performance")
        #         faster_dynamic_shapes = True  # Only supported from TRT 8.5+
        #     else:
        #         faster_dynamic_shapes = False
        #     self.config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, faster_dynamic_shapes)

    @abstractmethod
    def create_network(self, model_path):
        """Parse the ONNX or UFF graph and create the corresponding TensorRT network definition.

        Args:
            model_path (str): The path to the ONNX or UFF graph to load.
        """
        pass

    def set_calibrator(self,
                       inputs=None,
                       calib_cache=None,
                       calib_input=None,
                       calib_num_images=5000,
                       calib_batch_size=8,
                       calib_data_file=None,
                       image_mean=None):
        """Simple function to set an int8 calibrator. (Default is ImageBatcher based)

        Args:
            inputs (list): Inputs to the network
            calib_input (str): The path to a directory holding the calibration images.
            calib_cache (str): The path where to write the calibration cache to,
                         or if it already exists, load it from.
            calib_num_images (int): The maximum number of images to use for calibration.
            calib_batch_size (int): The batch size to use for the calibration process.

        Returns:
            No explicit returns.
        """
        logger.info("Calibrating using ImageBatcher")

        self.config.int8_calibrator = EngineCalibrator(calib_cache)
        if not os.path.exists(calib_cache):
            calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
            calib_dtype = trt.nptype(inputs[0].dtype)
            self.config.int8_calibrator.set_image_batcher(
                ImageBatcher(calib_input, calib_shape, calib_dtype,
                             max_num_images=calib_num_images,
                             exact_batches=True))

    def _logger_info_IBuilderConfig(self):
        """Print tensorrt.tensorrt.IBuilderConfig"""
        if self.config:
            logger.info("TensorRT engine build configurations:")

            opt_prof = self.config.get_calibration_profile()
            if opt_prof:
                logger.info('  OptimizationProfile: ')
                for index in range(self.network.num_inputs):
                    tensor = self.network.get_input(index)
                    if tensor.is_shape_tensor:
                        min_shape, opt_shape, max_shape = opt_prof.get_shape_input(tensor.name)
                    else:
                        min_shape, opt_shape, max_shape = opt_prof.get_shape(tensor.name)
                    logger.info('    \"%s\": %s, %s, %s', tensor.name, min_shape, opt_shape, max_shape)

            logger.info(' ')
            if self.config.get_flag(trt.BuilderFlag.FP16):
                logger.info('  BuilderFlag.FP16')
            if self.config.get_flag(trt.BuilderFlag.INT8):
                logger.info('  BuilderFlag.INT8')
            if self.config.get_flag(trt.BuilderFlag.DEBUG):
                logger.info('  BuilderFlag.DEBUG')
            if self.config.get_flag(trt.BuilderFlag.GPU_FALLBACK):
                logger.info('  BuilderFlag.GPU_FALLBACK')
            if self.config.get_flag(trt.BuilderFlag.STRICT_TYPES):
                logger.info('  BuilderFlag.STRICT_TYPES')
            if self.config.get_flag(trt.BuilderFlag.REFIT):
                logger.info('  BuilderFlag.REFIT')
            if self.config.get_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE):
                logger.info('  BuilderFlag.DISABLE_TIMING_CACHE')
            if self.config.get_flag(trt.BuilderFlag.TF32):
                logger.info('  BuilderFlag.TF32')
            if self.config.get_flag(trt.BuilderFlag.SPARSE_WEIGHTS):
                logger.info('  BuilderFlag.SPARSE_WEIGHTS')
            if self.config.get_flag(trt.BuilderFlag.SAFETY_SCOPE):
                logger.info('  BuilderFlag.SAFETY_SCOPE')
            if self.config.get_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS):
                logger.info('  BuilderFlag.OBEY_PRECISION_CONSTRAINTS')
            if self.config.get_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS):
                logger.info('  BuilderFlag.PREFER_PRECISION_CONSTRAINTS')
            if self.config.get_flag(trt.BuilderFlag.DIRECT_IO):
                logger.info('  BuilderFlag.DIRECT_IO')
            if self.config.get_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS):
                logger.info('  BuilderFlag.REJECT_EMPTY_ALGORITHMS')
            if self.config.get_flag(trt.BuilderFlag.ENABLE_TACTIC_HEURISTIC):
                logger.info('  BuilderFlag.ENABLE_TACTIC_HEURISTIC')

            logger.info(' ')
            # Return int32 and thus cannot represent >2GB
            logger.info('  Note: max representabile value is 2,147,483,648 bytes or 2GB.')
            pool_limit = self.config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE)
            logger.info('  MemoryPoolType.WORKSPACE = %d bytes', pool_limit)
            pool_limit = self.config.get_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM)
            logger.info('  MemoryPoolType.DLA_MANAGED_SRAM = %d bytes', pool_limit)
            pool_limit = self.config.get_memory_pool_limit(trt.MemoryPoolType.DLA_LOCAL_DRAM)
            logger.info('  MemoryPoolType.DLA_LOCAL_DRAM = %d bytes', pool_limit)
            pool_limit = self.config.get_memory_pool_limit(trt.MemoryPoolType.DLA_GLOBAL_DRAM)
            logger.info('  MemoryPoolType.DLA_GLOBAL_DRAM = %d bytes', pool_limit)

            logger.info(' ')
            if self.config.get_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805):
                logger.info('  PreviewFeature.FASTER_DYNAMIC_SHAPES_0805')
            if self.config.get_preview_feature(trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805):
                logger.info('  PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805')
            if self.config.get_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION):
                logger.info('  QuantizationFlag.CALIBRATE_BEFORE_FUSION')
            tactic_sources = self.config.get_tactic_sources()
            logger.info('  Tactic Sources = %d', tactic_sources)

    def create_engine(self, engine_path, precision,
                      calib_input=None, calib_cache=None, calib_num_images=5000,
                      calib_batch_size=8, calib_data_file=None, calib_json_file=None):
        """Build the TensorRT engine and serialize it to disk.

        Args:
            engine_path (str): The path where to serialize the engine to.
            precision (str): The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
            calib_input (str): The path to a directory holding the calibration images.
            calib_cache (str): The path where to write the calibration cache to,
                         or if it already exists, load it from.
            calib_num_images (int): The maximum number of images to use for calibration.
            calib_batch_size (int): The batch size to use for the calibration process.
            calib_json_file (str): The path to JSON file containing tensor scale dictionary for QAT
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        logger.debug("Building %s Engine in %s", precision, engine_path)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        if self.batch_size is None:
            self.batch_size = calib_batch_size
            self.builder.max_batch_size = self.batch_size

        # This should be only applied for ONNX
        if self.batch_size != calib_batch_size and self.batch_size > 0:
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
                self.config.set_flag(trt.BuilderFlag.INT8)
                if self.builder.platform_has_fast_fp16 and not self._strict_type:
                    # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                    self.config.set_flag(trt.BuilderFlag.FP16)
                else:
                    self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                logger.info("Calibrating using tensor scales for QAT model")
                # Load from calib_json_file
                self.calibration_cache_from_dict(calib_cache, calib_json_file)
                # Set dynamic ranges of tensors using scales from QAT
                self._set_tensor_dynamic_ranges(
                    network=self.network, tensor_scale_dict=self.tensor_scale_dict
                )

            else:
                if self.builder.platform_has_fast_fp16 and not self._strict_type:
                    # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                    self.config.set_flag(trt.BuilderFlag.FP16)
                else:
                    self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                self.config.set_flag(trt.BuilderFlag.INT8)
                # Set ImageBatcher based calibrator
                self.set_calibrator(inputs=inputs,
                                    calib_cache=calib_cache,
                                    calib_input=calib_input,
                                    calib_num_images=calib_num_images,
                                    calib_batch_size=calib_batch_size,
                                    calib_data_file=calib_data_file)

        self._logger_info_IBuilderConfig()
        with self.builder.build_engine(self.network, self.config) as engine, \
                open(engine_path, "wb") as f:
            logger.debug("Serializing engine to file: %s", engine_path)
            f.write(engine.serialize())

    def calibration_cache_from_dict(self, calibration_cache=None, calib_json=None):
        """Write calibration cache file for QAT model.

        This function converts a tensor scale dictionary generated by processing
        QAT models to TRT readable format. By default we set it as a
        trt.IInt8.EntropyCalibrator2 cache file.

        Args:
            calibration_cache (str): Path to output calibration cache file.
            calib_json (str): Path to calibration json file containing scale value

        Returns:
            No explicit returns.
        """
        if not os.path.exists(calib_json):
            raise FileNotFoundError(f"Calibration JSON file is required for QAT \
                                    but {calib_json} does not exist.")

        with open(calib_json, "r", encoding="utf-8") as f:
            self.tensor_scale_dict = json.load(f)["tensor_scales"]

        if calibration_cache is not None:
            cal_cache_str = f"TRT-{self._trt_version_number}-EntropyCalibration2\n"
            assert not os.path.exists(calibration_cache), (
                "A pre-existing cache file exists. Please delete this "
                "file and re-run export."
            )
            # Converting float numbers to hex representation.
            for tensor in self.tensor_scale_dict:
                scaling_factor = self.tensor_scale_dict[tensor] / 127.0
                cal_scale = hex(struct.unpack(
                    "i", struct.pack("f", scaling_factor))[0])
                assert cal_scale.startswith(
                    "0x"), "Hex number expected to start with 0x."
                cal_scale = cal_scale[2:]
                cal_cache_str += tensor + ": " + cal_scale + "\n"
            with open(calibration_cache, "w", encoding="utf-8") as f:
                f.write(cal_cache_str)

    def _set_tensor_dynamic_ranges(self, network, tensor_scale_dict):
        """Set the scaling factors obtained from quantization-aware training.

        Args:
            network: TensorRT network object.
            tensor_scale_dict (dict): Dictionary mapping names to tensor scaling factors.
        """
        tensors_found = []
        for idx in range(network.num_inputs):
            input_tensor = network.get_input(idx)
            if input_tensor.name in tensor_scale_dict:
                tensors_found.append(input_tensor.name)
                cal_scale = tensor_scale_dict[input_tensor.name]
                input_tensor.dynamic_range = (-cal_scale, cal_scale)

        for layer in network:
            found_all_outputs = True
            for idx in range(layer.num_outputs):
                output_tensor = layer.get_output(idx)
                if output_tensor.name in tensor_scale_dict:
                    tensors_found.append(output_tensor.name)
                    cal_scale = tensor_scale_dict[output_tensor.name]
                    output_tensor.dynamic_range = (-cal_scale, cal_scale)
                else:
                    found_all_outputs = False
            if found_all_outputs:
                layer.precision = trt.int8
        tensors_in_dict = tensor_scale_dict.keys()
        if set(tensors_in_dict) != set(tensors_found):
            logger.info("Tensors in scale dictionary but not in network: %s",
                        set(tensors_in_dict) - set(tensors_found))
