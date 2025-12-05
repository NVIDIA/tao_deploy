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

from abc import ABC
import sys
import json
import logging
import numpy as np
import os
import struct
import tensorrt as trt
import traceback

from nvidia_tao_deploy.cv.common.parser import MetaGraph, UFF_ENABLED
from nvidia_tao_deploy.utils import LEGACY_API_MODE
from nvidia_tao_deploy.engine.calibrator import EngineCalibrator
from nvidia_tao_deploy.utils.image_batcher import ImageBatcher

TRT_8_API = LEGACY_API_MODE()


precision_mapping = {
    'fp32': trt.float32,
    'fp16': trt.float16,
    'int32': trt.int32,
    'int8': trt.int8,
}

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
                 is_qat=False,
                 timing_cache_path=None,
                 strongly_typed=False):
        """Create a TensorRT engine.

        Parameters
        ----------
        batch_size : int, optional
            Batch size used for calibration.
        verbose : bool, optional
            If enabled, a higher verbosity level will be set on the TensorRT logger.
        max_batch_size : int, optional
            Maximum batch size.
        opt_batch_size : int, optional
            Optimal batch size.
        min_batch_size : int, optional
            Minimum batch size.
        workspace : int, optional
            Max memory workspace to allow, in Gb.
        strict_type_constraints : bool, optional
            Whether or not to apply strict_type_constraints for INT8 mode.
        force_ptq : bool, optional
            Flag to force post training quantization using TensorRT
            for a QAT trained model. This is required if the inference platform is
            a Jetson with a DLA.
        is_qat : bool, optional
            Whether or not the model is a QAT.
        timing_cache_path : str, optional
            Path to timing cache that will be created/read/updated.
        strongly_typed : bool, optional
            Whether to enable strongly typed mode for quantized models.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        # Set max workspace size.
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace * (2 ** 30)
        )

        self.timing_cache_path = timing_cache_path
        self._set_timing_cache()

        self.batch_size = batch_size
        self.max_batch_size, self.opt_batch_size, self.min_batch_size = max_batch_size, opt_batch_size, min_batch_size
        self.network = None
        self.parser = None

        # Disable QAT regardless of is_qat flag if force_ptq is True
        logger.info("Setting up QAT mode: {is_qat}".format(is_qat=is_qat))
        self._is_qat = is_qat if not force_ptq else False
        self._strict_type = strict_type_constraints
        self._strongly_typed = strongly_typed

        self._trt_version_number = NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + \
            NV_TENSORRT_PATCH
        if self._trt_version_number < 8600:
            if self._trt_version_number >= 8500:
                logger.info("TRT version is lower than 8.6. Setting PreviewFeature.FASTER_DYNAMIC_SHAPES_0805 for better performance")
                faster_dynamic_shapes = True  # Only supported from TRT 8.5+
            else:
                faster_dynamic_shapes = False
            self.config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, faster_dynamic_shapes)

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
            self._input_dims = {}
            for k, v in input_dims.items():
                self._input_dims[k] = v[1:]

            logger.info("Network Description")
            opt_profile = self.builder.create_optimization_profile()
            for model_input in inputs: # noqa pylint: disable=W0622
                logger.info("Input '%s' with shape %s and dtype %s", model_input.name, model_input.shape, model_input.dtype)
                input_shape = model_input.shape
                input_name = model_input.name
                if self.batch_size <= 0:
                    real_shape_min = (self.min_batch_size, *input_shape[1:])
                    real_shape_opt = (self.opt_batch_size, *input_shape[1:])
                    real_shape_max = (self.max_batch_size, *input_shape[1:])
                    opt_profile.set_shape(
                        input=input_name,
                        min=real_shape_min,
                        opt=real_shape_opt,
                        max=real_shape_max
                    )
                else:
                    shape = (self.batch_size, *input_shape[1:])
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
            raise NotImplementedError(f"{file_format.capitalize()} backend is not supported for network.")

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
            if self.config.get_flag(trt.BuilderFlag.BF16):
                logger.info('  BuilderFlag.BF16')
            if self.config.get_flag(trt.BuilderFlag.INT8):
                logger.info('  BuilderFlag.INT8')
            if hasattr(trt.BuilderFlag, 'STRONGLY_TYPED') and self.config.get_flag(trt.BuilderFlag.STRONGLY_TYPED):
                logger.info('  BuilderFlag.STRONGLY_TYPED')
            if self.config.get_flag(trt.BuilderFlag.DEBUG):
                logger.info('  BuilderFlag.DEBUG')
            if self.config.get_flag(trt.BuilderFlag.GPU_FALLBACK):
                logger.info('  BuilderFlag.GPU_FALLBACK')
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
            if self.config.get_flag(trt.BuilderFlag.VERSION_COMPATIBLE):
                logger.info('  BuilderFlag.VERSION_COMPATIBLE')
            if self.config.get_flag(trt.BuilderFlag.FP8):
                logger.info('  BuilderFlag.FP8')
            if self.config.get_flag(trt.BuilderFlag.ERROR_ON_TIMING_CACHE_MISS):
                logger.info('  BuilderFlag.ERROR_ON_TIMING_CACHE_MISS')
            if self.config.get_flag(trt.BuilderFlag.DISABLE_COMPILATION_CACHE):
                logger.info('  BuilderFlag.DISABLE_COMPILATION_CACHE')
            if self.config.get_flag(trt.BuilderFlag.STRIP_PLAN):
                logger.info('  BuilderFlag.STRIP_PLAN')
            if self.config.get_flag(trt.BuilderFlag.REFIT_IDENTICAL):
                logger.info('  BuilderFlag.REFIT_IDENTICAL')
            if self.config.get_flag(trt.BuilderFlag.WEIGHT_STREAMING):
                logger.info('  BuilderFlag.WEIGHT_STREAMING')

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
            pool_limit = self.config.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM)
            logger.info('  MemoryPoolType.TACTIC_DRAM = %d bytes', pool_limit)
            pool_limit = self.config.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_SHARED_MEMORY)
            logger.info('  MemoryPoolType.TACTIC_SHARED_MEMORY = %d bytes', pool_limit)

            logger.info(' ')
            if self.config.get_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION):
                logger.info('  QuantizationFlag.CALIBRATE_BEFORE_FUSION')
            tactic_sources = self.config.get_tactic_sources()
            logger.info('  Tactic Sources = %d', tactic_sources)

    def _set_precision_constraints(self):
        """Set precision constraints based on platform."""
        if self.builder.platform_has_fast_fp16 and not self._strict_type:
            # Also enable fp16, as some layers may be even more efficient in fp16 than int8
            self.config.set_flag(trt.BuilderFlag.FP16)
        else:
            logger.info("Setting strict precision constraints.")
            if TRT_8_API:
                self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            else:
                # self.config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
                self.config.set_flag(trt.BuilderFlag.DIRECT_IO)
                self.config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

    def get_onnx_input_dims(self, onnx_inputs):
        """Get input dimension of ONNX model."""
        # @seanf: old version used onnx library's onnx.load() -> model.graph.input, but this was incorrect for mmcv models
        # It would treat every layer as an input
        # trt.OnnxParser does the job correctly
        logger.info('List inputs:')
        input_dims = {}
        for i, inputs in enumerate(onnx_inputs):
            logger.info('Input %s -> %s.', i, inputs.name)
            logger.info('%s.', inputs.shape[1:])
            logger.info('%s.', inputs.shape[0])
            input_dims[inputs.name] = inputs.shape
        return input_dims

    def get_uff_input_dims(self, model_path):
        """Get input dimension of UFF model."""
        if UFF_ENABLED:
            metagraph = MetaGraph()
            with open(model_path, "rb") as f:
                metagraph.ParseFromString(f.read())
            for node in metagraph.graphs[0].nodes:
                # if node.operation == "MarkOutput":
                #     print(f"Output: {node.inputs[0]}")
                if node.operation == "Input":
                    return np.array(node.fields['shape'].i_list.val)[1:]
            raise ValueError("Input dimension is not found in the UFF metagraph.")
        raise NotImplementedError(
            f"UFF parsing is not enabled for the current version of TensorRT ({trt.__version__})"
        )

    def parse_uff_model(self, model_path):
        """Parse a UFF model.

        Arg:

            model_path (str): Path to the UFF model.
        """
        # Setting this based for compatibility based on
        # TensorRT < 9.0.0
        if UFF_ENABLED:
            logger.info("Parsing UFF model")
            self.network = self.builder.create_network()
            self.parser = trt.UffParser()

            self.set_input_output_node_names()

            in_tensor_name = self._input_node_names[0]

            self._input_dims = self.get_uff_input_dims(model_path)
            input_dict = {in_tensor_name: self._input_dims}
            for key, value in input_dict.items():
                if self._data_format == "channels_first":
                    self.parser.register_input(key, value, trt.UffInputOrder(0))
                else:
                    self.parser.register_input(key, value, trt.UffInputOrder(1))
            for name in self._output_node_names:
                self.parser.register_output(name)
            self.builder.max_batch_size = self.max_batch_size
            try:
                assert self.parser.parse(model_path, self.network, trt.DataType.FLOAT)
            except AssertionError as e:
                logger.error("Failed to parse UFF File")
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)  # Fixed format
                tb_info = traceback.extract_tb(tb)
                _, line, _, text = tb_info[-1]
                raise AssertionError(
                    f"UFF parsing failed on line {line} in statement {text}"
                ) from e
        raise NotImplementedError(
            f"UFF model parsing is not implemented in TensorRT version {trt.__version__}."
        )

    def create_engine(self, engine_path, precision,
                      calib_input=None, calib_cache=None, calib_num_images=5000,
                      calib_batch_size=8, calib_data_file=None, calib_json_file=None,
                      layers_precision=None, profilingVerbosity="detailed",
                      tf2=False):
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
            if TRT_8_API:
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

        if self._is_qat and precision.lower() != "int8":
            raise ValueError(f"QAT model only supports data_type int8 but {precision} was provided.")

        # Set STRONGLY_TYPED flag for quantized models if requested
        if self._strongly_typed:
            logger.info("Setting STRONGLY_TYPED flag for quantized model")
            if hasattr(trt.BuilderFlag, 'STRONGLY_TYPED'):
                self.config.set_flag(trt.BuilderFlag.STRONGLY_TYPED)
            else:
                logger.warning("STRONGLY_TYPED flag not available in this TensorRT version")

        if precision.lower() == "fp16":
            if not self.builder.platform_has_fast_fp16:
                logger.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision.lower() == "int8":
            if not self.builder.platform_has_fast_int8:
                logger.warning("INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)
                self._set_precision_constraints()
                if self._is_qat:
                    if tf2:
                        # TF2 embeds QAT scales into the ONNX directly.
                        # Hence, no need to set dynamic range of tensors if tf2
                        pass
                    else:
                        logger.info("Setting tensor dynamic ranges from the QAT model tensor scales in {}".format(
                            calib_json_file
                        ))
                        self.calibration_cache_from_dict(calib_cache, calib_json=calib_json_file)
                        self._set_tensor_dynamic_ranges(
                            network=self.network,
                            tensor_scale_dict=self.tensor_scale_dict
                        )
                else:
                    self.set_calibrator(
                        inputs=inputs,
                        calib_cache=calib_cache,
                        calib_input=calib_input,
                        calib_num_images=calib_num_images,
                        calib_batch_size=calib_batch_size,
                        calib_data_file=calib_data_file
                    )

        if precision.lower() != "fp32":
            if layers_precision is not None and len(layers_precision) > 0:
                self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
                self.set_layer_precisions(layers_precision)

        if profilingVerbosity == "detailed":
            self.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        self._logger_info_IBuilderConfig()
        self._write_engine(engine_path)
        self._write_timing_cache()

    def _write_engine(self, engine_path: str):
        """Save engine based on the builder."""
        assert all([
            self.builder is not None,
            self.network is not None,
            self.config is not None
        ]), (
            "Make sure the network, build config and builder was defined."
        )
        with self.builder.build_serialized_network(self.network, self.config) as engine_bytes, \
                open(engine_path, "wb") as f:
            logger.debug("Serializing engine to file: %s", engine_path)
            f.write(engine_bytes)
        assert f.closed, (
            f"Engine file was not successfully serialized to {engine_path}"
        )
        # This is for a case where users want to build a pipeline
        # chaining the engine builder and the inferencer for results.
        return engine_bytes

    def _set_timing_cache(self):
        """Create timing cache and merge it with previous one if provided."""
        if self.timing_cache_path:
            timing_cache = self.config.create_timing_cache(b"")

            if os.path.exists(self.timing_cache_path):
                logger.info('Using timing cache %s', self.timing_cache_path)
                with open(self.timing_cache_path, "rb") as f:
                    loaded_timing_cache = self.config.create_timing_cache(f.read())
                    timing_cache.combine(loaded_timing_cache, ignore_mismatch=False)

            self.config.set_timing_cache(timing_cache, ignore_mismatch=False)

    def _write_timing_cache(self):
        """Serialize timing cache to file for future use."""
        if self.timing_cache_path:
            with self.config.get_timing_cache().serialize() as timing_cache, open(self.timing_cache_path, "wb") as f:
                f.write(timing_cache)
            assert os.path.exists(self.timing_cache_path), "Failed to write timing cache"

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

    def set_layer_precisions(self, layer_precisions):
        """Set the layer precision for specified layers.

        This function control per-layer precision constraints. Effective only when
        "OBEY_PRECISION_CONSTRAINTS" or "PREFER_PRECISION_CONSTRAINTS" is set by builder config.
        the layer name is identical to the node name from your ONNX model.

        Example:
        if you want to set some layers precision to "fp32" when you build TensorRT INT8 engine:

            builder = EngineBuilder()
            builder.create_network()
            layer_precisions = {'/backbone/conv1':'fp32', '/backbone/conv2':'fp32', '/backbone/conv3':'fp32'}
            builder.set_layer_precisions(layer_precisions)

        Besides, "*" can be used as a layer_name to specify the default precision for all the unspecified layers.
        if you only need to set a few layers precision to "int8" and all the other layers to "fp32", you can do like:

            builder = EngineBuilder()
            builder.create_network()
            layer_precisions = {'*':'fp32', '/backbone/conv1':'int8', '/backbone/conv2':'int8', '/backbone/conv3':'int8'}
            builder.set_layer_precisions(layer_precisions)

        Args:
            layer_precisions (dict): Dictionary mapping layers to precision. for example: {"layername":"dtype",...}, "dtype" should be in [fp32","fp16","int32","int8"]

        Returns:
            No explicit returns.
        """
        has_global_precision = "*" in layer_precisions.keys()
        global_precision = precision_mapping[layer_precisions["*"]] if has_global_precision else trt.float32
        has_layer_precision_skipped = False
        for layer in self.network:

            layer_name = layer.name
            if layer.name in layer_precisions.keys():
                layer.precision = precision_mapping[layer_precisions[layer.name]]
                logger.info("Setting precision for layer {} to {}.".format(layer_name, layer.precision))
            elif has_global_precision:
                # We should not set the layer precision if its default precision is INT32 or Bool.
                if layer.precision in (trt.int32, trt.bool):
                    has_layer_precision_skipped = True
                    logger.info("Skipped setting precision for layer {} because the \
                                default layer precision is INT32 or Bool.".format(layer_name))
                    continue

                #  We should not set the constant layer precision if its weights are in INT32.
                if layer.type == trt.LayerType.CONSTANT:
                    has_layer_precision_skipped = True
                    logger.info("Skipped setting precision for layer {} because this \
                                constant layer has INT32 weights.".format(layer_name))
                    continue

                #  We should not set the layer precision if the layer operates on a shape tensor.
                if layer.num_inputs >= 1 and layer.get_input(0).is_shape_tensor:

                    has_layer_precision_skipped = True
                    logger.info("Skipped setting precision for layer {} because this layer \
                                operates on a shape tensor.".format(layer_name))
                    continue

                if (layer.num_inputs >= 1 and layer.get_input(0).dtype == trt.int32 and layer.num_outputs >= 1 and layer.get_output(0).dtype == trt.int32):

                    has_layer_precision_skipped = True
                    logger.info("Skipped setting precision for layer {} because this \
                                layer has INT32 input and output.".format(layer_name))
                    continue

                #  All heuristics passed. Set the layer precision.
                layer.precision = global_precision

        if has_layer_precision_skipped:
            logger.info("Skipped setting precisions for some layers. Check verbose logs for more details.")
