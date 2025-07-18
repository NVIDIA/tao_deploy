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

"""Base TensorRT inferencer."""

from abc import abstractmethod
import logging

import numpy as np
from nvidia_tao_deploy.types.tensors import InputTensor, Tensor
from nvidia_tao_deploy.utils import LEGACY_API_MODE
from nvidia_tao_deploy.inferencer.base_inferencer import BaseInferencer
from nvidia_tao_deploy.inferencer.utils import allocate_buffers

import tensorrt as trt

TRT_8_API = LEGACY_API_MODE()
logger = logging.getLogger(__name__)


class TRTInferencer(BaseInferencer):
    """Base TRT Inferencer."""

    def __init__(self,
                 engine_path: str,
                 trt_logger=trt.Logger(trt.Logger.WARNING),
                 input_shape: dict = None,
                 batch_size: int = None,
                 reshape: bool = False,
                 data_format="channel_first"):
        """Init.

        Args:
            engine_path (str): The path to the serialized engine to load from disk.
        """
        super().__init__(model_path=engine_path, logger=trt_logger)
        self.context = self.engine.create_execution_context()
        self.data_format = data_format
        self.execute_async = False
        if TRT_8_API:
            self.max_batch_size = self.engine.max_batch_size
        assert self.context, (
            "Valid context wasn't loaded."
        )
        self.profile_idx = None
        self.define_io_tensors(reshape=reshape, input_shape=input_shape, batch_size=batch_size)
        assert self.stream, "Stream should not be none."

        for idx in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(idx), self.bindings[idx])

        logger.debug("Input tensor details.")
        for tensor in self.input_tensors:
            logger.debug(tensor)
        logger.debug("Output tensor details.")
        for tensor in self.output_tensors:
            logger.debug(tensor)

        for inpt in self.inputs:
            logger.debug(str(inpt))

        for output in self.outputs:
            logger.debug(str(output))

        self.numpy_array = {
            tensor.tensor_name: np.zeros((tensor.max_batch_size, *tensor.tensor_shape[1:]), dtype=trt.nptype(tensor.tensor_dtype))
            for tensor in self.input_tensors
        }

    def define_io_tensors(self, reshape: bool = False, input_shape: dict = None, batch_size: int = None):
        """Define and allocate IO buffers for TRT inference."""
        self.input_tensors = []
        self.output_tensors = []
        for idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(idx)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            # Note: When deriving the shape from the engine context, by default it sets
            # the batch size to be max (from the optimization profile). Otherwise this
            # gets messy when deriving from the engine, especially for dynamic shapes,
            # because output tensor batch sizes are also -1.
            tensor_shape = self.context.get_tensor_shape(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            if is_input:
                self.define_input_tensor(tensor_name, tensor_dtype, tensor_shape, input_shape, batch_size)
            else:
                self.define_output_tensor(tensor_name, tensor_dtype, tensor_shape)

        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(
            self.engine,
            self.context,
            profile_idx=self.profile_idx,
            reshape=reshape
        )

    def define_input_tensor(self, tensor_name, tensor_dtype, tensor_shape, input_shape, batch_size):
        """Define input tensor."""
        if TRT_8_API:
            if len(tensor_shape) == 4 or tensor_shape[0] < 0:
                self.etlt_type = "onnx"
            else:
                self.etlt_type == "uff"
        else:
            # From TensorRT 10.x only ONNX is supported as a valid model parser.
            self.etlt_type = "onnx"
        io_tensor_kwargs = {
            "data_format": self.data_format
        }

        # Handling dynamic batch sizes.
        self.profile_idx = 0
        if tensor_shape[0] < 0:
            optimization_profile = self.engine.get_tensor_profile_shape(tensor_name, self.profile_idx)
            assert len(optimization_profile) == 3, "Invalid optimization profile."
            io_tensor_kwargs["optimization_profile"] = optimization_profile

        io_tensor = InputTensor(tensor_name, tensor_shape, tensor_dtype, **io_tensor_kwargs)
        self.input_tensors.append(io_tensor)

        self.max_batch_size = io_tensor.max_batch_size
        if TRT_8_API and self.engine.has_implicit_batch_dimension:
            self.max_batch_size = self.engine.max_batch_size
        elif self.etlt_type == "onnx":
            if input_shape:
                self.context.set_input_shape(tensor_name, input_shape)
                self.max_batch_size = input_shape[0]
            else:
                self.context.set_input_shape(tensor_name, [batch_size] + list(tensor_shape[1:]))
                self.max_batch_size = batch_size
                self.execute_async = True

    def define_output_tensor(self, tensor_name, tensor_dtype, tensor_shape):
        """Define output tensor."""
        io_tensor_kwargs = {
            "data_format": self.data_format
        }
        io_tensor = Tensor(tensor_name, tensor_shape, tensor_dtype, **io_tensor_kwargs)
        self.output_tensors.append(io_tensor)

    def setup_inferencer_session(self, logger):
        """Setup the inferencer session and required context etc."""
        self.logger = logger
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(self.logger)
        self.stream = None

    def load_model(self, model_path: str):
        """Method to load a previously serialized TensorRT engine.

        Args:
            model_path (str): Path to the model file.
        """
        with open(model_path, 'rb') as f:
            engine_data = f.read()
            engine = self.trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def _copy_input_to_host(self, inputs):
        """Copy the input array for pagelocked host memory.

        Args:
            inputs (np.ndarray): Input array. Each element corresponds to a named input tensor
        """
        for idx, tensor_name in enumerate(self.numpy_array.keys(), start=0):
            actual_batch_size = len(inputs[idx])
            if actual_batch_size > self.max_batch_size:
                raise ValueError(
                    f"input list bigger ({actual_batch_size}) than "
                    f"engine max batch size ({self.max_batch_size})"
                )
            # numpy_array always has max_batch_size
            # (For static batch sizes, max_batch_size == actual_batch_size)
            self.numpy_array[tensor_name][:actual_batch_size] = inputs[idx]
            # ...copy them into appropriate place into memory...
            # (self.inputs was returned earlier by allocate_buffers())
            np.copyto(self.inputs[idx].host, self.numpy_array[tensor_name].ravel())

    @abstractmethod
    def infer(self, imgs, scales=None):
        """Execute inference on a batch of images.

        The images should already be batched and preprocessed.
        Memory copying to and from the GPU device will be performed here.

        Args:
            imgs (np.ndarray): A numpy array holding the image batch.
            scales: The image resize scales for each image in this batch.
                Default: No scale postprocessing applied.

        Returns:
            A nested list for each image in the batch and each detection in the list.
        """
        detections = {}
        return detections

    def __del__(self):
        """Clear things up on object deletion."""
        # Clear session and buffer
        if self.trt_runtime:
            del self.trt_runtime

        if self.context:
            del self.context

        if self.engine:
            del self.engine

        if self.stream:
            del self.stream

        # Loop through inputs and free inputs.
        for inp in self.inputs:
            inp.device.free()

        # Loop through outputs and free them.
        for out in self.outputs:
            out.device.free()
