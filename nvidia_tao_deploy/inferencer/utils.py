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

"""Base utility functions for TensorRT inferencer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pycuda.autoinit # noqa pylint: disable=unused-import
import pycuda.driver as cuda
import tensorrt as trt


class HostDeviceMem(object):
    """Clean data structure to handle host/device memory."""

    def __init__(self, host_mem, device_mem, npshape, name: str = None):
        """Initialize a HostDeviceMem data structure.

        Args:
            host_mem (cuda.pagelocked_empty): A cuda.pagelocked_empty memory buffer.
            device_mem (cuda.mem_alloc): Allocated memory pointer to the buffer in the GPU.
            npshape (tuple): Shape of the input dimensions.

        Returns:
            HostDeviceMem instance.
        """
        self.host = host_mem
        self.device = device_mem
        self.numpy_shape = npshape
        self.name = name

    def __str__(self):
        """String containing pointers to the TRT Memory."""
        return "Name: " + self.name + "\nHost:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        """Return the canonical string representation of the object."""
        return self.__str__()


def do_inference(context, bindings, inputs,
                 outputs, stream, batch_size=1,
                 execute_v2=False, return_raw=False):
    """Generalization for multiple inputs/outputs.

    inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    # Run inference.
    if execute_v2:
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    else:
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    # Synchronize the stream
    stream.synchronize()

    if return_raw:
        return outputs

    # Return only the host outputs.
    return [out.host for out in outputs]


def allocate_buffers(engine, context=None, reshape=False):
    """Allocates host and device buffer for TRT engine inference.

    This function is similair to the one in common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.

    Args:
        engine (trt.ICudaEngine): TensorRT engine
        context (trt.IExecutionContext): Context for dynamic shape engine
        reshape (bool): To reshape host memory or not (FRCNN)

    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]
    binding_to_type = {"Input": np.float32, "NMS": np.float32, "NMS_1": np.int32,
                       "BatchedNMS": np.int32, "BatchedNMS_1": np.float32,
                       "BatchedNMS_2": np.float32, "BatchedNMS_3": np.float32,
                       "generate_detections": np.float32,
                       "mask_head/mask_fcn_logits/BiasAdd": np.float32,
                       "softmax_1": np.float32,
                       "input_1": np.float32,
                       # D-DETR
                       "inputs": np.float32,
                       "pred_boxes": np.float32,
                       "pred_logits": np.float32}

    for binding in engine:
        binding_id = engine.get_binding_index(str(binding))
        binding_name = engine.get_binding_name(binding_id)
        if context:
            size = trt.volume(context.get_binding_shape(binding_id))
            dims = context.get_binding_shape(binding_id)
        else:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dims = engine.get_binding_shape(binding)
        # avoid error when bind to a number (YOLO BatchedNMS)
        size = engine.max_batch_size if size == 0 else size
        if str(binding) in binding_to_type:
            dtype = binding_to_type[str(binding)]
        else:
            dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)

        # FRCNN requires host memory to be reshaped into target shape
        if reshape and not engine.binding_is_input(binding):
            if engine.has_implicit_batch_dimension:
                target_shape = (engine.max_batch_size, dims[0], dims[1], dims[2])
            else:
                target_shape = dims
            host_mem = host_mem.reshape(*target_shape)

        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem, dims, name=binding_name))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, dims, name=binding_name))
    return inputs, outputs, bindings, stream
