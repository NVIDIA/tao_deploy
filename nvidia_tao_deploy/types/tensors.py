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

"""TensorRT tensor types"""

import inspect
import tensorrt as trt

VALID_DATA_FORMATS = ["channel_first", "channel_last"]


class Tensor:
    """Simple data structure containing all the required metadata for the inference tensor."""

    def __init__(self,
                 tensor_name: str,
                 tensor_shape: tuple | list,
                 tensor_dtype: trt.float32,
                 optimization_profile: list = None,
                 data_format="channel_first",
                 batch_size=None):
        """Constructor to initialize the data structure."""
        self.tensor_name = tensor_name
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.optimization_profile = optimization_profile
        assert data_format in VALID_DATA_FORMATS, (
            f"Invalid data format encountered {data_format}. Valid options are {VALID_DATA_FORMATS}"
        )
        self.data_format = data_format
        self.batch_size = batch_size

    @property
    def size(self):
        """Get size of the input tensor."""
        batch_size = self.tensor_shape[0] if self.tensor_shape[0] > 0 else self.batch_size
        return trt.volume(self.shape) if batch_size is None else trt.volume(
            (batch_size,) + tuple(x for x in self.shape)
        )

    @property
    def shape(self):
        """Get shape of the array without batch size."""
        return self.tensor_shape[1:]

    def __str__(self):
        """Get string representation of this data structure."""
        inspectable_members = [item for item in inspect.getmembers(self) if not item[0].startswith("_")]
        string_representation = ""
        for member in inspectable_members:
            if not inspect.ismethod(member[1]):
                string_representation += f"{member[0]}: {member[1]}\n"
        return string_representation

    @property
    def height(self):
        """Get height of this tensor."""
        if len(self.tensor_shape) != 4:
            return None
        if self.data_format == "channel_first":
            return self.tensor_shape[2]
        return self.tensor_shape[1]

    @property
    def width(self):
        """Get width of this tensor."""
        if len(self.tensor_shape) != 4:
            return None
        if self.data_format == "channel_first":
            return self.tensor_shape[3]
        return self.tensor_shape[2]


class InputTensor(Tensor):
    """Simple data structure containing all the required metadata for the inference tensor."""

    def __init__(self,
                 tensor_name: str,
                 tensor_shape: tuple | list,
                 tensor_dtype: trt.float32,
                 optimization_profile: list = None,
                 data_format="channel_first",
                 batch_size=None):
        """Constructor to initialize the data structure."""
        super().__init__(
            tensor_name=tensor_name,
            tensor_shape=tensor_shape,
            tensor_dtype=tensor_dtype,
            optimization_profile=optimization_profile,
            data_format=data_format,
            batch_size=batch_size)

    @property
    def size(self):
        """Get size of the input tensor."""
        array_shape = self.tensor_shape[-3:]
        batch_size = self.tensor_shape[0]
        if batch_size < 0:
            assert self.optimization_profile, (
                "For dynamic batch sizes, optimization profiles must be defined from the TensorRT engine."
            )
            batch_size = self.optimization_profile[2][0]
        return trt.volume((batch_size, ) + array_shape)

    @property
    def max_batch_size(self):
        """Get max batch size of the tensor."""
        if self.tensor_shape[0] < 0:
            assert self.optimization_profile, (
                "For dynamic batch sizes, optimization profiles must be defined from the TensorRT engine."
            )
            return self.optimization_profile[2][0]
        return self.tensor_shape[0]
