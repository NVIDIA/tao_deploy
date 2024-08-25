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

"""EfficientDet TensorRT inferencer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pycuda.autoinit # noqa pylint: disable=unused-import
import pycuda.driver as cuda
import tensorrt as trt

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer


class EfficientDetInferencer(TRTInferencer):
    """Implements inference for the EfficientDet TensorRT engine."""

    def __init__(self, engine_path, max_detections_per_image=100):
        """Init.

        Args:
            engine_path (str): The path to the serialized engine to load from disk.
            max_detections_per_image (int): The maximum number of detections to visualize
        """
        # Load TRT engine
        super().__init__(engine_path)
        self.max_detections_per_image = max_detections_per_image

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
                self._input_shape = shape
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """Get the specs for the input tensor of the network. Useful to prepare memory allocations.

        Args:
            None

        Returns:
            the shape of the input tensor.
            (numpy) datatype of the input tensor.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """Get the specs for the output tensors of the network. Useful to prepare memory allocations.

        Args:
            None

        Returns:
            specs: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, imgs, scales=None):
        """Execute inference on a batch of images.

        The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.

        Args:
            imgs: A numpy array holding the image batch.
            scales: The image resize scales for each image in this batch.
                    Default: No scale postprocessing applied.

        Returns:
            detections: A nested list for each image in the batch and each detection in the list.
        """
        # Prepare the output data
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(imgs))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])

        nums = self.max_detections_per_image
        boxes = outputs[1][:, :nums, :]

        scores = outputs[2][:, :nums]
        classes = outputs[3][:, :nums]

        # Reorganize from y1, x1, y2, x2 to x1, y1, x2, y2
        boxes[:, :, [0, 1]] = boxes[:, :, [1, 0]]
        boxes[:, :, [2, 3]] = boxes[:, :, [3, 2]]

        # convert x2, y2 to w, h
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        # Scale the box
        for i in range(len(boxes)):
            boxes[i] /= scales[i]

        detections = {}
        detections['num_detections'] = np.array([nums] * self.batch_size).astype(np.int32)
        detections['detection_classes'] = classes + 1
        detections['detection_scores'] = scores
        detections['detection_boxes'] = boxes

        return detections

    def __del__(self):
        """Simple function to destroy tensorrt handlers."""
        if self.context:
            del self.context

        if self.engine:
            del self.engine

        if self.allocations:
            self.allocations.clear()
