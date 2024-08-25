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

"""TensorRT Engine class for Mask2former."""

import numpy as np

import tensorrt as trt  # pylint: disable=unused-import

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import allocate_buffers, do_inference


def sigmoid(x):
    """Numpy sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


class Mask2formerInferencer(TRTInferencer):
    """Implements inference for the Mask2former TensorRT engine."""

    def __init__(self, engine_path,
                 input_shape=None,
                 batch_size=None,
                 data_format="channel_first",
                 is_inference=False):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last
        """
        # Load TRT engine
        super().__init__(engine_path)
        self.is_inference = is_inference
        self.max_batch_size = self.engine.max_batch_size
        self.execute_v2 = False

        # Execution context is needed for inference
        self.context = None

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        self._input_shape = []

        for binding in range(self.engine.num_bindings):
            if self.engine.binding_is_input(binding):
                self._input_shape = self.engine.get_binding_shape(binding)[-3:]
        assert len(self._input_shape) == 3, "Engine doesn't have valid input dimensions"

        if data_format == "channel_first":
            self.height = self._input_shape[1]
            self.width = self._input_shape[2]
        else:
            self.height = self._input_shape[0]
            self.width = self._input_shape[1]

        # set binding_shape for dynamic input
        if (input_shape is not None) or (batch_size is not None):
            self.context = self.engine.create_execution_context()
            if input_shape is not None:
                self.context.set_binding_shape(0, input_shape)
                self.max_batch_size = input_shape[0]
            else:
                self.context.set_binding_shape(0, [batch_size] + list(self._input_shape))
                self.max_batch_size = batch_size
            self.execute_v2 = True

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine,
                                                                                 self.context)
        if self.context is None:
            self.context = self.engine.create_execution_context()

        input_volume = trt.volume(self._input_shape)
        self.numpy_array = np.zeros((self.max_batch_size, input_volume))
        self.postprocess_fn_map = {
            1: self.semantic_postprocess_fn,
            3: self.instance_postprocess_fn,
            4: self.panoptic_postprocess_fn
        }

    def infer(self, imgs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        # Verify if the supplied batch size is not too big
        max_batch_size = self.max_batch_size
        actual_batch_size = len(imgs)
        if actual_batch_size > max_batch_size:
            raise ValueError(f"image_paths list bigger ({actual_batch_size}) than \
                               engine max batch size ({max_batch_size})")

        self.numpy_array[:actual_batch_size] = imgs.reshape(actual_batch_size, -1)
        # ...copy them into appropriate place into memory...
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, self.numpy_array.ravel())

        # ...fetch model outputs...
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=max_batch_size,
            execute_v2=self.execute_v2)

        # ...and return results up to the actual batch size.
        y_pred = [i.reshape(max_batch_size, -1)[:actual_batch_size] for i in results]

        # Process TRT outputs to proper format
        return self.postprocess_fn_map[len(y_pred)](y_pred, actual_batch_size)

    def semantic_postprocess_fn(self, y_encoded, batch_size):
        """Function to process TRT model output.

        Args:
            y_encoded (list): list of TRT outputs in numpy
            batch_size (int): batch size from TRT engine

        Returns:
            semseg (np.ndarray): (B x C x H x W) mask prediction
        """
        assert len(y_encoded) == 1
        pred_masks = y_encoded[-1]
        pred_masks = pred_masks.reshape((batch_size, -1, self.height, self.width))
        return pred_masks

    def instance_postprocess_fn(self, y_encoded, batch_size):
        """Function to process TRT model output.

        Args:
            y_encoded (list): list of TRT outputs in numpy
            batch_size (int): batch size from TRT engine

        Returns:
            pred_classes (np.ndarray): (B x C) labels prediction
            pred_masks (np.ndarray): (B x C x H x W) mask prediction
            pred_scores (np.ndarray): (B x C) scores prediction
        """
        assert len(y_encoded) == 3
        pred_classes, pred_masks, pred_scores = y_encoded
        pred_masks = pred_masks.reshape((batch_size, -1, self.height, self.width))
        return pred_classes, pred_masks, pred_scores

    def panoptic_postprocess_fn(self, y_encoded, batch_size):
        """Function to process TRT model output.

        Args:
            y_encoded (list): list of TRT outputs in numpy
            batch_size (int): batch size from TRT engine

        Returns:
            prob_mask (np.ndarray): (B x C x H x W) prob mask prediction
            pred_masks (np.ndarray): (B x C x H x W) mask prediction
            pred_scores (np.ndarray): (B x C) scores prediction
            pred_classes (np.ndarray): (B x C) labels prediction
        """
        assert len(y_encoded) == 4
        pred_scores, pred_classes, pred_masks, prob_mask = y_encoded
        pred_masks = pred_masks.reshape((batch_size, -1, self.height, self.width))
        prob_mask = prob_mask.reshape((batch_size, -1, self.height, self.width))
        return prob_mask, pred_masks, pred_scores, pred_classes

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
