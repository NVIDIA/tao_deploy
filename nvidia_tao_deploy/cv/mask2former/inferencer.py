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

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import do_inference


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
        super().__init__(engine_path,
                         input_shape=input_shape,
                         batch_size=batch_size,
                         data_format=data_format)
        self.is_inference = is_inference
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
        # Wrapped in list since arg is list of named tensor inputs
        # For Mask2Former, there is just 1: [inputs]
        self._copy_input_to_host([imgs])

        # ...fetch model outputs...
        # 1 named output: [pred_masks]
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)

        # Process TRT outputs to proper format
        return self.postprocess_fn_map[len(results)](results)

    def semantic_postprocess_fn(self, y_encoded):
        """Function to process TRT model output.

        Args:
            y_encoded (list): list of TRT outputs in numpy

        Returns:
            semseg (np.ndarray): (B x C x H x W) mask prediction
        """
        assert len(y_encoded) == 1
        pred_masks = y_encoded[-1]
        pred_masks = np.reshape(pred_masks.host, pred_masks.numpy_shape)
        return pred_masks

    def instance_postprocess_fn(self, y_encoded):
        """Function to process TRT model output.

        Args:
            y_encoded (list): list of TRT outputs in numpy

        Returns:
            pred_classes (np.ndarray): (B x C) labels prediction
            pred_masks (np.ndarray): (B x C x H x W) mask prediction
            pred_scores (np.ndarray): (B x C) scores prediction
        """
        assert len(y_encoded) == 3
        pred_masks, pred_scores, pred_classes = y_encoded
        pred_masks = np.reshape(pred_masks.host, pred_masks.numpy_shape)
        pred_classes = np.reshape(pred_classes.host, pred_classes.numpy_shape)
        pred_scores = np.reshape(pred_scores.host, pred_scores.numpy_shape)
        return pred_classes, pred_masks, pred_scores

    def panoptic_postprocess_fn(self, y_encoded):
        """Function to process TRT model output.

        Args:
            y_encoded (list): list of TRT outputs in numpy

        Returns:
            prob_mask (np.ndarray): (B x C x H x W) prob mask prediction
            pred_masks (np.ndarray): (B x C x H x W) mask prediction
            pred_scores (np.ndarray): (B x C) scores prediction
            pred_classes (np.ndarray): (B x C) labels prediction
        """
        assert len(y_encoded) == 4
        prob_mask, pred_masks, pred_scores, pred_classes = y_encoded
        pred_masks = np.reshape(pred_masks.host, pred_masks.numpy_shape)
        prob_mask = np.reshape(prob_mask.host, prob_mask.numpy_shape)
        pred_scores = np.reshape(pred_scores.host, pred_scores.numpy_shape)
        pred_classes = np.reshape(pred_classes.host, pred_classes.numpy_shape)
        return prob_mask, pred_masks, pred_scores, pred_classes
