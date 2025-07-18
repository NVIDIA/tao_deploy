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

"""Utility class for performing TensorRT image inference."""

import logging
import numpy as np

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import do_inference
from nvidia_tao_deploy.cv.visual_changenet.segmentation.utils import ConfuseMatrixMeter


logger = logging.getLogger(__name__)


def trt_output_process_fn(y_encoded):
    """Function to process TRT model output."""
    return np.reshape(y_encoded.host, y_encoded.numpy_shape)


class ChangeNetInferencer(TRTInferencer):
    """Manages TensorRT objects for model inference."""

    def __init__(self, engine_path, input_shape=None, batch_size=None, data_format="channel_first", n_class=2, mode='test'):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last
            n_class (int): Number of output classes
            mode (str): either 'test' or 'predict' for evaluation and inference
        """
        # Load TRT engine
        super().__init__(
            engine_path,
            input_shape=input_shape,
            batch_size=batch_size,
            data_format=data_format,
            reshape=False
        )
        self.mode = mode
        if self.mode == 'test':
            self.running_metric = ConfuseMatrixMeter(n_class=n_class)

    def infer(self, imgs, target=None):  # noqa pylint: disable=W0237
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        # For VCN, there are 2 named inputs that imgs contains: [inputs0, inputs1]
        # imgs is already a list, so we can just pass it in since arg is list of named tensor inputs
        self._copy_input_to_host(imgs)

        # ...fetch model outputs...
        # 4 named results: [output1, output2, output3, output_final]
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)

        # Process TRT outputs to proper format using last FM output segmentation map
        y_pred = trt_output_process_fn(results[-1])
        if self.mode == 'test':
            self._update_metric(y_pred, target)
        return y_pred

    def _update_metric(self, pred, target):
        """Calculates running metrics for evaluation"""
        pred = np.argmax(pred, axis=1, keepdims=True)
        current_score = self.running_metric.update_cm(pr=pred, gt=target)

        return current_score
