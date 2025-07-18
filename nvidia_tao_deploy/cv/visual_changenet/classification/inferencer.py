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

"""Utility class for performing TensorRT image inference for Visual ChangeNet-Classification"""

import logging
import numpy as np

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import do_inference
from nvidia_tao_deploy.cv.visual_changenet.classification.utils import softmax

logger = logging.getLogger(__name__)


def trt_output_process_fn(y_encoded, diff_module='learnable', batch_size=1):
    """Function to process TRT model output."""
    # For euclidian, shape is (B,), should be (B, 1)
    # For learnable, shape is (B, 2)
    y = y_encoded.host
    y = y.reshape(batch_size, -1)
    if len(y.shape) == 1:
        y = np.expand_dims(y, 1)
    predictions_batch = []
    for output in y:
        if diff_module == 'learnable':
            assert len(output) > 1, "Use the correct diff_module in model_config. Supported values are ['euclidean', 'learnable']"
            output = softmax(output)[1]  # Get the Softmax probability of defective class
        else:
            assert diff_module == 'euclidean' and len(output) == 1, "Use the correct diff_module in model_config. Supported values are ['euclidean', 'learnable']"
        predictions_batch.append([output])
    return np.array(predictions_batch)


class ChangeNetInferencer(TRTInferencer):
    """Manages TensorRT objects for model inference."""

    def __init__(self, engine_path, input_shape=None, batch_size=None, data_format="channel_first", diff_module='learnable'):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last,
            diff_module (str): either 'learnable' or 'euclidean'
        """
        # Load TRT engine
        super().__init__(
            engine_path,
            input_shape=input_shape,
            batch_size=batch_size,
            data_format=data_format,
            reshape=False
        )
        self.diff_module = diff_module

    def infer(self, input_images):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        # For VCN, there are 2 named inputs that input_images contains: [inputs0, inputs1]
        self._copy_input_to_host(input_images)

        # ...fetch model outputs...
        # 1 named result: [output]
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)
        # Process TRT outputs to proper format
        pred = trt_output_process_fn(results[0], self.diff_module, self.max_batch_size)
        # ...and return results up to the actual batch size.
        # (Since we do not use the numpy_shape attribute, we have to manually truncate the batch size)
        actual_batch_size = len(input_images[0])
        pred = pred[:actual_batch_size]
        return pred
