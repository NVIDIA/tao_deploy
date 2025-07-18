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

logger = logging.getLogger(__name__)


def trt_output_process(y_encoded):
    """Function to process TRT model output.

    Args:
        y_encoded (list): list of TRT outputs in numpy

    Returns:
        siam_pred (np.ndarray)
    """
    return [np.reshape(out.host, out.numpy_shape) for out in y_encoded]


class OpticalInspectionInferencer(TRTInferencer):
    """Manages TensorRT objects for model inference."""

    def __init__(self, engine_path, input_shape=None, batch_size=None, data_format="channel_first"):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last
        """
        # Load TRT engine
        # logger.info("Loading engine from {}".format(engine_path))
        super().__init__(engine_path,
                         input_shape=input_shape,
                         batch_size=batch_size,
                         data_format=data_format)

    def infer(self, input_images):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        # 2 named inputs: [input_1, input_2]
        # input_images is already a list, so we can just pass it in since arg is list of named tensor inputs
        self._copy_input_to_host(input_images)

        # ...fetch model outputs...
        # 2 named results: [siam_pred, 208]
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)

        # Process TRT outputs to proper format
        pred = trt_output_process(results)
        return pred
