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

import numpy as np

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import do_inference


def trt_output_process_fn(y_encoded, height, width):
    """function to process TRT model output."""
    y_encoded = [np.reshape(out.host, out.numpy_shape) for out in y_encoded]
    det_out, keep_k = y_encoded
    result = []
    for idx, k in enumerate(keep_k.reshape(-1)):
        det = det_out[idx].reshape(-1, 7)[:k]
        xmin = det[:, 3] * width
        ymin = det[:, 4] * height
        xmax = det[:, 5] * width
        ymax = det[:, 6] * height
        cls_id = det[:, 1]
        conf = det[:, 2]
        result.append(np.stack((cls_id, conf, xmin, ymin, xmax, ymax), axis=-1))

    return np.array(result)


class SSDInferencer(TRTInferencer):
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
        super().__init__(engine_path,
                         input_shape=input_shape,
                         batch_size=batch_size,
                         data_format=data_format)

        self.height = self.input_tensors[0].height
        self.width = self.input_tensors[0].width

    def infer(self, imgs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        # Wrapped in list since arg is list of named tensor inputs
        # For SSD, there is just 1: [Input]
        self._copy_input_to_host([imgs])

        # ...fetch model outputs...
        # 2 named results: [NMS, NMS_1]
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)

        # Process TRT outputs to proper format
        return trt_output_process_fn(results, self.height, self.width)
