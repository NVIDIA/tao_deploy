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


def trt_output_process_fn(y_encoded):
    """Function to process TRT model output."""
    nms_out, _ = y_encoded
    nms_out = np.reshape(nms_out.host, nms_out.numpy_shape)
    # (x1, y1, x2, y2), shape = (N, 1, R, 4)
    nmsed_boxes = nms_out[:, 0, :, 3:7]
    # shape = (N, 1, R, 1)
    nmsed_scores = nms_out[:, 0, :, 2]
    # shape = (N, 1, R, 1)
    nmsed_classes = nms_out[:, 0, :, 1]

    result = []
    # apply the spatial pyramid pooling to the proposed regions
    for idx in range(nmsed_boxes.shape[0]):
        loc = nmsed_boxes[idx].reshape(-1, 4)
        cid = nmsed_classes[idx].reshape(-1, 1)
        conf = nmsed_scores[idx].reshape(-1, 1)
        result.append(np.concatenate((cid, conf, loc), axis=-1))
    return result


class FRCNNInferencer(TRTInferencer):
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

    def infer(self, imgs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        # Wrapped in list since arg is list of named tensor inputs
        # For Faster-RCNN, there is just 1: [input_image]
        self._copy_input_to_host([imgs])

        # ...fetch model outputs...
        # 2 named results: [nms_out, nms_out_1]
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)

        # Process TRT outputs to proper format
        return trt_output_process_fn(results)
