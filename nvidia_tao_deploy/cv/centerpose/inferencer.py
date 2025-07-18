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

"""TensorRT Engine class for CenterPose."""

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import do_inference
import numpy as np


def trt_output_process(y_encoded):
    """Function to process TRT model output.

    Args:
        y_encoded (list): list of TRT outputs in numpy

    Returns:
        pred_logits (np.ndarray): (B x NQ x N) logits of the prediction
        pred_boxes (np.ndarray): (B x NQ x 4) bounding boxes of the prediction
    """
    # Defining the output dictionary based on the output names and shapes defined during the engine
    # Generation.
    detections = {
        out.name: np.reshape(out.host, out.numpy_shape) for out in y_encoded
    }
    return detections


class CenterPoseInferencer(TRTInferencer):
    """Implements inference for the CenterPose TensorRT engine."""

    def __init__(self, engine_path, input_shape=None, batch_size=None, data_format="channel_first"):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last
        """
        # Load TRT engine
        super().__init__(
            engine_path,
            input_shape=input_shape,
            batch_size=batch_size,
            data_format=data_format,
            reshape=False
        )

    def infer(self, imgs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        # Wrapped in list since arg is list of named tensor inputs
        # For Centerpose, there is just 1: [input]
        self._copy_input_to_host([imgs])

        # ...fetch model outputs...
        # 7 named results: [bboxes, scores, kps, clses, obj_scale, kps_displacement_mean, kps_heatmap_mean]
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)

        # Process TRT outputs to proper format
        det = trt_output_process(results)
        return det
