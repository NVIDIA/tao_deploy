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

import numpy as np

from nvidia_tao_deploy.inferencer.utils import do_inference
from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer


class EfficientDetInferencer(TRTInferencer):
    """Implements inference for the EfficientDet TensorRT engine."""

    def __init__(self, engine_path, batch_size=1, max_detections_per_image=100, data_format="channel_last"):
        """Init.

        Args:
            engine_path (str): The path to the serialized engine to load from disk.
            max_detections_per_image (int): The maximum number of detections to visualize
        """
        # Load TRT engine
        super().__init__(engine_path,
                         batch_size=batch_size,
                         data_format=data_format)
        self.max_detections_per_image = max_detections_per_image

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
        # Wrapped in list since arg is list of named tensor inputs
        # For efficientdet, there is just 1: [image_arrays:0]
        self._copy_input_to_host([imgs])

        # ...fetch model outputs...
        # 4 named results: [num_detections, detection_boxes, detection_scores, detection_classes]
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)

        results = [np.reshape(out.host, out.numpy_shape) for out in results]

        nums = self.max_detections_per_image
        boxes = results[1][:, :nums, :]
        scores = results[2][:, :nums]
        classes = results[3][:, :nums]

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
        detections['num_detections'] = np.array([nums] * imgs.shape[0]).astype(np.int32)
        detections['detection_classes'] = classes + 1
        detections['detection_scores'] = scores
        detections['detection_boxes'] = boxes

        return detections
