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

"""Utility class for performing TensorRT image inference."""

import numpy as np

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import do_inference
from nvidia_tao_deploy.cv.pointpillars.utils.nms_cpu import nms_cpu


def trt_output_process_fn(y_encoded):
    """Function to process TRT model output."""
    return [np.reshape(out.host, out.numpy_shape) for out in y_encoded]


class PointPillarsInferencer(TRTInferencer):
    """Manages TensorRT objects for model inference."""

    def __init__(self, engine_path, input_shape=None, batch_size=None, data_format="channel_first"):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            input_shape (tuple): (batch, points, channel) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last
        """
        # Load TRT engine
        super().__init__(engine_path,
                         input_shape=input_shape,
                         batch_size=batch_size,
                         data_format=data_format)
        self.points_shape = self.context.get_tensor_shape("points")

    def infer(self, imgs, scales=None, points=None, num_points=None):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            points: lidar points.
            num_points: number of lidar points.
        """
        # Wrapped in list since arg is list of named tensor inputs
        if imgs is not None:
            raise ValueError("PointPillars does not take images as input")
        if scales is not None:
            raise ValueError("PointPillars does not take images as input")
        if len(points) != 1:
            raise ValueError(f"PointPillars input points batch size can only support 1, got {len(points)}")
        if len(points) != len(num_points):
            raise ValueError("PointPillars input tensors' batch size not the same.")
        points = np.expand_dims(points[0], axis=0)
        num_points = np.array(num_points, dtype=np.int32)
        if points.shape[1] > self.points_shape[1]:
            raise ValueError(
                f"Input LIDAR file has points number: {points.shape[1]} larger than "
                f"the one specified in the model: {self.points_shape[1]}, please "
                "re-export the model"
            )
        if points.shape[1] < self.points_shape[1]:
            points = np.pad(
                points,
                ((0, 0), (0, self.points_shape[1] - points.shape[1]), (0, 0)),
                constant_values=0.
            )
        self._copy_input_to_host([points, num_points])

        # ...fetch model outputs...
        # 0: output_boxes, 1: num_boxes
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)

        # Process TRT outputs to proper format
        return trt_output_process_fn(results)


class CustomNMS():
    """NMS module."""

    def __init__(self, post_process_cfg):
        """Initialize."""
        self.post_process_cfg = post_process_cfg

    def forward(self, output_boxes, num_boxes):
        """Forward method."""
        batch_output = []
        for idx, box_per_frame in enumerate(output_boxes):
            num_box_per_frame = num_boxes[idx]
            box_per_frame = box_per_frame[:num_box_per_frame, ...]
            box_preds = box_per_frame[:, 0:7]
            label_preds = box_per_frame[:, 7] + 1
            cls_preds = box_per_frame[:, 8]
            selected, selected_scores = nms_cpu(
                box_scores=cls_preds, box_preds=box_preds,
                nms_config=self.post_process_cfg.nms_config,
                score_thresh=self.post_process_cfg.score_thresh
            )
            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]
            final_output = np.concatenate(
                [
                    final_boxes,
                    final_scores.reshape((-1, 1)),
                    final_labels.reshape((-1, 1))
                ],
                axis=-1
            )
            batch_output.append(final_output)
        return batch_output


class CustomPostProcessing():
    """PostProcessing module."""

    def __init__(self, cfg):
        """Initialize."""
        self.custom_nms = CustomNMS(cfg)

    def forward(self, output_boxes, num_boxes):
        """Forward method."""
        return self.custom_nms.forward(
            output_boxes,
            num_boxes
        )


class TrtModelWrapper():
    """TensorRT model wrapper."""

    def __init__(self, cfg, trt_model):
        """Initialize."""
        self.cfg = cfg
        self.trt_model = trt_model
        self.post_processor = CustomPostProcessing(
            self.cfg.model.post_processing
        )

    def __call__(self, input_dict):
        """call method."""
        trt_output = self.trt_model.infer(None, points=input_dict["points"], num_points=input_dict["num_points"])
        return self.post_processor.forward(
            trt_output[0],
            trt_output[1],
        )
