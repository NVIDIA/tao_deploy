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

"""Utility functions to be used for D-DETR."""

import numpy as np


def box_cxcywh_to_xyxy(x):
    """Convert box from cxcywh to xyxy."""
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=-1)


def sigmoid(x):
    """Numpy-based sigmoid function."""
    return 1 / (1 + np.exp(-x))


def post_process(pred_logits, pred_boxes, target_sizes, num_select=100):
    """Perform the post-processing. Scale back the boxes to the original size.

    Args:
        pred_logits (np.ndarray): (B x NQ x 4) logit values from TRT engine.
        pred_boxes (np.ndarray): (B x NQ x 4) bbox values from TRT engine.
        target_sizes (np.ndarray): (B x 4) [w, h, w, h] containing original image dimension.
        num_select (int): Top-K proposals to choose from.

    Returns:
        labels (np.ndarray): (B x NS) class label of top num_select predictions.
        scores (np.ndarray): (B x NS) class probability of top num_select predictions.
        boxes (np.ndarray):  (B x NS x 4) scaled back bounding boxes of top num_select predictions.
    """
    # Sigmoid
    prob = sigmoid(pred_logits).reshape((pred_logits.shape[0], -1))

    # Get topk scores
    topk_indices = np.argsort(prob, axis=1)[:, ::-1][:, :num_select]

    scores = [per_batch_prob[ind] for per_batch_prob, ind in zip(prob, topk_indices)]
    scores = np.array(scores)

    # Get corresponding boxes
    topk_boxes = topk_indices // pred_logits.shape[2]
    # Get corresponding labels
    labels = topk_indices % pred_logits.shape[2]

    # Convert to x1, y1, x2, y2 format
    boxes = box_cxcywh_to_xyxy(pred_boxes)

    # Take corresponding topk boxes
    boxes = np.take_along_axis(boxes, np.repeat(np.expand_dims(topk_boxes, -1), 4, axis=-1), axis=1)

    # Scale back the bounding boxes to the original image size
    target_sizes = np.array(target_sizes)
    boxes = boxes * target_sizes[:, None, :]

    # Clamp bounding box coordinates
    for i, target_size in enumerate(target_sizes):
        w, h = target_size[0], target_size[1]
        boxes[i, :, 0::2] = np.clip(boxes[i, :, 0::2], 0.0, w)
        boxes[i, :, 1::2] = np.clip(boxes[i, :, 1::2], 0.0, h)

    return labels, scores, boxes
