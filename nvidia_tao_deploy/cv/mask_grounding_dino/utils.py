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

"""Utility functions to be used for Mask Grounding DINO."""
import cv2
from functools import partial
import numpy as np
from nvidia_tao_deploy.cv.deformable_detr.utils import sigmoid, box_cxcywh_to_xyxy


def post_process(pred_logits, pred_boxes, pred_masks, target_sizes, pos_maps, num_select=300):
    """Perform the post-processing. Scale back the boxes to the original size.

    Args:
        pred_logits (np.ndarray): (B x NQ x 4) logit values from TRT engine.
        pred_boxes (np.ndarray): (B x NQ x 4) bbox values from TRT engine.
        pred_masks (np.ndarray): (B x NQ x h x w) pred masks from TRT engine.
        target_sizes (np.ndarray): (B x 4) [w, h, w, h] containing original image dimension.
        num_select (int): Top-K proposals to choose from.
        pos_maps (np.ndarray): [N, max_len]. e.g [2, 256]
    Returns:
        labels (np.ndarray): (B x NS) class label of top num_select predictions.
        scores (np.ndarray): (B x NS) class probability of top num_select predictions.
        boxes (np.ndarray):  (B x NS x 4) scaled back bounding boxes of top num_select predictions.
    """
    bs = pred_logits.shape[0]
    # Sigmoid
    # prob_to_token = sigmoid(pred_logits).reshape((bs, pred_logits.shape[1], -1))
    prob_to_token = sigmoid(pred_logits)

    for label_ind in range(len(pos_maps)):
        if pos_maps[label_ind].sum() != 0:
            pos_maps[label_ind] = pos_maps[label_ind] / pos_maps[label_ind].sum()

    prob_to_label = prob_to_token @ pos_maps.T

    prob = prob_to_label  # 1, 900, 2

    # Get topk scores
    topk_indices = np.argsort(prob.reshape((bs, -1)), axis=1)[:, ::-1][:, :num_select]

    scores = [per_batch_prob[ind] for per_batch_prob, ind in zip(prob.reshape((bs, -1)), topk_indices)]
    scores = np.array(scores)

    # Get corresponding boxes
    topk_boxes = topk_indices // prob.shape[2]
    # Get corresponding labels
    labels = topk_indices % prob.shape[2]

    # Convert to x1, y1, x2, y2 format
    boxes = box_cxcywh_to_xyxy(pred_boxes)

    # Take corresponding topk boxes
    boxes = np.take_along_axis(boxes, np.repeat(np.expand_dims(topk_boxes, -1), 4, axis=-1), axis=1)

    # Scale back the bounding boxes to the original image size
    target_sizes = np.array(target_sizes)
    boxes = boxes * target_sizes[:, None, :]

    masks = []

    def _resize_func(x, width, height):
        return cv2.resize(x, (width, height), interpolation=cv2.INTER_LINEAR)

    # Clamp bounding box coordinates
    for i, target_size in enumerate(target_sizes):
        w, h = target_size[0], target_size[1]
        boxes[i, :, 0::2] = np.clip(boxes[i, :, 0::2], 0.0, w)
        boxes[i, :, 1::2] = np.clip(boxes[i, :, 1::2], 0.0, h)
        m = pred_masks[i][topk_boxes[i], ...]
        m = np.transpose(m, (1, 2, 0))

        N = 2  # small number, divisible by n_queries
        m_split = np.split(m, N, axis=2)
        m_split = list(map(partial(_resize_func, width=w, height=h), m_split))
        m_scaled = np.concatenate(m_split, axis=2)

        m_scaled = sigmoid(m_scaled)
        masks.append(m_scaled)
    return labels, scores, boxes, masks
