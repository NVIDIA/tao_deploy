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

"""Utility functions to be used for Grounding DINO."""

import numpy as np
from nvidia_tao_deploy.cv.deformable_detr.utils import sigmoid, box_cxcywh_to_xyxy


def post_process(pred_logits, pred_boxes, target_sizes, pos_maps, num_select=300):
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
    bs = pred_logits.shape[0]
    # Sigmoid
    # prob_to_token = sigmoid(pred_logits).reshape((bs, pred_logits.shape[1], -1))
    prob_to_token = sigmoid(pred_logits)

    for label_ind in range(len(pos_maps)):
        if pos_maps[label_ind].sum() != 0:
            pos_maps[label_ind] = pos_maps[label_ind] / pos_maps[label_ind].sum()

    prob_to_label = prob_to_token @ pos_maps.T

    prob = prob_to_label

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

    # Clamp bounding box coordinates
    for i, target_size in enumerate(target_sizes):
        w, h = target_size[0], target_size[1]
        boxes[i, :, 0::2] = np.clip(boxes[i, :, 0::2], 0.0, w)
        boxes[i, :, 1::2] = np.clip(boxes[i, :, 1::2], 0.0, h)

    return labels, scores, boxes


def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list):
    """Generate attention mask between each pair of special tokens.

    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        torch.Tensor: attention mask between each special tokens.
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = np.zeros((bs, num_token), dtype=bool)
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = np.stack(np.nonzero(special_tokens_mask), axis=1)

    # generate attention mask and positional ids
    attention_mask = (
        np.tile(np.expand_dims(np.eye(num_token, dtype=bool), axis=0), (bs, 1, 1))
    )
    position_ids = np.zeros((bs, num_token))
    cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if col in (0, num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1: col + 1, previous_col + 1: col + 1] = True
            position_ids[row, previous_col + 1: col + 1] = np.arange(
                0, col - previous_col
            )
            c2t_maski = np.zeros((num_token), dtype=bool)
            c2t_maski[previous_col + 1: col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col
    return attention_mask, position_ids


def create_positive_map(tokenized, tokens_positive, cat_list, caption, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j

    Args:
        tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = np.zeros((len(tokens_positive), max_text_len), dtype=float)

    for j, label in enumerate(tokens_positive):
        start_ind = caption.find(cat_list[label])
        end_ind = start_ind + len(cat_list[label]) - 1
        beg_pos = tokenized.char_to_token(start_ind)
        try:
            end_pos = tokenized.char_to_token(end_ind)
        except Exception:
            end_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end_ind - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end_ind - 2)
            except Exception:
                end_pos = None

        if beg_pos is None or end_pos is None:
            continue
        if beg_pos < 0 or end_pos < 0:
            continue
        if beg_pos > end_pos:
            continue
        # assert beg_pos is not None and end_pos is not None
        positive_map[j, beg_pos: end_pos + 1].fill(1)
    return positive_map


def tokenize_captions(tokenizer, cat_list, caption, max_text_len=256):
    """tokenize captions."""
    specical_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    tokenized = tokenizer(caption, padding="max_length", return_tensors="np", max_length=max_text_len)

    label_list = np.arange(len(cat_list))
    pos_map = create_positive_map(tokenized, label_list, cat_list, caption[0], max_text_len=max_text_len)

    (
        text_self_attention_masks,
        position_ids,
    ) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, specical_tokens)

    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[
            :, : max_text_len, : max_text_len]

        position_ids = position_ids[:, : max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, : max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, : max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : max_text_len]

    input_ids = tokenized["input_ids"].astype(int)
    attention_mask = tokenized["attention_mask"].astype(bool)
    position_ids = position_ids.astype(int)
    token_type_ids = tokenized["token_type_ids"].astype(int)
    text_self_attention_masks = text_self_attention_masks.astype(bool)

    return input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map
