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
from datetime import datetime
from functools import partial
import json
import logging
import numpy as np
import os
from pathlib import Path
import random
import sys
from omegaconf import ListConfig
import pandas as pd
from nvidia_tao_deploy.cv.common.constants import VALID_IMAGE_EXTENSIONS
from nvidia_tao_deploy.cv.common.logging import status_logging
from nvidia_tao_deploy.cv.deformable_detr.utils import sigmoid, box_cxcywh_to_xyxy
from nvidia_tao_deploy.cv.grounding_dino.utils import create_positive_map, generate_masks_with_special_tokens_and_transfer_map
from nvidia_tao_deploy.utils.image_batcher import ImageBatcher


# Common color mapping
default_colors = {
    # Basic colors
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),

    # Grays
    'gray': (128, 128, 128),
    'grey': (128, 128, 128),
    'lightgray': (211, 211, 211),
    'lightgrey': (211, 211, 211),
    'darkgray': (169, 169, 169),
    'darkgrey': (169, 169, 169),
    'silver': (192, 192, 192),

    # Extended colors
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'pink': (255, 192, 203),
    'brown': (165, 42, 42),
    'lime': (0, 255, 0),
    'navy': (0, 0, 128),
    'teal': (0, 128, 128),
    'olive': (128, 128, 0),
    'maroon': (128, 0, 0),
    'aqua': (0, 255, 255),
    'fuchsia': (255, 0, 255),

    # Light variants
    'lightred': (255, 128, 128),
    'lightgreen': (144, 238, 144),
    'lightblue': (173, 216, 230),
    'lightyellow': (255, 255, 224),
    'lightpink': (255, 182, 193),
    'lightcyan': (224, 255, 255),

    # Dark variants
    'darkred': (139, 0, 0),
    'darkgreen': (0, 100, 0),
    'darkblue': (0, 0, 139),
    'darkyellow': (204, 204, 0),
    'darkorange': (255, 140, 0),
    'darkviolet': (148, 0, 211),

    # Others
    'gold': (255, 215, 0),
    'indigo': (75, 0, 130),
    'violet': (238, 130, 238),
    'turquoise': (64, 224, 208),
    'salmon': (250, 128, 114),
    'coral': (255, 127, 80),
    'khaki': (240, 230, 140),
    'lavender': (230, 230, 250),
    'beige': (245, 245, 220),
    'ivory': (255, 255, 240),
    'crimson': (220, 20, 60),
    'tan': (210, 180, 140),
    'skyblue': (135, 206, 235),
    'plum': (221, 160, 221),
}


def convert_jsonl_to_json(jsonl_file_path):
    """Convert JSONL file to COCO JSON format.

    Args:
        jsonl_file_path (str): Path to the JSONL file

    Returns:
        str: Path to the converted JSON file
    """
    # Create output filename in current working directory
    base_name = os.path.splitext(os.path.basename(jsonl_file_path))[0]
    json_file_path = os.path.join(os.getcwd(), f"{base_name}_converted.json")

    logging.info("Converting JSONL file %s to JSON format...", jsonl_file_path)
    logging.info("Output file: %s", json_file_path)

    # Initialize COCO format structure
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    # Track unique categories and image IDs
    categories_dict = {}
    image_ids_seen = set()
    annotation_id = 1

    # Read JSONL file line by line
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())

                # Extract image information
                image_id = data.get('image_id', line_idx)

                # Add image info if not seen before
                if image_id not in image_ids_seen:
                    # Extract filename from full path if needed
                    file_name = data.get('file_name', f"image_{image_id}.jpg")
                    if file_name.startswith('/'):
                        # Extract just the filename from full path
                        file_name = os.path.basename(file_name)

                    image_info = {
                        'id': image_id,
                        'width': data.get('width', 640),
                        'height': data.get('height', 480),
                        'file_name': file_name
                    }
                    coco_data['images'].append(image_info)
                    image_ids_seen.add(image_id)

                # Handle annotations - RefCOCO style format
                if 'annotations' in data and data['annotations']:
                    annotations = data['annotations'] if isinstance(data['annotations'], list) else [data['annotations']]

                    for ann_data in annotations:
                        # Skip empty annotations
                        if ann_data.get('empty', False):
                            continue

                        cat_id = ann_data.get('category_id', 1)

                        # Add category if not seen before
                        if cat_id not in categories_dict:
                            categories_dict[cat_id] = cat_id
                            # For RefCOCO datasets, we'll use generic category names
                            coco_data['categories'].append({
                                'id': cat_id,
                                'name': f'category_{cat_id}',
                                'supercategory': 'object'
                            })

                        # Calculate area from bbox if not provided
                        bbox = ann_data.get('bbox', [0, 0, 1, 1])
                        area = ann_data.get('area')
                        if area is None and len(bbox) >= 4:
                            area = bbox[2] * bbox[3]  # width * height

                        # Create annotation
                        annotation = {
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': cat_id,
                            'bbox': bbox,
                            'area': area if area is not None else 1.0,
                            'iscrowd': ann_data.get('iscrowd', 0)
                        }

                        # Add segmentation if available
                        if 'segmentation' in ann_data:
                            annotation['segmentation'] = ann_data['segmentation']

                        # Add referring expression info as additional metadata (optional)
                        if 'sentence' in data:
                            sentence_info = data['sentence']
                            annotation['referring_expression'] = sentence_info.get('raw', '')
                            annotation['ref_id'] = sentence_info.get('ref_id')
                            annotation['sent_id'] = sentence_info.get('sent_id')

                        coco_data['annotations'].append(annotation)
                        annotation_id += 1

            except json.JSONDecodeError as e:
                logging.warning("Skipping malformed line %d in JSONL file: %s", line_idx + 1, e)
                continue

    # Write converted data to JSON file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)

    logging.info("Successfully converted JSONL to JSON:")
    logging.info("  - %d images", len(coco_data['images']))
    logging.info("  - %d annotations", len(coco_data['annotations']))
    logging.info("  - %d categories", len(coco_data['categories']))

    return json_file_path


def get_phrase_from_expression_numpy(predictions, tokenizer, input_ids):
    """
    NumPy version of get_phrase_from_expression for TensorRT inference.

    Args:
        predictions: List[Dict]: List of predictions from the model.
            Each pred has 'text_mask': np.array of shape [K, 256]
        tokenizer: Tokenizer (HuggingFace tokenizer)
        tokenized: Dict with "input_ids": np.array [B, 256]

    Returns:
        predictions: List[Dict] with 'phrase' field added and 'text_mask' removed
    """
    for b, pred in enumerate(predictions):
        pred_text_mask = pred.pop("text_mask")  # [K, 256] - NumPy array
        phrase_list = []

        for phrase_mask in pred_text_mask:
            # Find indices of active tokens (non-zero elements)
            # PyTorch: phrase_mask.nonzero(as_tuple=True)[0].tolist()
            # NumPy equivalent:
            non_zero_idx = np.where(phrase_mask)[0].tolist()

            # Alternative NumPy approaches:
            # non_zero_idx = np.nonzero(phrase_mask)[0].tolist()
            # non_zero_idx = phrase_mask.nonzero()[0].tolist()  # if phrase_mask is boolean

            # Map mask positions to actual token ids for this sample
            token_ids = input_ids[b, non_zero_idx].tolist()

            # Decode while skipping special/padding tokens
            decoded = tokenizer.decode(
                [tid for tid in token_ids if tid not in tokenizer.all_special_ids],
                skip_special_tokens=True
            )
            phrase_list.append(decoded)

        pred["phrase"] = phrase_list

    return predictions


def nms_numpy(boxes, scores, iou_threshold):
    """
    Non-Maximum Suppression using NumPy and OpenCV.

    Args:
        boxes: np.array of shape (N, 4) in format [x1, y1, x2, y2]
        scores: np.array of shape (N,)
        iou_threshold: float, IoU threshold for suppression

    Returns:
        np.array: indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    # Convert to format expected by OpenCV NMS: [x, y, width, height]
    boxes_cv = np.zeros_like(boxes)
    boxes_cv[:, 0] = boxes[:, 0]  # x1 -> x
    boxes_cv[:, 1] = boxes[:, 1]  # y1 -> y
    boxes_cv[:, 2] = boxes[:, 2] - boxes[:, 0]  # x2 - x1 -> width
    boxes_cv[:, 3] = boxes[:, 3] - boxes[:, 1]  # y2 - y1 -> height

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_cv.tolist(),
        scores.tolist(),
        score_threshold=0.0,  # We already filtered by confidence
        nms_threshold=iou_threshold
    )

    if len(indices) > 0:
        return indices.flatten().astype(np.int32)
    return np.array([], dtype=np.int32)


def threshold_predictions_numpy(pred_text_masks, pred_scores, pred_boxes, pred_masks,
                                conf_threshold=0.0,
                                nms_threshold=0.0):
    """
    NumPy version of threshold_predictions for TensorRT inference.

    Args:
        predictions (List[Dict]): List of predictions from the model.
            Each prediction dict should contain numpy arrays instead of tensors.
        conf_threshold (float): Confidence score threshold.
        nms_threshold (float): NMS threshold.
    Returns:
        List[Dict]: List of filtered predictions, with empty arrays if no valid predictions.
    """
    filtered_predictions = []

    for pred_text_mask, pred_score, pred_box, pred_mask in zip(pred_text_masks, pred_scores, pred_boxes, pred_masks):
        image_size = pred_mask.shape[1:]   # tuple or list (H, W)
        image_names = pred_box.shape[0]  # string

        assert pred_box.shape[0] == pred_score.shape[0], "Boxes and scores must have same length"

        # Handle empty predictions
        if pred_box.shape[0] == 0:
            filtered_predictions.append({
                'image_size': image_size,
                'image_names': image_names,
                'boxes': np.empty((0, 4), dtype=np.float32),
                'scores': np.empty((0,), dtype=np.float32),
                'masks': np.empty((0, *image_size), dtype=np.float32),
                'text_mask': np.empty((0,), dtype=np.float32),
            })
            continue

        # Apply confidence threshold
        keep = pred_score >= conf_threshold

        if np.sum(keep) == 0:
            # No predictions survive confidence threshold
            filtered_predictions.append({
                'image_size': image_size,
                'image_names': image_names,
                'boxes': np.empty((0, 4), dtype=np.float32),
                'scores': np.empty((0,), dtype=np.float32),
                'masks': np.empty((0, *image_size), dtype=np.float32),
                'text_mask': np.empty((0,), dtype=np.float32),
            })
            continue

        # Apply NMS if threshold > 0
        if nms_threshold > 0:
            # First filter by confidence threshold
            pred_text_mask = pred_text_mask[keep]
            pred_box = pred_box[keep]
            pred_score = pred_score[keep]
            pred_mask = pred_mask[keep]

            # Apply NMS
            nms_indices = nms_numpy(pred_box, pred_score, nms_threshold)
            keep = nms_indices

        # Build final filtered prediction
        filtered_predictions.append({
            'image_size': image_size,
            'image_names': image_names,
            'boxes': pred_box[keep],
            'scores': pred_score[keep],
            'masks': pred_mask[keep],
            'text_mask': pred_text_mask[keep],
        })

    return filtered_predictions


def get_valid_mask_numpy(union_mask_logits, masks, threshold=0.5):
    """
    Convert PyTorch get_valid_mask function to NumPy/OpenCV.

    Args:
        union_mask_logits: np.array of shape (B, 2, S_h, S_w), logits for background/foreground
        masks: np.array of shape (B, N, H, W), individual object masks
        threshold: float, minimum intersection-over-mask ratio to keep the mask

    Returns:
        List[np.array]: List of boolean arrays, each of shape (N,) for each batch item
    """
    B, C, _, _ = union_mask_logits.shape
    assert C == 2, f"Expected 2 channels for background/foreground, got {C}"

    # Step 1: Convert union logits to binary predictions
    # PyTorch: preds = union_mask_logits.argmax(dim=1, keepdim=True).float()
    preds = np.argmax(union_mask_logits, axis=1, keepdims=True).astype(np.float32)  # (B, 1, S_h, S_w)

    # Step 2: Resize predictions to match mask resolution
    # PyTorch: preds_up = F.interpolate(preds, size=(H, W), mode="bilinear", align_corners=False)
    preds_bin = []
    for b in range(B):
        # OpenCV resize expects (width, height), so we use (W, H)
        H, W, N = masks[b].shape
        resized = cv2.resize(preds[b, 0], (W, H), interpolation=cv2.INTER_LINEAR)
        preds_bin.append((sigmoid(resized) > 0.5).astype(np.float32))

    # Step 3: Apply sigmoid and binarize
    # PyTorch: preds_bin = (preds_up.sigmoid() > 0.5).float()

    # Step 4: Validate each mask against union mask
    keep_mask = []
    for b in range(B):
        pred_mask = preds_bin[b]
        gt_masks = masks[b]
        # Compute intersection for each mask
        # PyTorch: inter = (pred_mask * gt_masks).flatten(1).sum(1)
        intersections = []
        areas = []

        for n in range(N):
            # Element-wise multiplication gives intersection
            intersection_pixels = pred_mask * gt_masks[..., n]  # (H, W)
            inter = intersection_pixels.sum()  # Total intersection pixels

            # Compute mask area
            area = gt_masks[n].sum() + 1e-6  # Add small epsilon to avoid division by zero

            intersections.append(inter)
            areas.append(area)

        # Convert to arrays for vectorized operations
        intersections = np.array(intersections)  # (N,)
        areas = np.array(areas)  # (N,)

        # Compute intersection-over-mask ratios
        ratios = intersections / areas  # (N,)

        # Apply threshold
        keep = ratios > threshold  # (N,) boolean array
        keep_mask.append(keep)

    keep_mask = np.array(keep_mask)

    return keep_mask


def post_process_v2(pred_logits, pred_boxes, pred_masks, target_sizes, pos_maps, no_targets, union_mask_logits=None, text_threshold=0.0, num_select=300, task="OD"):
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
    if no_targets is not None:
        # no_targets = no_targets.squeeze(2)
        no_targets = (sigmoid(no_targets) > 0.5).astype(np.int32)
    else:
        no_targets = None

    bs = pred_logits.shape[0]
    # Sigmoid
    # prob_to_token = sigmoid(pred_logits).reshape((bs, pred_logits.shape[1], -1))
    prob_to_token = sigmoid(pred_logits)

    if task == "OD":
        for label_ind in range(len(pos_maps)):
            if pos_maps[label_ind].sum() != 0:
                pos_maps[label_ind] = pos_maps[label_ind] / pos_maps[label_ind].sum()

        prob_to_label = prob_to_token @ pos_maps.T
    else:
        prob_to_label = prob_to_token

    prob = prob_to_label  # (bs, nq, n_classes)

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

    if task == "VG":
        text_masks = prob_to_token > text_threshold
        text_masks = np.take_along_axis(text_masks, np.repeat(np.expand_dims(topk_boxes, -1), text_masks.shape[-1], axis=-1), axis=1)

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

    if task == "VG" and union_mask_logits is not None:
        valid_masks = get_valid_mask_numpy(union_mask_logits, masks, threshold=0.5)
        if valid_masks.any():
            text_masks_list, scores_list, boxes_list, masks_list, no_targets_list = [], [], [], [], []
            for i in range(len(valid_masks)):
                text_masks_list.append(text_masks[i][valid_masks[i]])
                scores_list.append(scores[i][valid_masks[i]])
                boxes_list.append(boxes[i][valid_masks[i]])
                masks_list.append(masks[i].transpose(2, 0, 1)[valid_masks[i]])
                no_targets_list.append(no_targets[i])
            return text_masks_list, scores_list, boxes_list, masks_list, no_targets_list

        return text_masks, scores, boxes, masks, no_targets

    return labels, scores, boxes, masks


class MultiTaskImageBatcher(ImageBatcher):
    """Multi-task image batcher for Mask Grounding DINO supporting both OD and VG tasks."""

    def __init__(self, input_path, shape, dtype,
                 max_num_images=None, exact_batches=False, preprocessor="EfficientDet",
                 img_std=[0.229, 0.224, 0.225],
                 img_mean=[0.485, 0.456, 0.406],
                 data_path=None,
                 task='VG'):
        """Initialize.

        Args:
            input_path: The input directory to read images from. (list or str)
            shape: The tensor shape of the batch to prepare, either in channels_first or channels_last format.
            dtype: The (numpy) datatype to cast the batched data to.
            max_num_images: The maximum number of images to read from the directory.
            exact_batches: This defines how to handle a number of images that is not an exact
                multiple of the batch size. If false, it will pad the final batch with zeros to reach
                the batch size. If true, it will *remove* the last few images in excess of a batch size
                multiple, to guarantee batches are exact (useful for calibration).
            preprocessor: Set the preprocessor to use, depending on which network is being used.
            img_std: Set img std for DDETR use case
            img_mean: Set img mean for DDETR use case
        """

        def is_image(path):
            return os.path.isfile(path) and path.lower().endswith(VALID_IMAGE_EXTENSIONS)

        if task == 'OD':
            self.evaluation = False
            self.images = []
            if isinstance(input_path, (ListConfig, list)):
                # Multiple directories
                for image_dir in input_path:
                    self.images.extend(str(p.resolve()) for p in Path(image_dir).glob("**/*") if p.suffix in VALID_IMAGE_EXTENSIONS)
                # Shuffle so that we sample uniformly from the sequence
                random.shuffle(self.images)
            else:
                if os.path.isdir(input_path):
                    self.images = [str(p.resolve()) for p in Path(input_path).glob("**/*") if p.suffix in VALID_IMAGE_EXTENSIONS]
                    self.images.sort()
                elif os.path.isfile(input_path):
                    if is_image(input_path):
                        self.images.append(input_path)
        else:
            self.images = []
            self.captions = []
            img_dir = input_path[0]
            with open(data_path, encoding='utf-8') as f:
                loaded_data = [json.loads(line) for line in f]
            for data in loaded_data:
                self.images.append(os.path.join(img_dir, data['image_path']))
                self.captions.append(data['expression'])

        self.num_images = len(self.images)
        if self.num_images < 1:
            logging.error("No valid %s images found in %s", '/'.join(VALID_IMAGE_EXTENSIONS), input_path)
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        self.channel = -1
        if self.shape[1] in [3, 1]:
            self.format = "channels_first"
            self.height = self.shape[2]
            self.width = self.shape[3]
            self.channel = self.shape[1]
        elif self.shape[3] in [3, 1]:
            self.format = "channels_last"
            self.height = self.shape[1]
            self.width = self.shape[2]
            self.channel = self.shape[3]
        assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            raise ValueError("Not enough images to create batches")
        self.images = self.images[0:self.num_images]

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            if task == 'VG':
                self.batches.append((self.images[start:end], self.captions[start:end]))
            else:
                self.batches.append(self.images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

        self.preprocessor = preprocessor
        self.img_std = img_std
        self.img_mean = img_mean

    def get_batch_VG(self):
        """Retrieve the batches.

        This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        """
        for _, (batch_image_paths, batch_captions) in enumerate(self.batches):
            batch_images = np.zeros(self.shape, dtype=self.dtype)
            batch_scales = [None] * len(batch_image_paths)
            for idx, (image_path, _) in enumerate(zip(batch_image_paths, batch_captions)):
                self.image_index += 1
                batch_images[idx], batch_scales[idx] = self.preprocess_image(image_path)
            self.batch_index += 1
            batch_data = [batch_images, batch_captions]

            yield batch_data, batch_image_paths, batch_scales


def tokenize_captions(task, tokenizer, cat_list, caption, max_text_len=256):
    """
    Tokenize text captions and generate attention masks for Mask Grounding DINO.

    Args:
        task (str): Task type - "OD" for Object Detection or "VG" for Visual Grounding
        tokenizer: HuggingFace tokenizer for text encoding
        cat_list (list): List of category names/labels
        caption (list): Text caption(s) to tokenize
        max_text_len (int): Maximum sequence length. Defaults to 256.

    Returns:
        tuple: (input_ids, attention_mask, position_ids, token_type_ids,
                text_self_attention_masks, pos_map)
                where pos_map is only generated for OD task, None for VG.
    """
    special_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    tokenized = tokenizer(caption, padding="max_length", return_tensors="np", max_length=max_text_len)

    if task == "OD":
        label_list = np.arange(len(cat_list))
        pos_map = create_positive_map(tokenized, label_list, cat_list, caption[0], max_text_len=max_text_len)
    else:
        pos_map = None

    (
        text_self_attention_masks,
        position_ids,
    ) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens)

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


def save_evaluation_results(eval_results, results_dir, task_type):
    """
    Save evaluation results in multiple formats for comprehensive analysis.

    Args:
        eval_results (dict): Dictionary containing evaluation metrics
        results_dir (str): Directory to save results
        task_type (str): Type of task (VG or OD)
    """
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save detailed JSON results
    json_file = os.path.join(results_dir, f"detailed_results_{task_type}_{timestamp}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, default=str)
    logging.info("Detailed results saved to: %s", json_file)

    # 2. Save key metrics as CSV
    csv_file = os.path.join(results_dir, f"key_metrics_{task_type}_{timestamp}.csv")

    # Extract key numeric metrics
    key_metrics = {}
    for key, value in eval_results.items():
        if isinstance(value, (int, float)) and not isinstance(value, dict):
            key_metrics[key] = float(value)

    # Create DataFrame and save CSV
    df = pd.DataFrame([key_metrics])
    df.to_csv(csv_file, index=False)
    logging.info("Key metrics CSV saved to: %s", csv_file)

    # 3. Save human-readable summary
    summary_file = os.path.join(results_dir, f"evaluation_summary_{task_type}_{timestamp}.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"MASK GROUNDING DINO EVALUATION RESULTS - {task_type} TASK\n")
        f.write("=" * 80 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task Type: {task_type}\n")
        f.write(f"Total Samples: {eval_results.get('num_images', 'N/A')}\n\n")

        # Core Performance Metrics
        f.write("CORE PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        if 'gIoU' in eval_results:
            f.write(f"Global IoU (gIoU):           {eval_results['gIoU']:.1f}%\n")
        if 'cIoU' in eval_results:
            f.write(f"Cumulative IoU (cIoU):       {eval_results['cIoU']:.1f}%\n")
        if 'mean_iou' in eval_results:
            f.write(f"Mean IoU:                    {eval_results['mean_iou'] * 100:.1f}%\n")
        if 'accuracy' in eval_results:
            f.write(f"Overall Accuracy:            {eval_results['accuracy'] * 100:.1f}%\n")

        # mAP Metrics
        f.write("\nMAP METRICS:\n")
        f.write("-" * 40 + "\n")
        if 'mAP50' in eval_results:
            f.write(f"mAP@IoU=0.5:                 {eval_results['mAP50']:.1f}%\n")
        if 'mAP' in eval_results:
            f.write(f"mAP (0.5-0.95):              {eval_results['mAP']:.1f}%\n")

        # Precision at High IoU Thresholds
        f.write("\nPRECISION AT HIGH IoU THRESHOLDS:\n")
        f.write("-" * 40 + "\n")
        if 'Pr@0.7' in eval_results:
            f.write(f"Precision@IoU=0.7:           {eval_results['Pr@0.7']:.1f}%\n")
        if 'Pr@0.8' in eval_results:
            f.write(f"Precision@IoU=0.8:           {eval_results['Pr@0.8']:.1f}%\n")
        if 'Pr@0.9' in eval_results:
            f.write(f"Precision@IoU=0.9:           {eval_results['Pr@0.9']:.1f}%\n")

        # Classification Accuracy
        f.write("\nCLASSIFICATION ACCURACY:\n")
        f.write("-" * 40 + "\n")
        if 'N_acc' in eval_results:
            f.write(f"True Positive Rate (N_acc):  {eval_results['N_acc']:.1f}%\n")
        if 'T_acc' in eval_results:
            f.write(f"True Negative Rate (T_acc):  {eval_results['T_acc']:.1f}%\n")

        # IoU Distribution
        f.write("\nIoU DISTRIBUTION:\n")
        f.write("-" * 40 + "\n")
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            key = f"IoU@{thresh}"
            if key in eval_results:
                f.write(f"IoU@{thresh}:                   {eval_results[key] * 100:.1f}%\n")

        # Confusion Matrix
        if 'confusion_matrix' in eval_results:
            cm = eval_results['confusion_matrix']
            f.write("\nCONFUSION MATRIX:\n")
            f.write("-" * 40 + "\n")
            f.write(f"True Positives (TP):         {cm.get('TP', 0)}\n")
            f.write(f"True Negatives (TN):         {cm.get('TN', 0)}\n")
            f.write(f"False Positives (FP):        {cm.get('FP', 0)}\n")
            f.write(f"False Negatives (FN):        {cm.get('FN', 0)}\n")

        # Additional Info
        f.write("\nADDITIONAL INFORMATION:\n")
        f.write("-" * 40 + "\n")
        if 'not_empty_predictions' in eval_results:
            f.write(f"Non-empty predictions:       {eval_results['not_empty_predictions']}\n")
        if 'iou_std' in eval_results:
            f.write(f"IoU Standard Deviation:      {eval_results['iou_std']:.3f}\n")

        f.write("\n" + "=" * 80 + "\n")

    logging.info("Human-readable summary saved to: %s", summary_file)

    # 4. Save compact results for quick reference
    compact_file = os.path.join(results_dir, "latest_results.json")
    compact_results = {
        "timestamp": timestamp,
        "task_type": task_type,
        "gIoU": eval_results.get('gIoU', 0.0),
        "cIoU": eval_results.get('cIoU', 0.0),
        "mAP50": eval_results.get('mAP50', 0.0),
        "mAP": eval_results.get('mAP', 0.0),
        "accuracy": eval_results.get('accuracy', 0.0) * 100,
        "mean_iou": eval_results.get('mean_iou', 0.0) * 100,
        "Pr@0.7": eval_results.get('Pr@0.7', 0.0),
        "Pr@0.8": eval_results.get('Pr@0.8', 0.0),
        "Pr@0.9": eval_results.get('Pr@0.9', 0.0),
        "num_samples": eval_results.get('num_images', 0)
    }

    with open(compact_file, "w", encoding="utf-8") as f:
        json.dump(compact_results, f, indent=2)
    logging.info("Compact results saved to: %s", compact_file)

    # Log evaluation metrics to status logger
    s_logger = status_logging.get_status_logger()
    s_logger.kpi = compact_results
    s_logger.write(
        status_level=status_logging.Status.RUNNING,
        message="Evaluation metrics computed successfully"
    )
    logging.info("Evaluation metrics logged to status logger")

    logging.info("\n" + "=" * 80)
    logging.info("EVALUATION RESULTS SAVED TO:")
    logging.info("=" * 80)
    logging.info("📊 Detailed JSON:      %s", json_file)
    logging.info("📈 Key Metrics CSV:    %s", csv_file)
    logging.info("📄 Summary Report:     %s", summary_file)
    logging.info("⚡ Latest Results:     %s", compact_file)
    logging.info("=" * 80)


def get_rgb_from_color_name(color_name):
    """
    Get RGB values (0-255) from a color name.

    Args:
        color_name (str): Name of the color (case-insensitive)

    Returns:
        tuple: (R, G, B) values as integers from 0-255

    Raises:
        ValueError: If color name is not recognized
    """
    if isinstance(color_name, tuple):
        return color_name
    # Normalize input
    color_name = color_name.lower().strip()

    if color_name in default_colors:
        return default_colors[color_name]
    raise ValueError("Invalid color name")


def random_color(color_list):
    """
    Randomly pick one color from a list of colors.

    Args:
        color_list (list): List of color names or RGB tuples

    Returns:
        The randomly selected color (same type as input elements)

    Raises:
        ValueError: If color_list is empty
    """
    if not color_list:
        raise ValueError("Color list cannot be empty")

    return random.choice(color_list)
