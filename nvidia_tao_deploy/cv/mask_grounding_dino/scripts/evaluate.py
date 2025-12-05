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

"""Standalone TensorRT evaluation."""

import os
import copy
import json
import logging
import six
import numpy as np
import tensorrt as trt
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import pycocotools.mask as maskUtils

from nvidia_tao_core.config.mask_grounding_dino.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.mask_grounding_dino.dataloader import MGDINOCOCOLoader
from nvidia_tao_deploy.cv.mask_grounding_dino.utils import tokenize_captions

from nvidia_tao_deploy.cv.mask_grounding_dino.inferencer import MaskGDINOInferencer

from nvidia_tao_deploy.cv.mask_grounding_dino.metrics.coco_metric import MaskGDinoEvaluationMetric
from nvidia_tao_deploy.cv.mask_grounding_dino.metrics.vg_metric import VGEvaluationMetric
from nvidia_tao_deploy.cv.mask_grounding_dino.utils import (
    post_process_v2, convert_jsonl_to_json, threshold_predictions_numpy,
    get_phrase_from_expression_numpy, save_evaluation_results
)

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name='grounding_dino', mode='evaluate')
def main(cfg: ExperimentConfig) -> None:
    """Mask Grounding DINO TRT evaluation."""
    if not os.path.exists(cfg.evaluate.trt_engine):
        raise FileNotFoundError(f"Provided evaluate.trt_engine at {cfg.evaluate.trt_engine} does not exist!")

    assert cfg.dataset.batch_size == 1, "Evaluation batch size must be 1."
    max_text_len = cfg.model.max_text_len
    num_select = cfg.model.num_select
    conf_threshold = cfg.evaluate.conf_threshold
    task = cfg.dataset.test_data_sources.data_type

    # Convert JSONL to JSON if needed
    original_json_file = cfg.dataset.test_data_sources.json_file
    if original_json_file and original_json_file.lower().endswith('.jsonl'):
        logging.info("Detected JSONL file: %s", original_json_file)
        converted_json_file = convert_jsonl_to_json(original_json_file)
        json_file_to_use = converted_json_file
    else:
        json_file_to_use = original_json_file

    # Choose appropriate evaluation metric based on task
    if task == "VG":
        eval_metric = VGEvaluationMetric(
            gt_json_file=json_file_to_use,
            iou_threshold=cfg.evaluate.conf_threshold)
        logging.info("Using VG evaluation metric for Visual Grounding task")
    else:
        # Check if ground truth has segmentation annotations
        with open(json_file_to_use, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        has_segmentation = any('segmentation' in ann for ann in gt_data.get('annotations', []))

        eval_metric = MaskGDinoEvaluationMetric(
            json_file_to_use,
            include_mask=has_segmentation)

        if has_segmentation:
            logging.info("Using COCO evaluation metric for Object Detection with mask evaluation")
        else:
            logging.info("Using COCO evaluation metric for Object Detection (bbox only)")

    # Create TensorRT inferencer with dynamic shape support
    logger.info("Creating TensorRT inferencer with dynamic shape support...")
    trt_infer = MaskGDINOInferencer(cfg.evaluate.trt_engine,
                                    batch_size=cfg.dataset.batch_size,
                                    num_classes=max_text_len,
                                    task=cfg.dataset.test_data_sources.data_type)
    logger.info("TensorRT inferencer successfully created!")
    logger.info("Loaded engine with %d inputs and %d outputs", len(trt_infer.input_tensors), len(trt_infer.output_tensors))

    c, h, w = trt_infer.input_tensors[0].shape

    dl = MGDINOCOCOLoader(
        val_json_file=json_file_to_use,
        shape=(cfg.dataset.batch_size, c, h, w),
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
        batch_size=cfg.dataset.batch_size,
        data_format="channels_first",
        image_std=cfg.dataset.augmentation.input_std,
        image_dir=cfg.dataset.test_data_sources.image_dir,
        eval_samples=None)

    # Log dataloader statistics
    logging.info("Dataloader created with %d image-caption pairs", len(dl.image_caption_pairs))
    logging.info("Number of batches: %d", dl.n_batches)
    logging.info("Batch size: %d", dl.batch_size)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder_type)

    # Initialize input variables
    input_ids = None
    attention_mask = None
    position_ids = None
    token_type_ids = None
    text_self_attention_masks = None
    pos_map = None

    if task == "OD":
        category_dict = dl.coco.loadCats(dl.coco.getCatIds())
        cat_lists = [item['name'] for item in category_dict]
        captions = [" . ".join(cat_lists) + ' .'] * cfg.dataset.batch_size
        input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map = tokenize_captions(task, tokenizer, cat_lists, captions, max_text_len)
    else:
        cat_lists = None

    predictions = {
        'detection_scores': [],
        'detection_boxes': [],
        'detection_classes': [],
        'detection_masks': [],
        'source_id': [],
        'image_info': [],
        'num_detections': [],
        'target_phrases': [],  # Store target phrases for VG evaluation
        'sent_ids': []         # Store sentence IDs for exact GT matching
    }

    def evaluation_preds(preds):
        """Evaluate predictions and compute metrics.

        Args:
            preds: Dictionary containing prediction arrays

        Returns:
            Dictionary of evaluation metrics
        """
        # Essential to avoid modifying the source dict
        _preds = copy.deepcopy(preds)

        # For OD task, deduplicate predictions by image_id since the dataloader
        # may create multiple samples per image (one per annotation)
        if task == "OD":
            # Find unique image IDs and keep only the first occurrence
            seen_images = {}
            for idx, img_id in enumerate(_preds['source_id']):
                # Extract scalar image_id from list format
                actual_img_id = img_id[0] if isinstance(img_id, (list, np.ndarray)) else img_id
                if actual_img_id not in seen_images:
                    seen_images[actual_img_id] = idx

            # Keep only unique image predictions
            unique_indices = sorted(seen_images.values())
            if len(unique_indices) < len(_preds['source_id']):
                for key in _preds.keys():
                    if isinstance(_preds[key], list):
                        _preds[key] = [_preds[key][i] for i in unique_indices]

        if task == "VG":
            # For VG: Keep nested list structure, no concatenation
            # Each image can have different numbers of detections
            pass  # No concatenation needed for VG

            # Extract target phrases and sentence IDs for VG evaluation
            target_phrases = _preds.get('target_phrases', [])
            sent_ids = _preds.get('sent_ids', [])
            eval_results = eval_metric.predict_metric_fn(_preds, target_phrases=target_phrases, sent_ids=sent_ids)
        else:
            # For OD: Concatenate prediction arrays
            # Skip metadata keys that should not be concatenated
            skip_keys = {'target_phrases', 'sent_ids'}

            for k, _ in six.iteritems(_preds):
                if k not in skip_keys:
                    _preds[k] = np.concatenate(_preds[k], axis=0)

            eval_results = eval_metric.predict_metric_fn(_preds)

        return eval_results

    for i, (imgs, scale, source_id, labels, captions, sent_ids) in enumerate(tqdm(dl, total=len(dl), desc="Producing predictions")):
        image = np.array(imgs)

        if task == "VG":
            # Use captions from the dataloader for VG task (now individual captions per sample)
            batch_captions = captions  # captions is now a list of individual captions

            input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map = tokenize_captions(task, tokenizer, cat_lists, batch_captions, max_text_len)

        image_info = []
        target_sizes = []
        for idx, label in enumerate(labels):
            image_info.append([label[-1][0], label[-1][1], scale[idx], label[-1][2], label[-1][3]])
            # target_sizes needs to [W, H, W, H]
            target_sizes.append([label[-1][3], label[-1][2], label[-1][3], label[-1][2]])
        image_info = np.array(image_info)
        inputs = (image, input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks)

        if task == "OD":
            pred_logits, pred_boxes, pred_masks = trt_infer.infer(inputs)
            no_targets = None
            union_mask_logits = None
            class_labels, scores, boxes, masks = post_process_v2(pred_logits, pred_boxes, pred_masks, target_sizes, pos_map, num_select, task=task)

            # Convert class labels from 0-indexed to 1-indexed to match COCO category IDs
            # Model outputs class indices starting from 0, but COCO category IDs start from 1
            class_labels = class_labels + 1
        else:
            pred_logits, pred_boxes, pred_masks, no_targets, union_mask_logits = trt_infer.infer(inputs)
            text_labels, scores, boxes, masks, no_targets = post_process_v2(
                pred_logits, pred_boxes, pred_masks, target_sizes, pos_map,
                no_targets, union_mask_logits,
                text_threshold=cfg.evaluate.test_threshold,
                num_select=num_select, task=task)
            filtered_res = threshold_predictions_numpy(text_labels, scores, boxes, masks, conf_threshold=cfg.evaluate.conf_threshold, nms_threshold=cfg.evaluate.nms_threshold)
            filtered_res = get_phrase_from_expression_numpy(filtered_res, tokenizer, input_ids)
            class_labels = [f["phrase"] for f in filtered_res]
            scores = [f["scores"] for f in filtered_res]
            boxes = [f["boxes"] for f in filtered_res]
            masks = [f["masks"] for f in filtered_res]

        # Handle mask encoding and box format conversion
        if task == "VG":
            # For VG: masks and boxes are lists of individual arrays
            encoded_masks = []
            for mask_array in masks:
                if len(mask_array.shape) == 2:  # Single mask
                    binary_mask = (mask_array > 0.5).astype(np.uint8)
                    rle_mask = maskUtils.encode(np.asfortranarray(binary_mask))
                    encoded_masks.append(rle_mask)
                elif len(mask_array.shape) == 3:  # Multiple masks
                    for instance_mask in mask_array:
                        binary_mask = (instance_mask > 0.5).astype(np.uint8)
                        rle_mask = maskUtils.encode(np.asfortranarray(binary_mask))
                        encoded_masks.append(rle_mask)

            # Convert boxes from (x1,y1,x2,y2) to (x,y,w,h) for each detection
            num_images = len(boxes)
            for img_idx in range(num_images):
                num_boxes_per_image = len(boxes[img_idx])
                for j in range(num_boxes_per_image):
                    boxes[img_idx][j][2:] -= boxes[img_idx][j][:2]

        else:  # OD task
            # For OD: masks and boxes are batch arrays
            segments = masks[0] > conf_threshold
            segments = segments.transpose((2, 0, 1))
            encoded_masks = []
            for instance_mask in segments:
                # Apply threshold and convert to binary (should already be binary from conf_threshold)
                binary_mask = (instance_mask > 0.5).astype(np.uint8)
                rle_mask = maskUtils.encode(np.asfortranarray(binary_mask))
                encoded_masks.append(rle_mask)
            # Convert to xywh
            boxes[:, :, 2:] -= boxes[:, :, :2]

        predictions['detection_classes'].append(class_labels)
        predictions['detection_scores'].append(scores)
        predictions['detection_boxes'].append(boxes)
        predictions['detection_masks'].append(encoded_masks)
        predictions['num_detections'].append(np.array([num_select] * cfg.dataset.batch_size).astype(np.int32))
        predictions['image_info'].append(image_info)
        predictions['source_id'].append(source_id)

        # Store target phrases and sentence IDs for VG evaluation
        if task == "VG":
            # For VG task, captions contain the target referring expressions
            predictions['target_phrases'].append(captions[0] if captions else "N/A")
            predictions['sent_ids'].append(sent_ids[0] if sent_ids else -1)
        else:
            # For OD task, no specific target phrases or sentence IDs
            predictions['target_phrases'].append("N/A")
            predictions['sent_ids'].append(-1)

    eval_results = evaluation_preds(preds=predictions)

    # Save comprehensive evaluation results
    save_evaluation_results(eval_results, cfg.results_dir, cfg.dataset.test_data_sources.data_type)

    # Clean up converted JSON file if it was created
    if (original_json_file and original_json_file.lower().endswith('.jsonl') and
            json_file_to_use != original_json_file and os.path.exists(json_file_to_use)):
        try:
            os.remove(json_file_to_use)
            logging.info("Cleaned up converted file: %s", json_file_to_use)
        except OSError as e:
            logging.warning("Failed to clean up converted file %s: %s", json_file_to_use, e)

    logging.info("Finished evaluation.")


if __name__ == '__main__':
    main()
