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

"""Standalone TensorRT inference."""

import os
import logging
import numpy as np
from PIL import Image
import tensorrt as trt
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nvidia_tao_core.config.mask_grounding_dino.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner

from nvidia_tao_deploy.cv.mask_grounding_dino.utils import tokenize_captions
from nvidia_tao_deploy.cv.mask_grounding_dino.utils import post_process_v2, MultiTaskImageBatcher, threshold_predictions_numpy, get_phrase_from_expression_numpy
from nvidia_tao_deploy.cv.mask_grounding_dino.inferencer import MaskGDINOInferencer

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="infer", schema=ExperimentConfig
)
@monitor_status(name='mask_grounding_dino', mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """Mask Grounding DINO TRT Inference."""
    if not os.path.exists(cfg.inference.trt_engine):
        raise FileNotFoundError(f"Provided inference.trt_engine at {cfg.inference.trt_engine} does not exist!")

    # assert cfg.dataset.batch_size == 1, "Inference batch size must be 1."
    max_text_len = cfg.model.max_text_len
    num_select = cfg.model.num_select
    color_map = cfg.inference.color_map or {}
    task = cfg.dataset.infer_data_sources.data_type

    # Initialize variables that may be assigned conditionally
    input_ids = attention_mask = position_ids = token_type_ids = text_self_attention_masks = None
    inv_classes = None

    trt_infer = MaskGDINOInferencer(cfg.inference.trt_engine,
                                    batch_size=cfg.dataset.batch_size,
                                    num_classes=max_text_len,
                                    task=cfg.dataset.infer_data_sources.data_type)

    # Handle potential dynamic shape access
    try:
        c, h, w = trt_infer.input_tensors[0].shape
    except (IndexError, AttributeError) as e:
        logger.warning("Could not extract shape from input tensor: %s", e)
        # Fallback to default shape
        c, h, w = 3, 480, 480
        logger.info("Using fallback shape: (%s, %s, %s)", c, h, w)
    # Handle image_dir: if it's a string, wrap in list; if already a list, use as-is
    image_dir = cfg.dataset.infer_data_sources.image_dir
    if isinstance(image_dir, str):
        image_dir_list = [image_dir]
    else:
        image_dir_list = list(image_dir)

    batcher = MultiTaskImageBatcher(image_dir_list,
                                    (cfg.dataset.batch_size, c, h, w),
                                    trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
                                    data_path=cfg.dataset.infer_data_sources.captions,
                                    preprocessor="DDETR",
                                    task=task)

    if task == "OD":
        cat_list = list(cfg.dataset.infer_data_sources["captions"])
        caption = [" . ".join(cat_list) + ' .'] * cfg.dataset.batch_size
        inv_classes = dict(enumerate(cat_list))

        for cat in cat_list:
            if cat not in color_map:
                color_map[cat] = tuple(np.random.randint(256, size=3))
    else:
        cat_list = None

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder_type)

    if task == "OD":
        input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map = tokenize_captions(task, tokenizer, cat_list, caption, max_text_len)

    # Create results directories
    output_annotate_root = os.path.join(cfg.results_dir, "images_annotated")
    output_label_root = os.path.join(cfg.results_dir, "labels")

    os.makedirs(output_annotate_root, exist_ok=True)
    os.makedirs(output_label_root, exist_ok=True)

    if task == "VG":
        batch_generator = batcher.get_batch_VG()
    else:
        batch_generator = batcher.get_batch()

    for batches, img_paths, scales in tqdm(batch_generator, total=batcher.num_batches, desc="Producing predictions"):
        # Handle last batch as we artifically pad images for the last batch idx
        len_batches = batches[0].shape[0] if task == "VG" else len(batches)

        if len(img_paths) != len_batches:
            batches = batches[:len(img_paths)]

        if task == "VG":
            (images, captions) = batches
            # print(f"images: {img_paths}, captions: {captions}")  # Disabled for large datasets
        else:
            images = batches

        if task == "VG":
            input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map = tokenize_captions(task, tokenizer, cat_list, captions, max_text_len)

        inputs = (images, input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks)

        if task == "OD":
            pred_logits, pred_boxes, pred_masks = trt_infer.infer(inputs)
            no_targets, union_mask_logits = None, None
        else:
            pred_logits, pred_boxes, pred_masks, no_targets, union_mask_logits = trt_infer.infer(inputs)

        target_sizes = []
        for batch, scale in zip(images, scales):
            _, new_h, new_w = batch.shape
            orig_h, orig_w = int(scale[0] * new_h), int(scale[1] * new_w)
            target_sizes.append([orig_w, orig_h, orig_w, orig_h])

        if task == "VG":
            text_labels, scores, boxes, masks, no_targets = post_process_v2(pred_logits, pred_boxes, pred_masks, target_sizes, pos_map, no_targets, union_mask_logits, text_threshold=cfg.inference.text_threshold, num_select=num_select, task=task)
            filtered_res = threshold_predictions_numpy(text_labels, scores, boxes, masks, conf_threshold=cfg.inference.conf_threshold, nms_threshold=cfg.inference.nms_threshold)
            filtered_res = get_phrase_from_expression_numpy(filtered_res, tokenizer, input_ids)
            class_labels = [f["phrase"] for f in filtered_res]
            scores = [f["scores"] for f in filtered_res]
            boxes = [f["boxes"] for f in filtered_res]
            masks = [f["masks"] for f in filtered_res]
        else:
            class_labels, scores, boxes, masks = post_process_v2(pred_logits, pred_boxes, pred_masks, target_sizes, pos_map, num_select, task=task)

        if task == "VG":
            y_pred_valid = [class_labels[i] + scores[i].tolist() + boxes[i].tolist() for i in range(len_batches)]
        else:
            y_pred_valid = np.concatenate([class_labels[..., None], scores[..., None], boxes], axis=-1)  # bs, n_select, 6

        # Track image name counts for clean sequential naming
        image_counts = {}

        # Process each prediction individually
        for batch_idx, (img_path, pred, pred_masks) in enumerate(zip(img_paths, y_pred_valid, masks)):
            # Load Image
            img = Image.open(img_path)
            img_filename = os.path.basename(img_path)
            base_name, ext = os.path.splitext(img_filename)

            # Generate unique filename for duplicate images
            if img_filename not in image_counts:
                image_counts[img_filename] = 0
                unique_filename = img_filename  # First occurrence uses original name
            else:
                image_counts[img_filename] += 1
                unique_filename = f"{base_name}_{image_counts[img_filename]}{ext}"

            # Use VG-specific drawing for VG tasks, original method for OD tasks
            if task == "VG":
                # Extract prediction data for this batch
                batch_class_labels = class_labels[batch_idx]
                batch_scores = scores[batch_idx]
                batch_boxes = boxes[batch_idx]
                batch_masks = masks[batch_idx]

                # Filter out empty labels and low confidence
                valid_predictions = []
                for i, (label, score, box, mask) in enumerate(zip(batch_class_labels, batch_scores, batch_boxes, batch_masks)):
                    if label.strip() != "" and score >= cfg.inference.conf_threshold:
                        valid_predictions.append((label, score, box, mask))

                if valid_predictions:
                    # Prepare data for draw_bbox_VG_simple
                    final_labels = [pred[0] for pred in valid_predictions]
                    final_scores = [pred[1] for pred in valid_predictions]
                    final_boxes = [pred[2] for pred in valid_predictions]
                    final_masks = [pred[3] for pred in valid_predictions]

                    # Use draw_bbox_VG_simple for flattened data
                    bbox_img, label_strings = trt_infer.draw_bbox_vg_simple(
                        img, final_labels, final_scores, final_boxes, final_masks,
                        threshold=cfg.inference.conf_threshold, color_map=color_map
                    )
                else:
                    bbox_img = img
                    label_strings = []
            else:
                # Use original method for OD tasks
                bbox_img, label_strings = trt_infer.draw_bbox(img, pred, pred_masks, inv_classes, cfg.inference.conf_threshold, color_map)

            # Save with unique filename
            bbox_img.save(os.path.join(output_annotate_root, unique_filename))

            # Store labels with unique filename
            label_base_name, _ = os.path.splitext(unique_filename)
            label_file_name = os.path.join(output_label_root, label_base_name + ".txt")
            with open(label_file_name, "w", encoding="utf-8") as f:
                for l_s in label_strings:
                    f.write(l_s)

    logging.info("Inference results were saved at %s.", cfg.results_dir)


if __name__ == '__main__':
    main()
