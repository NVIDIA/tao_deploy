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

from nvidia_tao_deploy.cv.grounding_dino.utils import tokenize_captions
from nvidia_tao_deploy.cv.mask_grounding_dino.utils import post_process
from nvidia_tao_deploy.cv.mask_grounding_dino.inferencer import MaskGDINOInferencer
from nvidia_tao_deploy.utils.image_batcher import ImageBatcher

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

    assert cfg.dataset.batch_size == 1, "Inference batch size must be 1."
    max_text_len = cfg.model.max_text_len
    num_select = cfg.model.num_select
    color_map = cfg.inference.color_map or {}
    trt_infer = MaskGDINOInferencer(cfg.inference.trt_engine,
                                    batch_size=cfg.dataset.batch_size,
                                    num_classes=max_text_len)

    c, h, w = trt_infer.input_tensors[0].shape
    batcher = ImageBatcher(list(cfg.dataset.infer_data_sources.image_dir),
                           (cfg.dataset.batch_size, c, h, w),
                           trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
                           preprocessor="DDETR")

    cat_list = list(cfg.dataset.infer_data_sources["captions"])
    caption = [" . ".join(cat_list) + ' .'] * cfg.dataset.batch_size
    inv_classes = dict(enumerate(cat_list))

    for cat in cat_list:
        if cat not in color_map:
            color_map[cat] = tuple(np.random.randint(256, size=3))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder_type)
    input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map = tokenize_captions(tokenizer, cat_list, caption, max_text_len)

    # Create results directories
    output_annotate_root = os.path.join(cfg.results_dir, "images_annotated")
    output_label_root = os.path.join(cfg.results_dir, "labels")

    os.makedirs(output_annotate_root, exist_ok=True)
    os.makedirs(output_label_root, exist_ok=True)

    for batches, img_paths, scales in tqdm(batcher.get_batch(), total=batcher.num_batches, desc="Producing predictions"):
        # Handle last batch as we artifically pad images for the last batch idx
        if len(img_paths) != len(batches):
            batches = batches[:len(img_paths)]

        inputs = (batches, input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks)
        pred_logits, pred_boxes, pred_masks = trt_infer.infer(inputs)

        target_sizes = []
        for batch, scale in zip(batches, scales):
            _, new_h, new_w = batch.shape
            orig_h, orig_w = int(scale[0] * new_h), int(scale[1] * new_w)
            target_sizes.append([orig_w, orig_h, orig_w, orig_h])

        class_labels, scores, boxes, masks = post_process(pred_logits, pred_boxes, pred_masks, target_sizes, pos_map, num_select)
        y_pred_valid = np.concatenate([class_labels[..., None], scores[..., None], boxes], axis=-1)  # bs, n_select, 6

        for img_path, pred, pred_masks in zip(img_paths, y_pred_valid, masks):
            # Load Image
            img = Image.open(img_path)

            # Resize of the original input image is not required for D-DETR
            # as the predictions are rescaled in post_process
            bbox_img, label_strings = trt_infer.draw_bbox(img, pred, pred_masks, inv_classes, cfg.inference.conf_threshold, color_map)
            img_filename = os.path.basename(img_path)
            bbox_img.save(os.path.join(output_annotate_root, img_filename))

            # Store labels
            filename, _ = os.path.splitext(img_filename)
            label_file_name = os.path.join(output_label_root, filename + ".txt")
            with open(label_file_name, "w", encoding="utf-8") as f:
                for l_s in label_strings:
                    f.write(l_s)

    logging.info("Inference results were saved at %s.", cfg.results_dir)


if __name__ == '__main__':
    main()
