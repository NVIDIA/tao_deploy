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
import cv2
import json
import logging
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.mask2former.inferencer import Mask2formerInferencer
from nvidia_tao_deploy.cv.mask2former.hydra_config.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.mask2former.d2.structures import Instances
from nvidia_tao_deploy.cv.mask2former.d2.visualizer import ColorMode, Visualizer
from nvidia_tao_deploy.cv.mask2former.d2.catalog import MetadataCatalog
from nvidia_tao_deploy.utils.image_batcher import ImageBatcher

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_metadata(cfg):
    """Prepare metadata for the dataset."""
    label_map = cfg.dataset.label_map
    with open(label_map, 'r', encoding='utf-8') as f:
        categories = json.load(f)

    if not cfg.dataset.contiguous_id:
        categories_full = [{'name': "nan", 'color': [0, 0, 0], 'isthing': 1, 'id': i + 1} for i in range(cfg.model.sem_seg_head.num_classes)]
        for cat in categories:
            categories_full[cat['id'] - 1] = cat
        categories = categories_full

    meta = {}
    thing_classes = [k["name"] for k in categories if k.get("isthing", 1)]
    thing_colors = [k.get("color", np.random.randint(0, 255, size=3).tolist()) for k in categories if k.get("isthing", 1)]
    stuff_classes = [k["name"] for k in categories]
    stuff_colors = [k.get("color", np.random.randint(0, 255, size=3).tolist()) for k in categories]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    if cfg.dataset.contiguous_id:
        thing_dataset_id_to_contiguous_id = {}
        stuff_dataset_id_to_contiguous_id = {}

        for i, cat in enumerate(categories):
            if cat.get("isthing", 1):
                thing_dataset_id_to_contiguous_id[cat["id"]] = i
            # in order to use sem_seg evaluator
            stuff_dataset_id_to_contiguous_id[cat["id"]] = i
    else:
        thing_dataset_id_to_contiguous_id = {j: j for j in range(len(categories))}
        stuff_dataset_id_to_contiguous_id = {j: j for j in range(len(categories))}
    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    return meta


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="infer", schema=ExperimentConfig
)
@monitor_status(name='mask2former', mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """Mask2former TRT Inference."""
    if not os.path.exists(cfg.inference.trt_engine):
        raise FileNotFoundError(f"Provided inference.trt_engine at {cfg.inference.trt_engine} does not exist!")

    metadata = get_metadata(cfg)
    MetadataCatalog.get("custom").set(
        thing_classes=metadata["thing_classes"],
        thing_colors=metadata["thing_colors"],
        stuff_classes=metadata["stuff_classes"],
        stuff_colors=metadata["stuff_colors"],
        thing_dataset_id_to_contiguous_id=metadata["thing_dataset_id_to_contiguous_id"],
        stuff_dataset_id_to_contiguous_id=metadata["stuff_dataset_id_to_contiguous_id"],
    )
    trt_infer = Mask2formerInferencer(
        cfg.inference.trt_engine,
        batch_size=cfg.dataset.test.batch_size,
        is_inference=True)

    _, hh, ww = trt_infer._input_shape

    # Create results directories
    if cfg.inference.results_dir:
        results_dir = cfg.inference.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "trt_inference")

    os.makedirs(results_dir, exist_ok=True)

    # Inference may not have labels. Hence, use image batcher
    batch_size = trt_infer.max_batch_size
    batcher = ImageBatcher(cfg.dataset.test.img_dir,
                           (batch_size,) + trt_infer._input_shape,
                           trt_infer.inputs[0].host.dtype,
                           preprocessor="Mask2former",
                           img_mean=cfg.dataset.pixel_mean,
                           img_std=cfg.dataset.pixel_std)

    for batches, img_paths, _ in tqdm(batcher.get_batch(), total=batcher.num_batches, desc="Producing predictions"):
        # Handle last batch as we artifically pad images for the last batch idx
        if len(img_paths) != len(batches):
            batches = batches[:len(img_paths)]
        predictions = trt_infer.infer(batches)
        if len(predictions) > 1:
            predictions = zip(*trt_infer.infer(batches))
        for i, prediction in enumerate(predictions):

            curr_path = img_paths[i]
            raw_img = load_image(curr_path, target_size=(ww, hh))
            visualizer = Visualizer(
                raw_img,
                MetadataCatalog.get("custom"),
                instance_mode=ColorMode.IMAGE)
            if len(prediction) == 4:
                pred_prob_mask, pred_mask, pred_score, pred_label = prediction
                keep = pred_score >= cfg.model.object_mask_threshold
                cur_prob_masks = pred_prob_mask[keep]
                cur_masks = pred_mask[keep]
                cur_classes = pred_label[keep]
                h, w = pred_prob_mask.shape[-2:]
                panoptic_seg = np.zeros((h, w))
                segments_info = []

                current_segment_id = 0
                if pred_prob_mask.shape[0] != 0:
                    # take argmax
                    cur_mask_ids = cur_prob_masks.argmax(0)
                    stuff_memory_list = {}
                    for k in range(cur_classes.shape[0]):
                        pred_class = cur_classes[k].item()
                        isthing = pred_class in metadata['thing_dataset_id_to_contiguous_id'].values()
                        mask_area = (cur_mask_ids == k).sum().item()
                        original_area = (cur_masks[k] >= 0.5).sum().item()
                        mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                        if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                            if mask_area / original_area < cfg.model.overlap_threshold:
                                continue

                            # merge stuff regions
                            if not isthing:
                                if int(pred_class) in stuff_memory_list.keys():
                                    panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                    continue
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                            current_segment_id += 1
                            panoptic_seg[mask] = current_segment_id

                            segments_info.append(
                                {
                                    "id": current_segment_id,
                                    "isthing": bool(isthing),
                                    "category_id": int(pred_class),  # original Mask2former
                                }
                            )
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg, segments_info
                )
            elif len(prediction) == 3:
                result = Instances([hh, ww])
                pred_cls, pred_mask, pred_score = prediction
                result.pred_masks = pred_mask
                result.scores = pred_score
                result.pred_classes = pred_cls
                vis_output = visualizer.draw_instance_predictions(
                    predictions=result,
                    mask_threshold=cfg.model.object_mask_threshold
                )
            else:
                mask_img = np.argmax(prediction, axis=0)
                vis_output = visualizer.draw_sem_seg(
                    mask_img,
                )
            cv2.imwrite(
                os.path.join(results_dir, os.path.basename(curr_path)[:-4] + ".jpg"),
                vis_output.get_image()
            )

    logging.info("Inference results were saved at %s.", results_dir)


def load_image(file_name, root_dir=None, target_size=None):
    """Load image.

    Args:
        file_name (str): relative path to an image file (.png).
    Return:
        image (PIL image): loaded image
    """
    root_dir = root_dir or ""
    image = Image.open(os.path.join(root_dir, file_name)).convert('RGB')
    if target_size:
        image = image.resize(target_size)
    image = np.array(image)
    return image


if __name__ == '__main__':
    main()
