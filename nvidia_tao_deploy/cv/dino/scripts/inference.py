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

"""Standalone TensorRT inference."""

import os
import logging
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.deformable_detr.inferencer import DDETRInferencer
from nvidia_tao_deploy.cv.deformable_detr.utils import post_process
from nvidia_tao_deploy.cv.dino.hydra_config.default_config import ExperimentConfig
from nvidia_tao_deploy.utils.image_batcher import ImageBatcher

from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="infer", schema=ExperimentConfig
)
@monitor_status(name='dino', mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """DINO TRT Inference."""
    if not os.path.exists(cfg.inference.trt_engine):
        raise FileNotFoundError(f"Provided inference.trt_engine at {cfg.inference.trt_engine} does not exist!")

    trt_infer = DDETRInferencer(cfg.inference.trt_engine,
                                batch_size=cfg.dataset.batch_size,
                                num_classes=cfg.dataset.num_classes)

    c, h, w = trt_infer._input_shape
    batcher = ImageBatcher(list(cfg.dataset.infer_data_sources.image_dir),
                           (cfg.dataset.batch_size, c, h, w),
                           trt_infer.inputs[0].host.dtype,
                           preprocessor="DDETR")

    with open(cfg.dataset.infer_data_sources.classmap, "r", encoding="utf-8") as f:
        classmap = [line.rstrip() for line in f.readlines()]
    classes = {c: i + 1 for i, c in enumerate(classmap)}

    # Create results directories
    if cfg.inference.results_dir is not None:
        results_dir = cfg.inference.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "trt_inference")
    os.makedirs(results_dir, exist_ok=True)
    output_annotate_root = os.path.join(results_dir, "images_annotated")
    output_label_root = os.path.join(results_dir, "labels")

    os.makedirs(output_annotate_root, exist_ok=True)
    os.makedirs(output_label_root, exist_ok=True)

    inv_classes = {v: k for k, v in classes.items()}

    for batches, img_paths, scales in tqdm(batcher.get_batch(), total=batcher.num_batches, desc="Producing predictions"):
        # Handle last batch as we artifically pad images for the last batch idx
        if len(img_paths) != len(batches):
            batches = batches[:len(img_paths)]
        pred_logits, pred_boxes = trt_infer.infer(batches)
        target_sizes = []
        for batch, scale in zip(batches, scales):
            _, new_h, new_w = batch.shape
            orig_h, orig_w = int(scale[0] * new_h), int(scale[1] * new_w)
            target_sizes.append([orig_w, orig_h, orig_w, orig_h])

        class_labels, scores, boxes = post_process(pred_logits, pred_boxes, target_sizes)
        y_pred_valid = np.concatenate([class_labels[..., None], scores[..., None], boxes], axis=-1)
        for img_path, pred in zip(img_paths, y_pred_valid):
            # Load Image
            img = Image.open(img_path)

            # Resize of the original input image is not required for D-DETR
            # as the predictions are rescaled in post_process
            bbox_img, label_strings = trt_infer.draw_bbox(img, pred, inv_classes, cfg.inference.conf_threshold, cfg.inference.color_map)
            img_filename = os.path.basename(img_path)
            bbox_img.save(os.path.join(output_annotate_root, img_filename))

            # Store labels
            filename, _ = os.path.splitext(img_filename)
            label_file_name = os.path.join(output_label_root, filename + ".txt")
            with open(label_file_name, "w", encoding="utf-8") as f:
                for l_s in label_strings:
                    f.write(l_s)

    logging.info("Finished inference.")


if __name__ == '__main__':
    main()
