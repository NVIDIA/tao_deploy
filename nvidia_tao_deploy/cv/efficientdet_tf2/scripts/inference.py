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
from PIL import Image
import logging
import numpy as np
import tensorrt as trt
from tqdm.auto import tqdm

from nvidia_tao_core.config.efficientdet_tf2.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.efficientdet_tf2.inferencer import EfficientDetInferencer
from nvidia_tao_deploy.cv.efficientdet_tf2.utils import get_label_dict, get_label_map

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.utils.image_batcher import ImageBatcher

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment_spec", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for TRT engine inference."""
    run_inference(cfg=cfg)


@monitor_status(name='efficientdet_tf2', mode='inference')
def run_inference(cfg: ExperimentConfig) -> None:
    """EfficientDet TRT inference."""
    trt_infer = EfficientDetInferencer(cfg.inference.trt_engine, batch_size=cfg.inference.batch_size)

    # Inference may not have labels. Hence, use image batcher
    batcher = ImageBatcher(cfg.inference.image_dir,
                           tuple(trt_infer.input_tensors[0].tensor_shape),
                           trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
                           preprocessor="EfficientDet")

    output_annotate_root = os.path.join(cfg.results_dir, "images_annotated")
    output_label_root = os.path.join(cfg.results_dir, "labels")

    os.makedirs(output_annotate_root, exist_ok=True)
    os.makedirs(output_label_root, exist_ok=True)

    if cfg.inference.label_map and not os.path.exists(cfg.inference.label_map):
        raise FileNotFoundError(f"Class map at {cfg.inference.label_map} does not exist.")

    if str(cfg.inference.label_map).endswith('.yaml'):
        inv_classes = get_label_map(cfg.inference.label_map)
    elif str(cfg.inference.label_map).endswith('.txt'):
        inv_classes = get_label_dict(cfg.inference.label_map)
    else:
        inv_classes = None
        logger.debug("label_map was not provided. Hence, class predictions will not be displayed on the visualization.")

    for batch, img_paths, scales in tqdm(batcher.get_batch(), total=batcher.num_batches, desc="Producing predictions"):
        detections = trt_infer.infer(batch, scales)

        y_pred_valid = np.concatenate([detections['detection_classes'][..., None],
                                      detections['detection_scores'][..., None],
                                      detections['detection_boxes']], axis=-1)

        for img_path, pred in zip(img_paths, y_pred_valid):
            # Load Image
            img = Image.open(img_path)

            # Convert xywh to xyxy
            pred[:, 4:] += pred[:, 2:4]

            bbox_img, label_strings = trt_infer.draw_bbox(img, pred, inv_classes, cfg.inference.min_score_thresh)
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
