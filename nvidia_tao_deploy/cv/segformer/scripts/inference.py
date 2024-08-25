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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from nvidia_tao_deploy.cv.segformer.inferencer import SegformerInferencer
from nvidia_tao_deploy.cv.segformer.dataloader import SegformerLoader
from nvidia_tao_deploy.cv.segformer.utils import imrescale, impad
from nvidia_tao_deploy.cv.segformer.hydra_config.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.unet.proto.utils import TargetClass, get_num_unique_train_ids
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_target_class_list(dataset):
    """Build a list of TargetClasses based on proto.

    Arguments:
        cost_function_config: CostFunctionConfig.
    Returns:
        A list of TargetClass instances.
    """
    target_classes = []
    orig_class_label_id_map = {}
    for target_class in dataset.palette:
        orig_class_label_id_map[target_class.seg_class] = target_class.label_id

    class_label_id_calibrated_map = orig_class_label_id_map.copy()
    for target_class in dataset.palette:
        label_name = target_class.seg_class
        train_name = target_class.mapping_class
        class_label_id_calibrated_map[label_name] = orig_class_label_id_map[train_name]

    train_ids = sorted(list(set(class_label_id_calibrated_map.values())))
    train_id_calibrated_map = {}
    for idx, tr_id in enumerate(train_ids):
        train_id_calibrated_map[tr_id] = idx

    class_train_id_calibrated_map = {}
    for label_name, train_id in class_label_id_calibrated_map.items():
        class_train_id_calibrated_map[label_name] = train_id_calibrated_map[train_id]

    for target_class in dataset.palette:
        target_classes.append(
            TargetClass(target_class.seg_class, label_id=target_class.label_id,
                        train_id=class_train_id_calibrated_map[target_class.seg_class]))

    for target_class in target_classes:
        logging.debug("Label Id %d: Train Id %d", target_class.label_id, target_class.train_id)

    return target_classes


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="infer", schema=ExperimentConfig
)
@monitor_status(name='segformer', mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """Segformer TRT Inference."""
    trt_infer = SegformerInferencer(cfg.inference.trt_engine, batch_size=cfg.dataset.batch_size)
    c, h, w = trt_infer._input_shape

    # Calculate number of classes from the spec file
    target_classes = build_target_class_list(cfg.dataset)
    num_classes = get_num_unique_train_ids(target_classes)

    dl = SegformerLoader(
        shape=(c, h, w),
        image_data_source=[cfg.dataset.test_dataset.img_dir],
        label_data_source=[cfg.dataset.test_dataset.ann_dir],
        num_classes=num_classes,
        dtype=trt_infer.inputs[0].host.dtype,
        batch_size=cfg.dataset.batch_size,
        is_inference=True,
        input_image_type=cfg.dataset.input_type,
        keep_ratio=cfg.dataset.test_dataset.pipeline.augmentation_config.resize.keep_ratio,
        pad_val=cfg.dataset.test_dataset.pipeline.Pad['pad_val'],
        image_mean=cfg.dataset.img_norm_cfg.mean,
        image_std=cfg.dataset.img_norm_cfg.std)

    # Create results directories
    if cfg.inference.results_dir:
        results_dir = cfg.inference.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "trt_inference")

    os.makedirs(results_dir, exist_ok=True)
    vis_dir = os.path.join(results_dir, "vis_overlay")
    os.makedirs(vis_dir, exist_ok=True)
    mask_dir = os.path.join(results_dir, "mask_labels")
    os.makedirs(mask_dir, exist_ok=True)

    # Load classwise rgb value from palette
    id_color_map = {}
    for p in cfg.dataset.palette:
        id_color_map[p['label_id']] = p['rgb']

    for i, (imgs, _) in tqdm(enumerate(dl), total=len(dl), desc="Producing predictions"):
        y_pred = trt_infer.infer(imgs)
        image_paths = dl.image_paths[np.arange(cfg.dataset.batch_size) + cfg.dataset.batch_size * i]

        for img_path, pred in zip(image_paths, y_pred):
            img_file_name = os.path.basename(img_path)

            # Store predictions as mask
            output = Image.fromarray(pred.astype(np.uint8)).convert('P')
            output.save(os.path.join(mask_dir, img_file_name))

            output_palette = np.zeros((num_classes, 3), dtype=np.uint8)
            for c_id, color in id_color_map.items():
                output_palette[c_id] = color

            output.putpalette(output_palette)
            output = output.convert("RGB")

            input_img = Image.open(img_path).convert('RGB')
            orig_width, orig_height = input_img.size
            input_img = np.asarray(input_img)
            input_img = imrescale(input_img, (w, h))
            input_img, padding = impad(input_img, shape=(h, w), pad_val=cfg.dataset.test_dataset.pipeline.Pad['pad_val'])

            if cfg.dataset.input_type == "grayscale":
                output = Image.fromarray(np.asarray(output).astype('uint8'))
            else:
                overlay_img = (np.asarray(input_img) / 2 + np.asarray(output) / 2).astype('uint8')
                output = Image.fromarray(overlay_img)

            # Crop out padded region and resize to original image
            output = output.crop((0, 0, w - padding[2], h - padding[3]))
            output = output.resize((orig_width, orig_height))
            output = Image.fromarray(np.asarray(output).astype('uint8'))
            output.save(os.path.join(vis_dir, img_file_name))

    logging.info("Finished inference.")


if __name__ == '__main__':
    main()
