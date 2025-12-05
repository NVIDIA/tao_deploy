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
import cv2
import tensorrt as trt
from tqdm.auto import tqdm

from nvidia_tao_core.config.segformer.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.segformer.inferencer import SegformerInferencer
from nvidia_tao_deploy.cv.segformer.dataloader import SegformerLoader
from nvidia_tao_deploy.cv.segformer.utils import imrescale
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
    trt_infer = SegformerInferencer(cfg.inference.trt_engine, batch_size=cfg.dataset.segment.batch_size)
    c, h, w = trt_infer.input_tensors[0].shape

    # Calculate number of classes from the spec file
    target_classes = build_target_class_list(cfg.dataset.segment)
    num_classes = get_num_unique_train_ids(target_classes)

    dl = SegformerLoader(
        shape=(c, h, w),
        image_data_source=[os.path.join(cfg.dataset.segment.root_dir, "images", cfg.dataset.segment.predict_split)],
        label_data_source=[os.path.join(cfg.dataset.segment.root_dir, "masks", cfg.dataset.segment.predict_split)],
        num_classes=num_classes,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
        batch_size=cfg.dataset.segment.batch_size,
        is_inference=True,
        keep_ratio=False,
        image_mean=np.array(cfg.dataset.segment.augmentation.mean) * 255,
        image_std=np.array(cfg.dataset.segment.augmentation.std) * 255)

    # Create results directories
    vis_dir = os.path.join(cfg.results_dir, "vis_overlay")
    os.makedirs(vis_dir, exist_ok=True)
    mask_dir = os.path.join(cfg.results_dir, "mask_labels")
    os.makedirs(mask_dir, exist_ok=True)

    # Load classwise rgb value from palette
    id_color_map = {}
    for p in cfg.dataset.segment.palette:
        id_color_map[p['label_id']] = p['rgb']

    for i, (imgs, _) in tqdm(enumerate(dl), total=len(dl), desc="Producing predictions"):
        y_pred = trt_infer.infer(imgs)
        image_paths = dl.image_paths[np.arange(cfg.dataset.segment.batch_size) + cfg.dataset.segment.batch_size * i]

        for img_path, pred in zip(image_paths, y_pred):
            img_file_name = os.path.basename(img_path)

            # Store predictions as mask
            pred = np.argmax(pred, axis=0).astype(np.uint8)
            # resize to original image size
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            output = Image.fromarray(pred.astype(np.uint8))
            output.save(os.path.join(mask_dir, img_file_name))

            color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for c_id, color in id_color_map.items():
                color_mask[pred == c_id] = color

            input_img = Image.open(img_path).convert('RGB')
            orig_width, orig_height = input_img.size
            input_img = np.asarray(input_img)
            input_img = imrescale(input_img, (w, h))

            overlay_img = ((input_img.astype(np.float32) * 0.5) + (color_mask.astype(np.float32) * 0.5)).astype(np.uint8)
            overlay_img = Image.fromarray(overlay_img)

            overlay_img = overlay_img.resize((orig_width, orig_height))
            overlay_img = Image.fromarray(np.asarray(overlay_img).astype('uint8'))
            overlay_img.save(os.path.join(vis_dir, img_file_name))

    logging.info("Finished inference.")


if __name__ == '__main__':
    main()
