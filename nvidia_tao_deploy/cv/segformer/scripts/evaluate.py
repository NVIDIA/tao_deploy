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

"""Standalone TensorRT evaluation."""

import os
import logging
import json
import tensorrt as trt
from tqdm.auto import tqdm
from collections import defaultdict
import cv2
import numpy as np

from nvidia_tao_core.config.segformer.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.segformer.inferencer import SegformerInferencer
from nvidia_tao_deploy.cv.segformer.dataloader import SegformerLoader
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.unet.proto.utils import TargetClass, get_num_unique_train_ids
from nvidia_tao_deploy.metrics.semantic_segmentation_metric import SemSegMetric
from nvidia_tao_deploy.cv.common.decorators import monitor_status


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_label_train_dic(target_classes):
    """Function to get mapping between class and train ids."""
    label_train_dic = {}
    for target in target_classes:
        label_train_dic[target.label_id] = target.train_id

    return label_train_dic


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
@monitor_status(name='segformer', mode='evaluate')
def main(cfg: ExperimentConfig) -> None:
    """Segformer TRT Evaluation."""
    trt_infer = SegformerInferencer(cfg.evaluate.trt_engine, batch_size=cfg.dataset.segment.batch_size)
    c, h, w = trt_infer.input_tensors[0].shape

    # Calculate number of classes from the spec file
    target_classes = build_target_class_list(cfg.dataset.segment)
    label_id_train_id_mapping = get_label_train_dic(target_classes)
    num_classes = get_num_unique_train_ids(target_classes)

    dl = SegformerLoader(
        shape=(c, h, w),
        image_data_source=[os.path.join(cfg.dataset.segment.root_dir, "images", cfg.dataset.segment.validation_split)],
        label_data_source=[os.path.join(cfg.dataset.segment.root_dir, "masks", cfg.dataset.segment.validation_split)],
        num_classes=num_classes,
        target_classes=target_classes,
        label_transform=cfg.dataset.segment.label_transform,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
        batch_size=cfg.dataset.segment.batch_size,
        is_inference=False,
        keep_ratio=False,
        image_mean=np.array(cfg.dataset.segment.augmentation.mean) * 255,
        image_std=np.array(cfg.dataset.segment.augmentation.std) * 255)

    # Load label mapping
    label_mapping = defaultdict(list)
    for p in cfg.dataset.segment.palette:
        label_mapping[p['label_id']].append(p['seg_class'])

    eval_metric = SemSegMetric(num_classes=num_classes,
                               train_id_name_mapping=label_mapping,
                               label_id_train_id_mapping=label_id_train_id_mapping)

    gt_labels = []
    pred_labels = []
    for imgs, labels in tqdm(dl, total=len(dl), desc="Producing predictions"):
        gt_labels.extend(labels)
        y_pred = trt_infer.infer(imgs)
        # get the argmax of the output
        y_pred = y_pred.argmax(axis=1)
        # opencv resize to labels shape
        y_pred = [cv2.resize(np.uint8(y), (labels[0].shape[1], labels[0].shape[0]), interpolation=cv2.INTER_NEAREST) for y in y_pred]
        pred_labels.extend(y_pred)

    metrices = eval_metric.get_evaluation_metrics(gt_labels, pred_labels)

    with open(os.path.join(cfg.results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(str(metrices["results_dic"]), f)
    logging.info("Finished evaluation.")


if __name__ == '__main__':
    main()
