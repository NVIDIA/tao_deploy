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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import os
import json
import numpy as np

from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score

from nvidia_tao_deploy.cv.classification_tf1.inferencer import ClassificationInferencer
from nvidia_tao_deploy.cv.classification_tf1.dataloader import ClassificationLoader

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.classification_pyt.hydra_config.default_config import ExperimentConfig

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name='classification_pyt', mode='evaluation')
def main(cfg: ExperimentConfig) -> None:
    """Classification TRT evaluation."""
    classmap = cfg.dataset.data.test.classes

    if classmap:
        # if classmap is provided, we explicitly set the mapping from the text file
        if not os.path.exists(classmap):
            raise FileNotFoundError(f"{classmap} does not exist!")

        with open(classmap, "r", encoding="utf-8") as f:
            mapping_dict = {line.rstrip(): idx for idx, line in enumerate(sorted(f.readlines()))}
    else:
        # If not, the order of the classes are alphanumeric as defined by Keras
        # Ref: https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/preprocessing/image.py#L507
        mapping_dict = {}
        for idx, subdir in enumerate(sorted(os.listdir(cfg.dataset.data.test.data_prefix))):
            if os.path.isdir(os.path.join(cfg.dataset.data.test.data_prefix, subdir)):
                mapping_dict[subdir] = idx

    # Load hparams
    target_names = [c[0] for c in sorted(mapping_dict.items(), key=lambda x:x[1])]
    top_k = cfg.evaluate.topk
    image_mean = [x / 255 for x in cfg.dataset.img_norm_cfg.mean]
    img_std = [x / 255 for x in cfg.dataset.img_norm_cfg.std]
    batch_size = cfg.evaluate.batch_size

    trt_infer = ClassificationInferencer(cfg.evaluate.trt_engine, data_format="channels_first", batch_size=batch_size)

    dl = ClassificationLoader(
        trt_infer._input_shape,
        [cfg.dataset.data.test.data_prefix],
        mapping_dict,
        data_format="channels_first",
        mode="torch",
        batch_size=cfg.evaluate.batch_size,
        image_mean=image_mean,
        image_std=img_std,
        dtype=trt_infer.inputs[0].host.dtype)

    gt_labels = []
    pred_labels = []
    for imgs, labels in tqdm(dl, total=len(dl), desc="Producing predictions"):
        gt_labels.extend(labels)
        y_pred = trt_infer.infer(imgs)
        pred_labels.extend(y_pred)

    # Check output classes
    output_num_classes = pred_labels[0].shape[0]
    if len(mapping_dict) != output_num_classes:
        raise ValueError(f"Provided class map has {len(mapping_dict)} classes while the engine expects {output_num_classes} classes.")

    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)

    # Metric calculation
    target_names = np.array([c[0] for c in sorted(mapping_dict.items(), key=lambda x:x[1])])
    target_labels = np.array([c[1] for c in sorted(mapping_dict.items(), key=lambda x:x[1])])
    if len(target_labels) == 2:
        # If there are only two classes, sklearn perceive the problem as binary classification
        # and requires predictions to be in (num_samples, ) rather than (num_samples, num_classes)
        scores = top_k_accuracy_score(gt_labels, pred_labels[:, 1], k=top_k, labels=target_labels)
    else:
        scores = top_k_accuracy_score(gt_labels, pred_labels, k=top_k, labels=target_labels)
    logging.info("Top %s scores: %s", top_k, scores)

    logging.info("Confusion Matrix")
    y_predictions = np.argmax(pred_labels, axis=1)
    print(confusion_matrix(gt_labels, y_predictions))
    logging.info("Classification Report")
    print(classification_report(gt_labels, y_predictions, labels=target_labels, target_names=target_names))

    # Store evaluation results into JSON
    eval_results = {"top_k_accuracy": scores}
    if cfg.evaluate.results_dir is not None:
        results_dir = cfg.evaluate.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "trt_evaluate")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f)
    logging.info("Finished evaluation.")


if __name__ == '__main__':

    main()
