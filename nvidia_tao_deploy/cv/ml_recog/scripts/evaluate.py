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

import logging

import os
import json
import numpy as np

import tensorrt as trt
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from nvidia_tao_core.config.ml_recog.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.ml_recog.inferencer import MLRecogInferencer
from nvidia_tao_deploy.cv.ml_recog.dataloader import MLRecogClassificationLoader
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def top_k_accuracy(ground_truth, predicted_labels, k):
    """Calculates the top k accuracy given ground truth labels and predicted labels.

    Args:
        ground_truth (numpy.ndarray): Array of ground truth labels.
        predicted_labels (numpy.ndarray): Array of predicted labels.
        k (int): The value of k for top k accuracy.

    Returns:
        float: The top k accuracy.
    """
    top_k = predicted_labels[:, :k]
    correct = np.sum(np.any(top_k == ground_truth, axis=1))
    return correct / len(predicted_labels)


def average_topk_accuracy_per_class(ground_truth, predicted_labels, k):
    """Calculates the average top k accuracy per class given ground truth labels and predicted labels.

    Args:
        ground_truth (numpy.ndarray): Array of ground truth labels.
        predicted_labels (numpy.ndarray): Array of predicted labels.
        k (int): The value of k for top k accuracy.

    Returns:
        accuracy (numpy.float64): The average top k accuracy per class.
    """
    num_classes = len(np.unique(ground_truth))
    class_acc = []
    for i in range(num_classes):
        class_indices = np.where(ground_truth == i)[0]
        class_pred = predicted_labels[class_indices]
        class_acc.append(top_k_accuracy(i, class_pred, k))
    return np.mean(class_acc)


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name='ml_recog', mode='evaluate')
def main(cfg: ExperimentConfig) -> None:
    """Mwteic Learning Recognition TRT evaluation."""
    classmap = cfg.dataset.class_map

    if classmap:
        # if classmap is provided, we explicitly set the mapping from the text file
        if not os.path.exists(classmap):
            raise FileNotFoundError(f"{classmap} does not exist!")

        with open(classmap, "r", encoding="utf-8") as f:
            mapping_dict = {line.rstrip(): idx for idx, line in enumerate(f.readlines())}
    else:
        # If not, the order of the classes are alphanumeric as defined by Keras
        # Ref: https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/preprocessing/image.py#L507
        mapping_dict = {}
        for idx, subdir in enumerate(sorted(os.listdir(cfg.dataset.val_dataset["reference"]))):
            if os.path.isdir(os.path.join(cfg.dataset.val_dataset["reference"], subdir)):
                mapping_dict[subdir] = idx

    # Load hparams
    target_names = [c[0] for c in sorted(mapping_dict.items(), key=lambda x: x[1])]
    top_k = cfg.evaluate.topk
    image_mean = cfg.dataset.pixel_mean
    img_std = cfg.dataset.pixel_std
    batch_size = cfg.evaluate.batch_size
    input_shape = (batch_size, cfg.model.input_channels, cfg.model.input_height, cfg.model.input_width)

    trt_infer = MLRecogInferencer(cfg.evaluate.trt_engine,
                                  input_shape=input_shape,
                                  batch_size=batch_size)

    gallery_dl = MLRecogClassificationLoader(
        trt_infer.input_tensors[0].shape,
        cfg.dataset.val_dataset["reference"],
        mapping_dict,
        batch_size=batch_size,
        image_mean=image_mean,
        image_std=img_std,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype))

    query_dl = MLRecogClassificationLoader(
        trt_infer.input_tensors[0].shape,
        cfg.dataset.val_dataset["query"],
        mapping_dict,
        batch_size=batch_size,
        image_mean=image_mean,
        image_std=img_std,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype))

    logging.info("Loading gallery dataset...")
    trt_infer.train_knn(gallery_dl, k=top_k)

    gt_labels = []
    pred_labels = []
    for imgs, labels in tqdm(query_dl, total=len(query_dl), desc="Producing predictions"):
        gt_labels.extend(labels)
        _, indices = trt_infer.infer(imgs)
        pred_y = []
        for ind in indices:
            pred_y.append([gallery_dl.labels[i] for i in ind])
        pred_labels.extend(pred_y)

    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)

    # Metric calculation
    target_names = np.array([c[0] for c in sorted(mapping_dict.items(), key=lambda x: x[1])])
    target_labels = np.array([c[1] for c in sorted(mapping_dict.items(), key=lambda x: x[1])])

    # get the average of class accuracies:
    scores = average_topk_accuracy_per_class(gt_labels, pred_labels, k=top_k)
    scores_top1 = average_topk_accuracy_per_class(gt_labels, pred_labels, k=1)
    logging.info("Top %s scores: %s", 1, scores_top1)
    logging.info("Top %s scores: %s", top_k, scores)

    logging.info("Confusion Matrix")
    y_predictions = pred_labels[:, 0]  # get the top 1 prediction
    print(confusion_matrix(gt_labels, y_predictions))
    logging.info("Classification Report")
    print(classification_report(gt_labels, y_predictions, labels=target_labels, target_names=target_names))

    # Store evaluation results into JSON
    eval_results = {"top_k_accuracy": scores,
                    "top_1_accuracy": scores_top1}

    with open(os.path.join(cfg.results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f)
    logging.info("Finished evaluation.")


if __name__ == '__main__':

    main()
