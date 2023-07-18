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

"""Semantic Segmentation mIoU calculation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
from tqdm.auto import tqdm

import math
import numpy as np

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


def getScoreAverage(scoreList):
    """Compute the average score of all classes."""
    validScores = 0
    scoreSum = 0.0
    for score in scoreList:
        if not math.isnan(score):
            validScores += 1
            scoreSum += score
    if validScores == 0:
        return float('nan')
    avg_score = scoreSum / validScores
    return avg_score


class SemSegMetric(object):
    """Semantic segmentation evaluation metric class."""

    def __init__(self, train_id_name_mapping, label_id_train_id_mapping, num_classes):
        """Constructs SemSeg evaluation class.

        Args:
            train_id_name_mapping (dict): dict of list with key being the label id and value a list of class names.
            num_classes (int): number of classes to evaluate
        """
        self.train_id_name_mapping = train_id_name_mapping
        self.label_id_train_id_mapping = label_id_train_id_mapping
        self.num_classes = num_classes
        assert self.num_classes == len(self.train_id_name_mapping), "Invalid size for class mapping!"

    def get_evaluation_metrics(self, ground_truths, predictions):
        """Generates semantic segmentation metrics.

        Args:
            ground_truths(list): List of ground truths numpy arrays.
            predictions(list): List of prediction numpy arrays.
        """
        metrices = self.compute_metrics_masks(ground_truths, predictions)
        recall_str = "Recall : " + str(metrices["rec"])
        precision_str = "Precision: " + str(metrices["prec"])
        f1_score_str = "F1 score: " + str(metrices["fmes"])
        mean_iou_str = "Mean IOU: " + str(metrices["mean_iou_index"])
        results_str = [recall_str, precision_str, f1_score_str, mean_iou_str]

        metrices_str_categorical = {}
        metrices_str = collections.defaultdict(dict)
        for k, v in metrices["results_dic"].items():
            class_name = str(k)
            for metric_type, val in v.items():
                metrices_str[str(metric_type)][class_name] = str(val)
        metrices_str_categorical["categorical"] = metrices_str

        for result in results_str:
            # This will print the results to the stdout
            print(f"{result}\n")
        return metrices

    def compute_metrics_masks(self, ground_truths, predictions):
        """Compute metrics for semantic segmentation.

        Args:
            ground_truths(list): List of ground truths numpy arrays.
            predictions(list): List of prediction numpy arrays.
        """
        conf_mat = np.zeros([self.num_classes, self.num_classes], dtype=np.float32)

        for pred, gt in tqdm(zip(predictions, ground_truths), desc="Calculating confusion matrix"):
            pred = pred.flatten()
            gt = gt.flatten()
            gt = np.vectorize(self.label_id_train_id_mapping.get)(gt)
            result = np.zeros((self.num_classes, self.num_classes))
            for i in range(len(gt)):
                result[gt[i]][pred[i]] += 1
            conf_mat += np.matrix(result)

        metrices = {}
        perclass_tp = np.diagonal(conf_mat).astype(np.float32)
        perclass_fp = conf_mat.sum(axis=0) - perclass_tp
        perclass_fn = conf_mat.sum(axis=1) - perclass_tp
        iou_per_class = perclass_tp / (perclass_fp + perclass_tp + perclass_fn)
        precision_per_class = perclass_tp / (perclass_fp + perclass_tp)
        recall_per_class = perclass_tp / (perclass_tp + perclass_fn)
        f1_per_class = []
        final_results_dic = {}
        for num_class in range(self.num_classes):
            name_class = "/".join(self.train_id_name_mapping[num_class])
            per_class_metric = {}
            prec = precision_per_class[num_class]
            rec = recall_per_class[num_class]
            iou = iou_per_class[num_class]
            f1 = (2 * prec * rec) / float((prec + rec))
            f1_per_class.append(f1)
            per_class_metric["precision"] = prec
            per_class_metric["Recall"] = rec
            per_class_metric["F1 Score"] = f1
            per_class_metric["iou"] = iou

            final_results_dic[name_class] = per_class_metric

        mean_iou_index = getScoreAverage(iou_per_class)
        mean_rec = getScoreAverage(recall_per_class)
        mean_precision = getScoreAverage(precision_per_class)
        mean_f1_score = getScoreAverage(f1_per_class)

        metrices["rec"] = mean_rec
        metrices["prec"] = mean_precision
        metrices["fmes"] = mean_f1_score
        metrices["mean_iou_index"] = mean_iou_index
        metrices["results_dic"] = final_results_dic

        return metrices
