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

"""mAP calculation."""

from functools import partial
import logging
import os
import numpy as np
# Suppress logging from matplotlib
from matplotlib import pyplot as plt  # noqa: E402

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


def batch_iou(box, box_list):
    """element-wise IOU to perform on a batch (box_list).

    Args:
        box: np array of shape (4,): the target box
        box_list: np array of shape (N, 4): a batch of boxes to match the box.

    Returns:
        np array of shape (N,). The IOU between target box and each single box in box_list
    """
    if box.ndim == 1:
        box = np.expand_dims(box, axis=0)
    if box_list.ndim == 1:
        box_list = np.expand_dims(box_list, axis=0)

    # Compute the IoU.

    min_xy = np.maximum(box[:, :2], box_list[:, :2])
    max_xy = np.minimum(box[:, 2:], box_list[:, 2:])

    interx = np.maximum(0, max_xy - min_xy)

    interx = interx[:, 0] * interx[:, 1]

    box_area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    box_list_areas = (box_list[:, 2] - box_list[:, 0]) * (box_list[:, 3] - box_list[:, 1])

    union_areas = box_area + box_list_areas - interx

    return interx / union_areas


def _per_img_match(x, n_classes, sorting_algorithm, matching_iou_threshold):
    """Helper function for multithreading matching.

    Do not call this function from outside. It's outside the class definition purely due to python
    pickle issue.

    Arguments:
        x (tuple): (gt_box, pred_box)
        n_classes (int): number of classes
        sorting_algorithm (str): Which sorting algorithm the matching algorithm should
            use. This argument accepts any valid sorting algorithm for Numpy's `argsort()`
            function. You will usually want to choose between 'quicksort' (fastest and most
            memory efficient, but not stable) and 'mergesort' (slight slower and less memory
            efficient, but stable). The official Matlab evaluation algorithm uses a stable
            sorting algorithm, so this algorithm is only guaranteed to behave identically if you
            choose 'mergesort' as the sorting algorithm, but it will almost always behave
            identically even if you choose 'quicksort' (but no guarantees).
        matching_iou_threshold (float): A prediction will be considered a true
            positive if it has a Jaccard overlap of at least `matching_iou_threshold` with any
            ground truth bounding box of the same class.
    """
    gt = x[0]
    pred = x[1]
    T = [[] for _ in range(n_classes)]
    P = [[] for _ in range(n_classes)]
    gt_cls = [gt[gt[:, 0].astype(int) == i, 1:] for i in range(n_classes)]
    gt_cls_valid = [np.ones((len(i), )) for i in gt_cls]
    gt_hard_count = [i[:, 0].sum() for i in gt_cls]

    desc_inds = np.argsort(-pred[:, 1], kind=sorting_algorithm)
    pred = pred[desc_inds]
    for pred_box in pred:
        pred_cls = int(pred_box[0])

        # if no GT in this class, simply recognize as FP
        if len(gt_cls[pred_cls]) == 0:
            T[pred_cls].append(0)
            P[pred_cls].append(pred_box[1])
            continue

        overlaps = batch_iou(box_list=gt_cls[pred_cls][:, -4:], box=pred_box[-4:])
        overlaps_unmatched = overlaps * gt_cls_valid[pred_cls]

        if np.max(overlaps_unmatched) >= matching_iou_threshold:
            # invalidate the matched gt
            matched_gt_idx = np.argmax(overlaps_unmatched)
            gt_cls_valid[pred_cls][matched_gt_idx] = 0.0
            if gt_cls[pred_cls][matched_gt_idx, 0] < 0.5:
                # this is not a hard box. We should append GT
                T[pred_cls].append(1)
                P[pred_cls].append(pred_box[1])
            else:
                logger.warning("Got label marked as difficult(occlusion > 0), "
                               "please set occlusion field in KITTI label to 0, "
                               "if you want to include it in mAP calculation "
                               "during validation/evaluation.")
                # this hard box is already processed. Deduct from gt_hard_cnt
                gt_hard_count[pred_cls] = gt_hard_count[pred_cls] - 1

        else:
            T[pred_cls].append(0)
            P[pred_cls].append(pred_box[1])

    for idx, cls_valid in enumerate(gt_cls_valid):
        non_match_count = int(round(cls_valid.sum() - gt_hard_count[idx]))
        T[idx].extend([1] * non_match_count)
        P[idx].extend([0.0] * non_match_count)

    return (T, P)


class KITTIMetric:
    """Computes the mean average precision of the given lists of pred and GT."""

    def __init__(self,
                 n_classes,
                 conf_thres=0.01,
                 matching_iou_threshold=0.5,
                 average_precision_mode='sample',
                 num_recall_points=11):
        """Initializes Keras / TensorRT objects needed for model inference.

        Args:
            n_classes (integer): Number of classes
            conf_thres (float): confidence threshold to consider a bbox.
            matching_iou_threshold (float, optional): A prediction will be considered a true
                positive if it has a Jaccard overlap of at least `matching_iou_threshold` with any
                ground truth bounding box of the same class.
            average_precision_mode (str, optional): Can be either 'sample' or 'integrate'. In the
                case of 'sample', the average precision will be computed according to the Pascal VOC
                formula that was used up until VOC 2009, where the precision will be sampled for
                `num_recall_points` recall values. In the case of 'integrate', the average precision
                will be computed according to the Pascal VOC formula that was used from VOC 2010
                onward, where the average precision will be computed by numerically integrating
                over the whole preciscion-recall curve instead of sampling individual points from
                it. 'integrate' mode is basically just the limit case of 'sample' mode as the number
                of sample points increases.
            num_recall_points (int, optional): The number of points to sample from the
                precision-recall-curve to compute the average precisions. In other words, this is
                the number of equidistant recall values for which the resulting precision will be
                computed. 11 points is the value used in the official Pascal VOC 2007 detection
                evaluation algorithm.
        """
        self.n_classes = n_classes
        self.conf_thres = conf_thres
        self.matching_iou_threshold = matching_iou_threshold
        self.average_precision_mode = average_precision_mode
        self.num_recall_points = num_recall_points

        self.gt_labels = None
        self.pred_labels = None
        self.T = None
        self.P = None
        self.ap = None

    def __call__(self, gt, pred, verbose=True, class_names=None, vis_path=None):
        """Compute AP of each classes and mAP.

        Arguments:
            gt (list of numpy arrays): A list of length n_eval_images. Each element is a numpy
                array of shape (n_bbox, 6). n_bbox is the number of boxes inside the image and
                6 elements for the bbox is [class_id, is_difficult, xmin, ymin, xmax, ymax].
                Note: is_difficult is 0 if the bbox is not difficult. 1 otherwise. Always set
                is_difficult to 0 if you don't have this field in your GT label.
            pred (list of numpy arrays): A list of length n_eval_images. Each element is a numpy
                array of shape (n_bbox, 6). n_bbox is the number of boxes inside the image and
                6 elements for the bbox is [class_id, confidence, xmin, ymin, xmax, ymax]
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            class_name(list): Name of object classes for vis.
            vis_path(string): Path to save vis image.

        Note: the class itself supports both normalized / un-normalized coords. As long as the
        coords is_normalized for gt and pred identical, the class gives correct results.

        Returns:
            A float, the mean average precision. A list of length n_classes. AP for each class
        """
        self.gt_labels = gt
        self.pred_labels = pred

        self.matching(sorting_algorithm='quicksort',
                      matching_iou_threshold=self.matching_iou_threshold,
                      verbose=verbose)

        if verbose:
            print('Start to calculate AP for each class')
        # Calc AP and plot PR curves
        self._calc_ap(sorting_algorithm='quicksort',
                      average_precision_mode=self.average_precision_mode,
                      num_recall_points=self.num_recall_points,
                      class_names=class_names,
                      vis_path=vis_path)
        # Save plots to image
        if vis_path is not None:
            plt.legend()
            plt.title("Precision-Recall curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.grid()
            save_path = os.path.join(vis_path, "PR_curve.png")
            plt.savefig(save_path)
            print(f"PR-curve image saved to {save_path}")
            plt.clf()
        # release memory
        self.gt_labels = None
        self.pred_labels = None

        return np.mean(self.ap), self.ap

    def matching(self, sorting_algorithm, matching_iou_threshold, verbose):
        """Generate T, P list for AP calculation.

        Arguments:
            T: 0 - negative match, 1 - positive match
            P: confidence of this prediction
        """
        if (self.gt_labels is None) or (self.pred_labels is None):
            raise ValueError("Matching cannot be called before the completion of prediction!")

        if len(self.gt_labels) != len(self.pred_labels):
            raise ValueError("Image count mismatch between ground truth and prediction!")

        T = [[] for _ in range(self.n_classes)]
        P = [[] for _ in range(self.n_classes)]

        per_img_match = partial(_per_img_match, n_classes=self.n_classes,
                                sorting_algorithm=sorting_algorithm,
                                matching_iou_threshold=matching_iou_threshold)

        results = []
        for x in zip(self.gt_labels, self.pred_labels):
            results.append(per_img_match(x))

        for t, p in results:
            for i in range(self.n_classes):
                T[i] += t[i]
                P[i] += p[i]

        self.T = T
        self.P = P

    def __voc_ap(
        self,
        rec,
        prec,
        average_precision_mode,
        num_recall_points,
        class_name=None,
        vis_path=None
    ):
        if average_precision_mode == 'sample':
            ap = 0.
            for t in np.linspace(0., 1.0, num_recall_points):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / float(num_recall_points)
            if class_name and vis_path:
                rec_arr = np.array(rec)
                prec_arr = np.array(prec)
                plt.plot(rec_arr, prec_arr, label=class_name)
        elif average_precision_mode == 'integrate':
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))
            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]
            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            if class_name and vis_path:
                if class_name != "bg":
                    plt.plot(mrec, mpre, label=class_name)
        else:
            raise ValueError("average_precision_mode should be either sample or integrate")
        return ap

    def _calc_ap(
        self,
        sorting_algorithm,
        average_precision_mode,
        num_recall_points,
        class_names=None,
        vis_path=None
    ):
        """compute the AP for classes."""
        if (self.T is None) or (self.P is None):
            raise ValueError("Matching must be done first!")

        self.ap = []
        class_idx = 0
        for T, P in zip(self.T, self.P):
            if class_names is not None:
                class_name = class_names[class_idx]
            else:
                class_name = None
            prec = []
            rec = []
            TP = 0.
            FP = 0.
            FN = 0.
            # sort according to prob.
            Ta = np.array(T)
            Pa = np.array(P)
            s_idx = np.argsort(-Pa, kind=sorting_algorithm)
            P = Pa[s_idx].tolist()
            T = Ta[s_idx].tolist()
            npos = np.sum(Ta)
            for t, p in zip(T, P):
                if t == 1 and p >= self.conf_thres:
                    TP += 1
                elif t == 1 and p < self.conf_thres:
                    FN += 1
                elif t == 0 and p >= self.conf_thres:
                    FP += 1
                if TP + FP == 0.:
                    precision = 0.
                else:
                    precision = float(TP) / (TP + FP)
                if npos > 0:
                    recall = float(TP) / float(npos)
                else:
                    recall = 0.0
                prec.append(precision)
                rec.append(recall)
            ap = self.__voc_ap(
                np.array(rec),
                np.array(prec),
                average_precision_mode,
                num_recall_points,
                class_name,
                vis_path
            )
            self.ap.append(ap)
            class_idx += 1
