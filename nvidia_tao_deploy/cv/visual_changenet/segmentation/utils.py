# Copyright (c) 2023 Chaminda Bandara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Original source taken from https://github.com/wgcban/ChangeFormer
#
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

"""Visual ChangeNet utils."""

import os
import numpy as np
import matplotlib.pyplot as plt

# LandSCD mapping for external use."""
colour_mappings_landSCD = {
    '0': (255, 255, 255),
    '1': (229, 156, 22),
    '2': (196, 67, 84),
    '3': (50, 153, 50),
    '4': (229, 115, 213),
    '5': (255, 0, 255),
    '6': (114, 229, 190),
    '7': (209, 69, 17),
    '8': (186, 218, 85),
    '9': (132, 17, 209),
}


def get_color_mapping(dataset_name, num_classes=None, color_mapping_custom=None):
    """Get the color mapping for semantic segmentation visualization for num_classes>2.

    For binary segmentation, black and white color coding is used.
    Args:
        dataset_name (str): The name of the dataset ('LandSCD' or 'custom').
        num_classes (int, optional): The number of classes in the dataset (default is None).
        color_mapping_custom (dict, optional): Custom color mapping provided as a dictionary with class indices as keys and RGB tuples as values (default is None).

    Returns:
        dict: A dictionary containing the color mapping for each class, where class indices are the keys, and RGB tuples are the values.
    """
    output_color_mapping = None
    if color_mapping_custom is not None:
        assert num_classes == len(color_mapping_custom.keys()), f"""Number of color mappings ({len(color_mapping_custom.keys())}) provided must match number of classes ({num_classes})"""
        output_color_mapping = color_mapping_custom
    elif dataset_name == 'LandSCD':
        output_color_mapping = colour_mappings_landSCD
    else:
        # If color mapping not provided, randomly generate color mapping
        color_mapping_custom = {}
        for i in range(num_classes):
            color_mapping_custom[str(i)] = tuple(np.random.randint(1, 254, size=(3)))  # Reserve (0,0,0) for mismatch pred
        color_mapping_custom[str('0')] = (255, 255, 255)  # Enforce no-change class to be white
        output_color_mapping = color_mapping_custom
    return output_color_mapping


def make_numpy_grid(vis, num_class=2, color_map=None):
    """Convert a batch of numpy tensors into a numpy grid for visualization using color map

    Args:
        vis (np.array): The input batch of numpy array images to be visualised.
        num_class (int, optional): The number of classes (default is 2).
        color_map (dict, optional): Custom color mapping provided as a dictionary with class indices as keys and RGB tuples as values (default is None).
            For binary segmentation, black and white color coding is used.

    Returns:
        np.ndarray: The numpy grid for visualization.
    """
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
        vis = np.squeeze(vis)
    if num_class > 2:
        vis_multi = vis[:, :, 0]
        color_coded = np.ones(np.shape(vis))

        # Can take custom color map/randomly generate color map for custom datasets
        assert color_map is not None, 'Provide a color map for output visualization'
        for i in range(num_class):
            color_coded[vis_multi == i] = color_map[str(i)]
        color_coded = color_coded / 255
        color_coded = color_coded.astype(float)
        return color_coded
    return vis


def de_norm(tensor_data):
    """Perform de-normalization on a tensor by reversing the normalization process.

    Args:
        tensor_data (torch.Tensor): The normalized tensor data to be de-normalized.

    Returns:
        torch.Tensor: The de-normalized tensor data.
    """
    # TODO: @zbhat check if this needs to change if mean/std augmentation made configurable
    return tensor_data * 0.5 + 0.5


def visualize_pred(output):
    """Helper function to visualize predictions for binary change detection (LEVIR-CD)"""
    pred = np.argmax(output, axis=2, keepdims=True)
    pred = pred * 255

    return pred


def visualize_pred_multi(output):
    """Helper function to visualize predictions for multi class change detection (landSCD)"""
    pred = np.argmax(output, axis=2, keepdims=True)
    return pred


def visualize_infer_output(name, output, img1, img2, n_class, color_map, vis_dir, gt=None, mode='test'):
    """Visualizes two input images along with segmentation change map prediction as a linear grid of images.

    Does not include GT segmentation change map during inference.
    """
    vis_input = de_norm(img1)
    vis_input2 = de_norm(img2)
    output = np.transpose(output, (1, 2, 0))

    if n_class > 2:
        # print("Visualising multiple classes")
        # TODO: Verify this and verify for bs>1
        if mode == 'test':
            vis_pred = make_numpy_grid(visualize_pred_multi(output), num_class=n_class, color_map=color_map)
            vis_gt = make_numpy_grid(gt, num_class=n_class, color_map=color_map)
        else:
            vis_pred = make_numpy_grid(visualize_pred_multi(output), num_class=n_class, color_map=color_map)

    else:
        vis_pred = make_numpy_grid(visualize_pred(output))
        vis_pred = np.clip(vis_pred, a_min=0.0, a_max=1.0)
        if mode == 'test':
            vis_gt = make_numpy_grid(gt)
    vis_pred = np.transpose(vis_pred, (2, 0, 1))

    # Combining horizontally
    line_width = 10  # width of the black line in pixels
    line = np.zeros((vis_input.shape[1], line_width, 3), dtype=np.uint8)  # create a black line
    line = np.transpose(line, (2, 0, 1))

    if mode == 'test':
        vis_gt = np.transpose(vis_gt, (2, 0, 1))
        if n_class > 2:
            vis = np.concatenate([vis_input, line, vis_input2, line, vis_pred, line, vis_gt], axis=2)
        else:
            vis = np.concatenate([vis_input, line, vis_input2, line, vis_pred, line, vis_gt], axis=2)
    else:
        if n_class > 2:
            vis = np.concatenate([vis_input, line, vis_input2, line, vis_pred], axis=2)
        else:
            vis = np.concatenate([vis_input, line, vis_input2, line, vis_pred], axis=2)

    vis = np.clip(vis, a_min=0.0, a_max=1.0)
    file_name = os.path.join(
        vis_dir, str(name) + '.jpg')
    vis = np.transpose(vis, (1, 2, 0))
    plt.imsave(file_name, vis)


# Metrics
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        """Initialize AverageMeter"""
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        """Initialize counter variables for metric calculation.

        Args:
            val (float): Initial value.
            weight (int): Initial weight.
        """
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        """Update AverageMeter with new value and weight.

        Args:
            val (float): New value to update with.
            weight (int, optional): Weight for the new value (default is 1).
        """
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        """Update AverageMeter with new value and weight.

        Args:
            val (float): New value to update with.
            weight (int, optional): Weight for the new value (default is 1).
        """
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        """Get the current value."""
        return self.val

    def average(self):
        """Get the average value."""
        return self.avg

    def get_scores(self):
        """Get scores and mean score using cm2score function."""
        scores_dict, mean_score_dict = cm2score(self.sum)
        return scores_dict, mean_score_dict

    def clear(self):
        """Clear the initialized status."""
        self.initialized = False


class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, n_class):
        """init"""
        super().__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        """Get the current confusion matrix, calculate the current F1 score, and update the confusion matrix"""
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        """get scores from confusion matrix"""
        scores_dict, mean_score_dict = cm2score(self.sum)
        return scores_dict, mean_score_dict


def harmonic_mean(xs):
    """Compute Harmonic mean"""
    harmonic_mean = len(xs) / sum((x + 1e-6)**-1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    """Compute F1 Score"""
    hist = confusion_matrix
    # n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # Acc = total acc, recall, precision, F1 are per class, mean_f1 is average F1 over all classes
    return mean_F1


def cm2score(confusion_matrix):
    """Compute Scores from Confusion Matrix"""
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    cls_iou = dict(zip(['iou_' + str(i) for i in range(n_class)], iu))

    cls_precision = dict(zip(['precision_' + str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_' + str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_' + str(i) for i in range(n_class)], F1))

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1': mean_F1}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)

    # Add mean metrics
    mean_recall = np.nanmean(recall)
    mean_precision = np.nanmean(precision)
    mean_score_dict = {'mprecision': mean_precision, 'mrecall': mean_recall}  # 'macc_': acc, 'miou_':mean_iu, 'mf1_':mean_F1,

    return score_dict, mean_score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """Compute the confusion matrix for a set of predictions"""

    def __fast_hist(label_gt, label_pred):
        """Collect values for Confusion Matrix

        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix

        Args:
            label_gt: <np.array> ground-truth
            label_pred: <np.array> prediction

        Returns:
            np.ndarray: The numpy grid for visualization.
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)

        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    """Get mIoU"""
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']  # pylint: disable=E1126
