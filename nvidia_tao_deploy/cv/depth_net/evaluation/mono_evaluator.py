# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""DepthNet evaluation metrics and utilities.

This module provides comprehensive evaluation capabilities for depth estimation models,
including various accuracy metrics and depth alignment utilities. It supports both
monocular and stereo depth estimation evaluation with industry-standard metrics.
"""

from typing import Tuple, List, Dict

import numpy as np
from numpy import ndarray


def align_depth_least_square(
    gt: ndarray,
    pred: ndarray,
):
    """
    Align predicted depth to ground truth using least squares optimization.

    This function computes a linear transformation (scale and shift) to align
    predicted depth values with ground truth depth values. The transformation
    minimizes the least squares error between the aligned prediction and ground truth.

    Args:
        gt (numpy.ndarray): Ground truth depth/disparity array of shape (H, W).
        pred (numpy.ndarray): Predicted depth/disparity array of shape (H, W).

    Returns:
        numpy.ndarray: Aligned depth/disparity array with the same shape as input.

    Raises:
        AssertionError: If ground truth and prediction shapes do not match.

    Example:
        >>> gt_depth = np.random.rand(480, 640)
        >>> pred_depth = gt_depth * 1.2 + 0.1  # Scaled and shifted
        >>> aligned_depth = align_depth_least_square(gt_depth, pred_depth)
        >>> print(f"Alignment error: {np.mean((aligned_depth - gt_depth)**2):.6f}")
    """
    ori_shape = pred.shape  # input shape
    gt = gt.squeeze()  # [H, W]
    pred = pred.squeeze()
    assert (
        gt.shape == pred.shape
    ), f"GT shape: {gt.shape}, Pred shape: {pred.shape} are not matched."

    gt_masked = gt.reshape((-1, 1))
    pred_masked = pred.reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X.flatten()

    aligned_pred = pred * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)
    return aligned_pred


def _delta_log_update(preds: ndarray, target: ndarray) -> Tuple[ndarray, int]:
    """
    Compute delta accuracy metrics for depth estimation evaluation.

    This function calculates the percentage of pixels where the ratio between
    predicted and ground truth depth is within certain thresholds (1.25, 1.25², 1.25³).
    These metrics are commonly used in depth estimation evaluation.

    Args:
        preds (numpy.ndarray): Predicted depth values of shape (N,).
        target (numpy.ndarray): Ground truth depth values of same shape as preds.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Delta accuracy values for each output channel
            - int: Number of valid observations

    Raises:
        ValueError: If prediction and target shapes do not match.

    Example:
        >>> preds = np.random.rand(1000)
        >>> targets = preds * 1.1  # 10% error
        >>> delta_acc, num_obs = _delta_log_update(preds, targets)
        >>> print(f"Delta accuracy: {delta_acc[0]/num_obs:.3f}")
    """
    if preds.shape != target.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape} vs target {target.shape}")

    preds = preds.astype(np.float32) if preds.dtype != np.float32 else preds
    target = target.astype(np.float32) if target.dtype != np.float32 else target

    thresh = np.maximum((target / preds), (preds / target))

    d1 = np.sum(thresh < 1.25, axis=0)

    return d1, target.shape[0]


def _abs_rel_update(preds: ndarray, target: ndarray) -> Tuple[ndarray, int]:
    """
    Compute absolute relative error for depth estimation evaluation.

    This function calculates the mean absolute relative error between predicted
    and ground truth depth values. The relative error is computed as |pred - gt| / gt.

    Args:
        preds (numpy.ndarray): Predicted depth values of shape (N,).
        target (numpy.ndarray): Ground truth depth values of same shape as preds.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Sum of absolute relative errors for each output channel
            - int: Number of valid observations

    Raises:
        ValueError: If prediction and target shapes do not match.

    Example:
        >>> preds = np.random.rand(1000)
        >>> targets = preds * 1.2  # 20% error
        >>> abs_rel_sum, num_obs = _abs_rel_update(preds, targets)
        >>> mean_abs_rel = abs_rel_sum[0] / num_obs
        >>> print(f"Mean absolute relative error: {mean_abs_rel:.4f}")
    """
    if preds.shape != target.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape} vs target {target.shape}")

    preds = preds.astype(np.float32) if preds.dtype != np.float32 else preds
    target = target.astype(np.float32) if target.dtype != np.float32 else target

    sum_abs_error = np.sum(np.abs(preds - target) / target, axis=0)
    return sum_abs_error, target.shape[0]


class MonoDepthEvaluator:
    """
    Comprehensive monocular depth estimation evaluation metrics calculator.

    This class provides a complete evaluation framework for depth estimation models,
    supporting both monocular and stereo depth estimation. It computes multiple
    industry-standard metrics including absolute relative error, delta accuracy,
    and end-point-error for monocular depth estimation.

    The class supports:
    - Depth alignment using least squares optimization
    - Multiple evaluation metrics computation
    - Batch processing of evaluation results
    - Configurable depth range filtering

    Attributes:
        align_gt (bool): Whether to align predictions to ground truth.
        min_depth (float): Minimum valid depth value for evaluation.
        max_depth (float): Maximum valid depth value for evaluation.
        sum_abs_rel (numpy.ndarray): Accumulated absolute relative errors.
        sum_d1 (numpy.ndarray): Accumulated delta accuracy values.
        total (int): Total number of processed samples.
    """

    def __init__(self, align_gt: bool = True, min_depth: float = 0.001, max_depth: float = 10, **kwargs):
        """
        Initialize the depth evaluation metrics calculator.

        Args:
            align_gt (bool, optional): Whether to align predicted depth to ground truth
                using least squares optimization. This can improve evaluation fairness
                when models have systematic scale/shift errors. Defaults to True.
            min_depth (float, optional): Minimum valid depth value for evaluation.
                Pixels with depth below this value are excluded. Defaults to 0.001.
            max_depth (float, optional): Maximum valid depth value for evaluation.
                Pixels with depth above this value are excluded. Defaults to 10.
            **kwargs: Additional keyword arguments (ignored for compatibility).

        Example:
            >>> evaluator = DepthMetric(
            ...     align_gt=True,
            ...     min_depth=0.1,
            ...     max_depth=80.0
            ... )
        """
        self.align_gt = align_gt
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.sum_abs_rel = 0.0
        self.sum_d1 = 0.0

        self.total = 0
        self.total_imgs = 0

    def update(self, post_processed_results: List[Dict]) -> None:
        """
        Update evaluation metrics with a batch of prediction results.

        This method processes a list of prediction results and accumulates the
        corresponding evaluation metrics. Each result should contain predicted
        depth, ground truth depth, and a valid mask.

        Args:
            post_processed_results (List[Dict]): List of dictionaries, where each
                dictionary contains:
                - 'depth_pred': Predicted depth/disparity array
                - 'disp_gt': Ground truth depth/disparity array
                - 'valid_mask': Boolean mask indicating valid pixels

        Example:
            >>> results = [
            ...     {
            ...         'depth_pred': pred_depth_1,
            ...         'disp_gt': gt_depth_1,
            ...         'valid_mask': valid_mask_1
            ...     },
            ...     {
            ...         'depth_pred': pred_depth_2,
            ...         'disp_gt': gt_depth_2,
            ...         'valid_mask': valid_mask_2
            ...     }
            ... ]
            >>> evaluator.update(results)
        """
        pred_list = []
        target_list = []
        for result in post_processed_results:
            pred_i = result['depth_pred']
            gt_i = result['disp_gt']
            valid_mask_i = (result['valid_mask'] &
                            np.isfinite(pred_i) & np.isfinite(gt_i))
            if self.align_gt:
                pred_aligned = align_depth_least_square(
                    gt=gt_i[valid_mask_i],
                    pred=pred_i[valid_mask_i],
                )
                pred_aligned = np.clip(pred_aligned, a_min=1e-8, a_max=None)  # avoid 0 disparity
                target = np.clip(gt_i[valid_mask_i], a_min=1e-8, a_max=None)  # avoid 0 disparity
            else:
                pred_aligned = np.clip(pred_i[valid_mask_i], a_min=self.min_depth, a_max=self.max_depth)  # avoid 0 disparity
                target = np.clip(gt_i[valid_mask_i], a_min=self.min_depth, a_max=self.max_depth)
            pred_list.append(pred_aligned)
            target_list.append(target)
        pred_aligned = np.concatenate(pred_list, axis=0)
        target = np.concatenate(target_list, axis=0)
        sum_abs_rel, num_obs = _abs_rel_update(pred_aligned, target)
        sum_d1, _ = _delta_log_update(pred_aligned, target)

        self.sum_abs_rel += sum_abs_rel
        self.sum_d1 += sum_d1

        self.total += num_obs

    def compute(self):
        """
        Compute final evaluation metrics from accumulated statistics.

        This method calculates the final evaluation metrics by normalizing the
        accumulated statistics by the total number of observations.

        Returns:
            dict: Dictionary containing the computed evaluation metrics:
                - 'd1': Delta accuracy (ratio < 1.25)
                - 'abs_rel': Mean absolute relative error
                - 'bp1': Bad pixel rate (>1px error)
                - 'bp2': Bad pixel rate (>2px error)
                - 'bp3': Bad pixel rate (>3px error)
                - 'epe': Mean end-point-error

        Example:
            >>> evaluator.update(results)
            >>> metrics = evaluator.compute()
            >>> print(f"Mean absolute relative error: {metrics['abs_rel']:.4f}")
            >>> print(f"Delta accuracy: {metrics['d1']:.4f}")
        """
        abs_rel = self.sum_abs_rel / self.total
        d1 = self.sum_d1 / self.total

        return {"d1": float(d1), "abs_rel": float(abs_rel)}
