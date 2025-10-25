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

""" Depth Net Evaluator in distributed mode. """

from typing import Tuple, List, Dict

import numpy as np
from numpy import ndarray


def _check_same_shape(preds: ndarray, target: ndarray) -> None:
    """Check if two arrays have the same shape."""
    if preds.shape != target.shape:
        raise ValueError(f"Input arrays must have the same shape, got {preds.shape} and {target.shape}")


def _abs_rel_update(preds: ndarray, target: ndarray, max_disparity: int = None) -> Tuple[ndarray, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input arrays.

    Args:
        preds (numpy.ndarray): Predicted array
        target (numpy.ndarray): Ground truth array

    Returns:
        sum_abs_error (numpy.ndarray): Sum of absolute relative error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if max_disparity is not None:
        mask = (target > 0.) & (target < max_disparity)
    else:
        mask = (target > 0.)

    preds = preds.astype(np.float32) if preds.dtype.kind != 'f' else preds
    target = target.astype(np.float32) if target.dtype.kind != 'f' else target

    sum_abs_error = np.sum(np.abs(preds[mask] - target[mask]) / target[mask], axis=0)
    return sum_abs_error, target.shape[0]


def _epe_error(preds: ndarray, target: ndarray, max_disparity: int = None) -> Tuple[float, float, float, float, float, int]:
    """Calculates and returns EPE error and other related stereo metrics.

    This private helper function computes several key metrics for stereo
    matching, including End-Point Error (EPE), D1-metric, and bad-pixel
    ratios at different thresholds. It handles batch processing and
    converts input arrays to floating point numbers.

    Args:
        preds (numpy.ndarray): The predicted disparity maps. Expected shape is `(N, C, H, W)`.
        target (numpy.ndarray): The ground truth disparity maps. Expected shape
            is the same as `preds`.
        max_disparity (int, optional): The maximum possible disparity value.
            Used to create a valid mask for the ground truth. Defaults to None.

    Returns:
        Tuple[float, float, float, float, float, int]: A tuple containing the
            sums of the following metrics across the batch:
            - sum_d1 (float): The sum of the D1-metric values.
            - sum_bp1 (float): The sum of the bad-pixel ratios with a threshold of 1.
            - sum_bp2 (float): The sum of the bad-pixel ratios with a threshold of 2.
            - sum_bp3 (float): The sum of the bad-pixel ratios with a threshold of 3.
            - sum_epe_val (float): The sum of the mean EPE values.
            - num_obs (int): The number of observations (i.e., the batch size).
    """
    _check_same_shape(preds, target)
    mask = (target > 0.) & (target < max_disparity)
    preds = preds.astype(np.float32) if preds.dtype.kind != 'f' else preds
    target = target.astype(np.float32) if target.dtype.kind != 'f' else target
    epe = np.abs(preds - target)
    B = target.shape[0]
    epe_mean = (epe[mask]).reshape(B, -1).sum(axis=-1) / (mask.reshape(B, -1).sum(axis=-1) + 1e-8)
    d1, bp1, bp2, bp3, epe_val = 0.0, 0.0, 0.0, 0.0, 0.0
    # assuming batch is not 1
    for i in range(B):
        d1 += np.mean(((epe[i][mask[i]] > 3) & (epe[i][mask[i]] / target[i][mask[i]] > 0.05)).astype(float))
        bp1 += np.mean((epe[i] > 1)[mask[i]].astype(float))
        bp2 += np.mean((epe[i] > 2)[mask[i]].astype(float))
        bp3 += np.mean((epe[i] > 3)[mask[i]].astype(float))
        epe_val += epe_mean[i]
    return d1, bp1, bp2, bp3, epe_val, target.shape[0]


def _sq_rel_update(preds: ndarray, target: ndarray, max_disparity: int = None) -> Tuple[ndarray, int]:
    """Update and returns variables required to compute Squared Relative Error.

    Args:
        preds (numpy.ndarray): Predicted array
        target (numpy.ndarray): Ground truth array
        max_disparity (int, optional): Maximum disparity value

    Returns:
        sum_sq_error (numpy.ndarray): Sum of squared relative error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if max_disparity is not None:
        mask = (target > 0.) & (target < max_disparity)
    else:
        mask = (target > 0.)

    preds = preds.astype(np.float32) if preds.dtype.kind != 'f' else preds
    target = target.astype(np.float32) if target.dtype.kind != 'f' else target

    # Match PyTorch implementation: divide by sum of absolute deviations from mean
    target_masked = target[mask]
    target_mean = np.mean(target_masked)
    sum_abs_dev = np.sum(np.abs(target_masked - target_mean))
    sum_sq_error = np.sum((preds[mask] - target[mask]) ** 2) / (sum_abs_dev + 1e-8)
    return sum_sq_error, target.shape[0]


def _rmse_update(preds: ndarray, target: ndarray, max_disparity: int = None) -> Tuple[ndarray, int]:
    """Update and returns variables required to compute RMSE.

    Args:
        preds (numpy.ndarray): Predicted array
        target (numpy.ndarray): Ground truth array
        max_disparity (int, optional): Maximum disparity value

    Returns:
        rmse (numpy.ndarray): Root mean squared error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if max_disparity is not None:
        mask = (target > 0.) & (target < max_disparity)
    else:
        mask = (target > 0.)

    preds = preds.astype(np.float32) if preds.dtype.kind != 'f' else preds
    target = target.astype(np.float32) if target.dtype.kind != 'f' else target

    # Match PyTorch implementation: compute MSE then take sqrt
    mse = np.mean((preds[mask] - target[mask]) ** 2)
    rmse = np.sqrt(mse)
    return rmse, target.shape[0]


def _rmse_log_update(preds: ndarray, target: ndarray, max_disparity: int = None) -> Tuple[ndarray, int]:
    """Update and returns variables required to compute RMSE log.

    Args:
        preds (numpy.ndarray): Predicted array
        target (numpy.ndarray): Ground truth array
        max_disparity (int, optional): Maximum disparity value

    Returns:
        rmse_log (numpy.ndarray): Root mean squared log error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if max_disparity is not None:
        mask = (target > 0.) & (target < max_disparity)
    else:
        mask = (target > 0.)

    preds = preds.astype(np.float32) if preds.dtype.kind != 'f' else preds
    target = target.astype(np.float32) if target.dtype.kind != 'f' else target

    # Match PyTorch implementation: compute MSE on log values then take sqrt
    log_preds = np.log(np.maximum(preds[mask], 1e-8))
    log_target = np.log(np.maximum(target[mask], 1e-8))
    mse_log = np.mean((log_preds - log_target) ** 2)
    rmse_log = np.sqrt(mse_log)
    return rmse_log, target.shape[0]


class StereoDepthEvaluator:
    """Depth Evaluation Metric Class."""

    def __init__(self, max_disparity=416, **kwargs):
        """Initialize for Depth Metric Class.
        Args:
            max_disparity (float): Maximum disparity value.
            **kwargs: Additional keyword arguments.
        """
        self.max_disparity = max_disparity
        num_outputs = 1
        self.sum_abs_rel = np.zeros(num_outputs)
        self.sum_sq_rel = np.zeros(num_outputs)
        self.sum_rmse = np.zeros(num_outputs)
        self.sum_rmse_log = np.zeros(num_outputs)
        self.sum_d1 = np.zeros(num_outputs)
        self.sum_bp1 = np.zeros(num_outputs)
        self.sum_bp2 = np.zeros(num_outputs)
        self.sum_bp3 = np.zeros(num_outputs)
        self.sum_epe = np.zeros(num_outputs)

        self.total = 0
        self.kwargs = kwargs

    def update(self, post_processed_results: List[Dict]) -> None:
        """Updates the metric results for a stereo estimation model.

        This function calculates various stereo metrics such as D1-metric, EPE,
        and different relative and squared errors. It accumulates these values
        into the class attributes for later aggregation.

        Args:
            post_processed_results: Either a list of dictionaries (new interface) or
                a single dictionary containing:
                - 'depth_pred': Predicted disparity maps of shape (H, W) or (N, C, H, W)
                - 'disp_gt': Ground truth disparity maps of same shape as depth_pred
                - 'valid_mask': Boolean mask indicating valid pixels (optional)

                Or for backward compatibility, can be a tuple of (preds, target) arrays:
                - preds (numpy.ndarray): The predicted disparity maps. The array is
                    expected to have a shape of `(N, C, H, W)`, where N is the batch
                    size, C is the number of output channels (usually 1 for disparity),
                    and H and W are the height and width of the maps.
                - target (numpy.ndarray): The ground truth disparity maps of same shape.

        Returns:
            None: This function does not return any value. It updates the internal
                state of the object by accumulating the calculated metrics.
        """
        # Handle new interface: list of dictionaries
        pred_list = []
        target_list = []
        for result in post_processed_results:
            pred_i = result['depth_pred']
            gt_i = result['disp_gt']

            invalid = gt_i == np.inf
            gt_i[invalid] = 0
            pred_i[invalid] = 0

            pred_list.append(pred_i)
            target_list.append(gt_i)

        # Concatenate all batches
        if not pred_list:
            # Handle empty list case
            return
        preds = np.concatenate(pred_list, axis=0)
        target = np.concatenate(target_list, axis=0)

        # Calculate metrics
        sum_d1, sum_bp1, sum_bp2, sum_bp3, sum_epe_val, num_obs = _epe_error(
            preds, target, max_disparity=self.max_disparity)
        sum_sq_rel, _ = _sq_rel_update(preds, target, max_disparity=self.max_disparity)
        sum_rmse, _ = _rmse_update(preds, target, max_disparity=self.max_disparity)
        sum_rmse_log, _ = _rmse_log_update(preds, target, max_disparity=self.max_disparity)
        sum_abs_rel, _ = _abs_rel_update(preds, target, max_disparity=self.max_disparity)
        self.sum_abs_rel += sum_abs_rel
        self.sum_sq_rel += sum_sq_rel
        self.sum_rmse += sum_rmse
        self.sum_rmse_log += sum_rmse_log
        self.sum_d1 += sum_d1
        self.sum_bp1 += sum_bp1
        self.sum_bp2 += sum_bp2
        self.sum_bp3 += sum_bp3
        self.sum_epe += sum_epe_val
        self.total += num_obs

    def compute(self):
        """Computes and returns the final depth evaluation metrics.

        This function aggregates the accumulated metric sums (`self.sum_*`)
        and divides them by the total number of observations (`self.total`)
        to compute the final, average metrics over the entire dataset.

        Returns:
            dict: A dictionary containing the following aggregated scalar
                metric values:
                - 'd1' (float): The final D1-metric.
                - 'bp1' (float): The final bad-pixel metric at threshold 1.
                - 'bp2' (float): The final bad-pixel metric at threshold 2.
                - 'bp3' (float): The final bad-pixel metric at threshold 3.
                - 'epe' (float): The final End-Point Error.
                - 'abs_rel' (float): The final absolute relative error.
                - 'sq_rel' (float): The final squared relative error.
                - 'rmse' (float): The final Root Mean Squared Error.
                - 'rmse_log' (float): The final Root Mean Squared Logarithmic Error.
        """
        abs_rel = self.sum_abs_rel / self.total
        sq_rel = self.sum_sq_rel / self.total
        rmse = self.sum_rmse / self.total  # Already RMSE from helper function
        rmse_log = self.sum_rmse_log / self.total  # Already RMSE log from helper function
        d1 = self.sum_d1 / self.total
        bp1 = self.sum_bp1 / self.total
        bp2 = self.sum_bp2 / self.total
        bp3 = self.sum_bp3 / self.total
        epe = self.sum_epe / self.total

        return {"d1": float(d1),
                "bp1": float(bp1), "bp2": float(bp2), "bp3": float(bp3),
                "epe": float(epe), "abs_rel": float(abs_rel), "sq_rel": float(sq_rel),
                "rmse": float(rmse), 'rmse_log': float(rmse_log)}
