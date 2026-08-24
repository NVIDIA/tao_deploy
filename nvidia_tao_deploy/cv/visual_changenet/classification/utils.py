# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet Classifier utility module"""

import numpy as np


class AOIMetrics:
    """AOI Metrics calculation for Visual ChangeNet-Classification"""

    def __init__(self, margin=2.0):
        """Initialize metrics

        Args:
            margin (float): Margin for classification (default is 2.0).
        """
        self.match_fail = np.zeros(1)
        self.tot_fail = np.zeros(1)
        self.match_pass = np.zeros(1)
        self.tot_pass = np.zeros(1)
        self.mismatch_fail = np.zeros(1)
        self.mismatch_pass = np.zeros(1)
        self.margin = margin

    def update(self, preds, target):
        """Update the metrics based on the predictions and targets.

        Args:
            preds (np.ndarray): Predicted distances.
            target (np.ndarray): Target labels.
        """
        preds, target = self._input_format(preds, target)
        for k, euc_dist in enumerate(preds, 0):
            if euc_dist > float(self.margin):
                # Model Classified as FAIL
                if target[k] == 1:
                    self.match_fail += 1
                    self.tot_fail += 1
                else:
                    self.mismatch_pass += 1
                    self.tot_pass += 1
            else:
                # Model Classified as PASS
                if target[k] == 0:
                    self.match_pass += 1
                    self.tot_pass += 1
                else:
                    self.mismatch_fail += 1
                    self.tot_fail += 1

    def compute(self):
        """Compute the metrics.

        Returns:
            dict: Dictionary containing the computed metrics.
        """
        total = self.tot_pass + self.tot_fail
        zero = np.zeros_like(total, dtype=float)
        metric_collect = {}
        metric_collect['total_accuracy'] = (
            zero if total == 0 else
            ((self.match_pass + self.match_fail) / total) * 100
        )
        metric_collect['defect_accuracy'] = (
            zero if self.tot_fail == 0 else
            (self.match_fail / self.tot_fail) * 100
        )
        metric_collect['false_alarm'] = (
            zero if self.tot_pass == 0 else
            (self.mismatch_pass / self.tot_pass) * 100
        )
        metric_collect['false_negative'] = (
            zero if self.tot_fail == 0 else
            (self.mismatch_fail / self.tot_fail) * 100
        )

        return metric_collect

    def _input_format(self, preds, target):
        return preds, target


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
