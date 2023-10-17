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
        metric_collect = {}
        metric_collect['total_accuracy'] = ((self.match_pass + self.match_fail) / (self.tot_pass + self.tot_fail)) * 100
        metric_collect['defect_accuracy'] = 0 if self.tot_fail == 0 else (self.match_fail / self.tot_fail) * 100
        metric_collect['false_alarm'] = (self.mismatch_pass / (self.tot_pass + self.tot_fail)) * 100
        metric_collect['false_negative'] = (self.mismatch_fail / (self.tot_pass + self.tot_fail)) * 100

        return metric_collect

    def _input_format(self, preds, target):
        return preds, target


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
