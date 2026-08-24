# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet classification metric regression tests."""

import numpy as np
import pytest

from nvidia_tao_deploy.cv.visual_changenet.classification.utils import AOIMetrics


@pytest.mark.visual_changenet
def test_false_rates_use_their_class_denominators():
    """FPR is over pass samples and FNR is over defect samples."""
    metrics = AOIMetrics(margin=2.0)
    predictions = np.array([0, 0, 0, 0, 0, 0, 3, 3, 3, 0])
    targets = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    metrics.update(predictions, targets)
    result = metrics.compute()

    assert result["total_accuracy"].item() == pytest.approx(70.0)
    assert result["defect_accuracy"].item() == pytest.approx(50.0)
    assert result["false_alarm"].item() == pytest.approx(25.0)
    assert result["false_negative"].item() == pytest.approx(50.0)


@pytest.mark.visual_changenet
@pytest.mark.parametrize(
    ("predictions", "targets", "expected_false_alarm", "expected_false_negative"),
    (
        (np.array([0, 3]), np.array([1, 1]), 0.0, 50.0),
        (np.array([3, 0]), np.array([0, 0]), 50.0, 0.0),
    ),
)
def test_false_rates_handle_single_class_inputs(
    predictions, targets, expected_false_alarm, expected_false_negative
):
    """An absent class gets zero while the present class keeps its error rate."""
    metrics = AOIMetrics(margin=2.0)
    metrics.update(predictions, targets)

    result = metrics.compute()

    assert np.isfinite(result["total_accuracy"])
    assert result["false_alarm"].item() == expected_false_alarm
    assert result["false_negative"].item() == expected_false_negative


@pytest.mark.visual_changenet
def test_metrics_handle_empty_state():
    """Computing before an update returns finite zero metrics."""
    result = AOIMetrics().compute()

    assert all(metric.item() == 0.0 for metric in result.values())
