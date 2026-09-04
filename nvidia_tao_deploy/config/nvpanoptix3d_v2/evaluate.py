# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the evaluation."""

from dataclasses import dataclass

from nvidia_tao_deploy.config.common.common_config import EvaluateConfig
from nvidia_tao_deploy.config.utils.types import FLOAT_FIELD


@dataclass
class NVPanoptix3Dv2EvaluateExpConfig(EvaluateConfig):
    """NVPanoptix3Dv2 evaluation configuration."""

    cls_threshold: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        display_name="class threshold",
        description="Minimum class score for a query to enter panoptic post-processing.",
    )
    mask_threshold: float = FLOAT_FIELD(
        value=0.25,
        default_value=0.25,
        valid_min=0.0,
        valid_max=1.0,
        display_name="mask threshold",
        description="Probability threshold binarizing the predicted masks.",
    )
    overlap_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        display_name="overlap threshold",
        description="""
        Minimum fraction of a segment that must survive occlusion by
        higher-scoring segments for it to be kept.""",
    )
