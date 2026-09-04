# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the inferencer."""

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_deploy.config.common.common_config import InferenceConfig
from nvidia_tao_deploy.config.utils.types import (
    BOOL_FIELD,
    FLOAT_FIELD,
    STR_FIELD
)


@dataclass
class NVPanoptix3Dv2InferenceExpConfig(InferenceConfig):
    """NVPanoptix3Dv2 inference configuration."""

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
    categories_json: Optional[str] = STR_FIELD(
        value=None,
        display_name="categories json",
        description="""
        Path to a categories JSON (a list of :code:`{id, name, isthing}` dicts)
        defining the open-vocabulary class list to predict against. Null uses
        the dataset's own categories.""",
    )
    output_dir: Optional[str] = STR_FIELD(
        value=None,
        display_name="output directory",
        description="Directory for prediction artifacts. Null uses results_dir.",
    )
    save_full_point_cloud: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="save full point cloud",
        description="""
        Reasoning variant only. Save one sample-level NPZ containing the full
        RGB point cloud and fused predicted segmentation mask. This is opt-in
        because dense multi-view point clouds can consume substantial disk
        space.
        """,
    )
