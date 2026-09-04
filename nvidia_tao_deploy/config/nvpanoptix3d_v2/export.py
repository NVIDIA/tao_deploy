# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for export."""

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_deploy.config.common.common_config import ExportConfig
from nvidia_tao_deploy.config.utils.types import (
    INT_FIELD,
    STR_FIELD
)


@dataclass
class NVPanoptix3Dv2ExportExpConfig(ExportConfig):
    """NVPanoptix3Dv2 export ONNX experiment config."""

    categories_json: Optional[str] = STR_FIELD(
        value=None,
        display_name="categories json",
        description="""
        Path to a categories JSON (a list of :code:`{id, name, isthing}`
        dicts) defining the vocabulary baked into the ONNX graph. Null uses
        the dataset taxonomy.""",
    )
    num_views: int = INT_FIELD(
        value=5,
        default_value=5,
        valid_min=2,
        description="Number of views in the exported multi-view input tensor.",
        display_name="number of views",
    )
