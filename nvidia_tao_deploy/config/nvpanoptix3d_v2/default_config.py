# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from dataclasses import dataclass

from nvidia_tao_deploy.config.utils.types import (
    DATACLASS_FIELD,
)
from nvidia_tao_deploy.config.common.common_config import (
    CommonExperimentConfig
)

from nvidia_tao_deploy.config.nvpanoptix3d_v2.dataset import NVPanoptix3Dv2DatasetConfig
from nvidia_tao_deploy.config.nvpanoptix3d_v2.deploy import NVPanoptix3Dv2GenTRTEngineExpConfig
from nvidia_tao_deploy.config.nvpanoptix3d_v2.evaluate import NVPanoptix3Dv2EvaluateExpConfig
from nvidia_tao_deploy.config.nvpanoptix3d_v2.export import NVPanoptix3Dv2ExportExpConfig
from nvidia_tao_deploy.config.nvpanoptix3d_v2.inference import NVPanoptix3Dv2InferenceExpConfig
from nvidia_tao_deploy.config.nvpanoptix3d_v2.model import NVPanoptix3Dv2ModelConfig
from nvidia_tao_deploy.config.nvpanoptix3d_v2.train import NVPanoptix3Dv2TrainExpConfig


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: NVPanoptix3Dv2ModelConfig = DATACLASS_FIELD(
        NVPanoptix3Dv2ModelConfig(),
        description="Configurable parameters to construct the model for the NVPanoptix3Dv2 experiment.",
    )
    dataset: NVPanoptix3Dv2DatasetConfig = DATACLASS_FIELD(
        NVPanoptix3Dv2DatasetConfig(),
        description="Configurable parameters to construct the dataset for the NVPanoptix3Dv2 experiment.",
    )
    train: NVPanoptix3Dv2TrainExpConfig = DATACLASS_FIELD(
        NVPanoptix3Dv2TrainExpConfig(),
        description="Configurable parameters to construct the trainer for the NVPanoptix3Dv2 experiment.",
    )
    inference: NVPanoptix3Dv2InferenceExpConfig = DATACLASS_FIELD(
        NVPanoptix3Dv2InferenceExpConfig(),
        description="Configurable parameters to construct the inferencer for the NVPanoptix3Dv2 experiment.",
    )
    evaluate: NVPanoptix3Dv2EvaluateExpConfig = DATACLASS_FIELD(
        NVPanoptix3Dv2EvaluateExpConfig(),
        description="Configurable parameters to construct the evaluator for the NVPanoptix3Dv2 experiment.",
    )
    export: NVPanoptix3Dv2ExportExpConfig = DATACLASS_FIELD(
        NVPanoptix3Dv2ExportExpConfig(),
        description="Configurable parameters to construct the exporter for the NVPanoptix3Dv2 experiment.",
    )
    gen_trt_engine: NVPanoptix3Dv2GenTRTEngineExpConfig = DATACLASS_FIELD(
        NVPanoptix3Dv2GenTRTEngineExpConfig(),
        description="Configurable parameters to construct the TensorRT engine builder for a NVPanoptix3Dv2 experiment.",
    )
