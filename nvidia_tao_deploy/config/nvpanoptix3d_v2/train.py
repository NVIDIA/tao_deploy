# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the trainer."""

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_deploy.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    STR_FIELD
)
from nvidia_tao_deploy.config.common.common_config import TrainConfig


@dataclass
class MetricDepthLossConfig:
    """Metric depth loss config."""

    weight: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        valid_min=0.0,
        display_name="metric depth weight",
        description="Outer multiplier on the metric-depth loss. Set to 0 to disable.",
    )
    silog_weight: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        display_name="SILog weight",
        description="Weight of the scale-invariant log term.",
    )
    absrel_weight: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0.0,
        display_name="AbsRel weight",
        description="Weight of the auxiliary absolute-relative term. Set to 0 to disable.",
    )
    silog_lambda: float = FLOAT_FIELD(
        value=0.85,
        default_value=0.85,
        valid_min=0.0,
        valid_max=1.0,
        display_name="SILog lambda",
        description="SILog variance-mixing constant.",
    )
    min_depth: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        math_cond="> 0.0",
        display_name="min depth",
        description="Lower bound of the supervised metric-depth range, in metres.",
    )
    max_depth: float = FLOAT_FIELD(
        value=20.0,
        default_value=20.0,
        math_cond="> 0.0",
        display_name="max depth",
        description="Upper bound of the supervised metric-depth range, in metres.",
    )


@dataclass
class PanopticLossConfig:
    """Loss weights for the panoptic variant."""

    class_weight: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0.0,
        display_name="class loss coefficient",
        description="Weight of the open-vocabulary classification loss.",
        popular="yes",
    )
    rank_weight: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        display_name="rank loss coefficient",
        description="""
        Weight of the matched-query cross-entropy that trains the
        one-label-per-query top-1 decision.""",
    )
    objectness_weight: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        display_name="objectness loss coefficient",
        description="Weight of the binary matched/unmatched query confidence loss.",
    )
    objectness_no_object_weight: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0.0,
        display_name="objectness no-object coefficient",
        description="Relative weight applied to the no-object objectness target.",
    )
    objectness_ignore_overlap_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        display_name="objectness ignore overlap threshold",
        description="""
        Unmatched queries whose predicted mask lies at least this fraction on
        ignored or crowd pixels receive no objectness target.""",
    )
    mask_weight: float = FLOAT_FIELD(
        value=20.0,
        default_value=20.0,
        valid_min=0.0,
        display_name="mask loss coefficient",
        description="Weight of the point-sampled binary mask loss.",
        popular="yes",
    )
    dice_weight: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        display_name="dice loss coefficient",
        description="Weight of the dice loss of the binary mask.",
        popular="yes",
    )
    num_loss_points: int = INT_FIELD(
        value=12288,
        default_value=12288,
        valid_min=1,
        display_name="number of loss points",
        description="Number of points sampled per mask for the mask and dice losses.",
    )


@dataclass
class ReasoningLossConfig:
    """Loss weights for the reasoning variant."""

    text_weight: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        display_name="text loss coefficient",
        description="Weight of the Qwen next-token cross-entropy on the answer.",
        popular="yes",
    )
    mask_weight: float = FLOAT_FIELD(
        value=20.0,
        default_value=20.0,
        valid_min=0.0,
        display_name="mask loss coefficient",
        description="Weight of the SAM3 binary mask loss.",
        popular="yes",
    )
    dice_weight: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        display_name="dice loss coefficient",
        description="Weight of the dice loss on the SAM3 mask.",
        popular="yes",
    )
    score_weight: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        display_name="score loss coefficient",
        description="Weight of the SAM3 query-selection score loss.",
    )


@dataclass
class NVPanoptix3Dv2TrainExpConfig(TrainConfig):
    """Train experiment config."""

    accum_iter: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        display_name="gradient accumulation iterations",
        description="Number of batches accumulated before an optimizer step.",
    )
    lr: float = FLOAT_FIELD(
        value=1.0e-4,
        default_value=1.0e-4,
        math_cond="> 0.0",
        display_name="learning rate",
        description="Peak learning rate after warmup.",
    )
    min_lr: float = FLOAT_FIELD(
        value=1.0e-6,
        default_value=1.0e-6,
        valid_min=0.0,
        display_name="minimum learning rate",
        description="Floor of the cosine decay schedule.",
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.05,
        default_value=0.05,
        valid_min=0.0,
        display_name="weight decay",
        description="""
        AdamW weight decay. Applied only to matrix and convolution weights;
        norms, biases, scalar temperatures and embeddings get zero decay.""",
    )
    warmup_epochs: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0.0,
        display_name="warmup epochs",
        description="Length of the linear learning-rate warmup, in epochs.",
    )
    precision: str = STR_FIELD(
        value="fp32",
        default_value="fp32",
        display_name="precision",
        description="Precision to run the training on.",
        valid_options=",".join(["fp32", "bf16", "fp16"])
    )
    is_dry_run: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="is dry run",
        description="Flag to run the trainer in dry-run mode.",
    )
    clip_grad_norm: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        display_name="clip gradient norm",
        description="Amount to clip the gradient by L2 norm. 0 disables clipping.",
    )
    val_check_interval: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        valid_max=1.0,
        display_name="val check interval",
        description="Fraction of a training epoch between validation runs.",
    )
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        display_name="pretrained model path",
        description="""
        Checkpoint to warm-start the model weights from. Ignored when a resume
        checkpoint is present, which always takes priority.""",
    )
    log_interval: int = INT_FIELD(
        value=50,
        default_value=50,
        valid_min=1,
        display_name="log interval",
        description="Number of training steps between scalar logs.",
    )
    metric_depth: MetricDepthLossConfig = DATACLASS_FIELD(
        MetricDepthLossConfig(),
        display_name="metric depth loss",
        description="""
        Hyper parameters for the metric-depth loss. Shared by both variants,
        which supervise the same metric scale head.""",
    )
    panoptic: PanopticLossConfig = DATACLASS_FIELD(
        PanopticLossConfig(),
        display_name="panoptic losses",
        description="Loss weights for the panoptic variant. Read when model_type is panoptic.",
    )
    reasoning: ReasoningLossConfig = DATACLASS_FIELD(
        ReasoningLossConfig(),
        display_name="reasoning losses",
        description="Loss weights for the reasoning variant. Read when model_type is reasoning.",
    )
