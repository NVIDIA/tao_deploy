# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the model."""

from typing import List, Optional
from dataclasses import dataclass

from nvidia_tao_deploy.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD
)


@dataclass
class MetricDepthHeadConfig:
    """Metric scale head config."""

    enable: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="enable metric depth head",
        description="""
        Flag to enable the metric-scale head. When enabled a single scene-level
        log-scale is regressed from the pooled VGGT patch tokens and applied to
        the relative depth and world points of every view.""",
    )
    feat_dim: int = INT_FIELD(
        value=2048,
        default_value=2048,
        valid_min=1,
        display_name="scene token dim",
        description="Channel width of the pooled VGGT patch token fed to the metric-scale head.",
    )
    hidden_dims: List[int] = LIST_FIELD(
        arrList=[256, 64],
        default_value=[256, 64],
        display_name="hidden dims",
        description="Hidden widths of the metric-scale MLP.",
    )
    metric_context_views: int = INT_FIELD(
        value=5,
        default_value=5,
        valid_min=1,
        display_name="metric context views",
        description="""
        Number of leading views used to estimate the single scene scale. The
        estimated scale is then applied to every inference view, including
        views outside this context window.""",
    )
    predict_shift: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="predict depth shift",
        description="""
        Also regress an additive depth shift, so
        :code:`d_metric = s * d_rel + b`. This config is used by the reasoning
        variant; the panoptic builder always uses scale-only correction. The
        shift lives in depth units and is never applied to 3D points.""",
    )


@dataclass
class BackboneConfig:
    """Backbone config."""

    pretrained_backbone_path: str = STR_FIELD(
        value="",
        default_value="",
        display_name="pretrained backbone path",
        description="""
        Path to the pretrained VGGT checkpoint. When empty the backbone falls
        back to the :code:`facebook/VGGT-1B` weights from the model hub.""",
    )
    load_from_hf: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="load backbone from HF",
        description="""
        Fall back to the :code:`facebook/VGGT-1B` hub weights when
        pretrained_backbone_path is unset. Keep this off so a missing or
        unmounted commercial checkpoint fails loudly instead of silently
        pulling non-commercial weights.""",
    )
    strict_load: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="strict backbone load",
        description="Require the VGGT checkpoint to match the built heads exactly.",
    )
    metric_depth_head: MetricDepthHeadConfig = DATACLASS_FIELD(
        MetricDepthHeadConfig(),
        display_name="metric depth head",
        description="Configuration hyper parameters for the metric-scale head.",
    )


@dataclass
class FeatureFusionConfig:
    """Feature fusion config."""

    dino_dim: int = INT_FIELD(
        value=1024,
        default_value=1024,
        valid_min=1,
        display_name="dino dim",
        description="Channel width of the DINOv2 token stream.",
    )
    vggt_dim: int = INT_FIELD(
        value=2048,
        default_value=2048,
        valid_min=1,
        display_name="vggt dim",
        description="Channel width of the VGGT token stream.",
    )
    hidden_dim: int = INT_FIELD(
        value=768,
        default_value=768,
        valid_min=1,
        display_name="hidden dim",
        description="Channel width of the fused token map.",
        popular="yes",
    )
    num_heads: int = INT_FIELD(
        value=12,
        default_value=12,
        valid_min=1,
        display_name="number of heads",
        description="Number of attention heads in the per-view spatial mixer.",
    )
    num_layers: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=0,
        display_name="number of layers",
        description="Number of blocks in the per-view spatial mixer.",
    )
    ff_dim_mult: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        display_name="feedforward dim multiplier",
        description="Feedforward expansion ratio inside each spatial-mixer block.",
    )


@dataclass
class PanopticDecoderConfig:
    """Panoptic decoder config."""

    hidden_dim: int = INT_FIELD(
        value=768,
        default_value=768,
        valid_min=1,
        display_name="hidden dim",
        description="Channel width of the decoder queries and memory.",
        popular="yes",
    )
    mask_dim: int = INT_FIELD(
        value=384,
        default_value=384,
        valid_min=1,
        display_name="mask dim",
        description="Channel width of the high-resolution mask features.",
    )
    ff_dim: int = INT_FIELD(
        value=2048,
        default_value=2048,
        valid_min=1,
        display_name="feedforward dim",
        description="Width of the feedforward network in the mask transformer.",
    )
    num_queries: int = INT_FIELD(
        value=200,
        default_value=200,
        valid_min=1,
        display_name="number of queries",
        description="Number of learned object queries.",
        popular="yes",
    )
    num_heads: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        display_name="number of heads",
        description="Number of attention heads in the mask transformer.",
    )
    dec_layers: int = INT_FIELD(
        value=6,
        default_value=6,
        valid_min=1,
        display_name="decoder layers",
        description="Number of mask-transformer decoder layers.",
    )
    fixed_vocab: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="fixed vocabulary",
        description="""
        Flag to encode the full class vocabulary once up front. The collator
        then performs cache lookups instead of re-encoding text every step.""",
    )
    label_mode: str = STR_FIELD(
        value="sigmoid",
        default_value="sigmoid",
        display_name="label mode",
        description="Classification head activation used for open-vocabulary labels.",
        valid_options=",".join(["sigmoid", "softmax"])
    )
    deep_supervision: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="deep supervision",
        description="Flag to supervise the intermediate decoder layer outputs.",
    )
    enable_objectness: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="enable objectness",
        description="""
        Flag to enable the vocabulary-independent matched/unmatched query
        confidence head.""",
    )


@dataclass
class UpscalerConfig:
    """LoftUp mask-feature upscaler config."""

    input_dim: int = INT_FIELD(
        value=768,
        default_value=768,
        valid_min=1,
        display_name="input dim",
        description="Channel width of the fused token map fed to the upscaler.",
    )
    dim: int = INT_FIELD(
        value=384,
        default_value=384,
        valid_min=1,
        display_name="output dim",
        description="Channel width of the upscaled mask features.",
    )
    output_stride: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=1,
        display_name="output stride",
        description="Stride of the upscaled mask features relative to the input image.",
    )
    patch_size: int = INT_FIELD(
        value=14,
        default_value=14,
        valid_min=1,
        display_name="patch size",
        description="Patch size of the token map consumed by the upscaler.",
    )
    color_feats: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="use color features",
        description="Flag to condition the upscaler on the input RGB image.",
    )
    n_freqs: int = INT_FIELD(
        value=20,
        default_value=20,
        valid_min=1,
        display_name="number of frequencies",
        description="Number of Fourier frequencies in the implicit featurizer.",
    )
    num_heads: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        display_name="number of heads",
        description="Number of attention heads in the upscaler cross-attention blocks.",
    )
    num_layers: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=1,
        display_name="number of layers",
        description="Number of upscaler cross-attention blocks.",
    )


@dataclass
class PanopticVariantConfig:
    """Architecture of the panoptic-segmentation variant."""

    feature_fusion: FeatureFusionConfig = DATACLASS_FIELD(
        FeatureFusionConfig(),
        display_name="feature fusion",
        description="Configuration hyper parameters for the DINOv2/VGGT feature fusion.",
    )
    panoptic_decoder: PanopticDecoderConfig = DATACLASS_FIELD(
        PanopticDecoderConfig(),
        display_name="panoptic decoder",
        description="Configuration hyper parameters for the panoptic segmentation decoder.",
    )
    upscaler: UpscalerConfig = DATACLASS_FIELD(
        UpscalerConfig(),
        display_name="upscaler",
        description="Configuration hyper parameters for the mask-feature upscaler.",
    )


@dataclass
class LoraConfig:
    """LoRA adapter config for the Qwen reasoner."""

    r: int = INT_FIELD(
        value=16,
        default_value=16,
        valid_min=1,
        display_name="LoRA rank",
        description="Rank of the LoRA update matrices.",
    )
    alpha: int = INT_FIELD(
        value=32,
        default_value=32,
        valid_min=1,
        display_name="LoRA alpha",
        description="LoRA scaling factor.",
    )
    dropout: float = FLOAT_FIELD(
        value=0.05,
        default_value=0.05,
        valid_min=0.0,
        valid_max=1.0,
        display_name="LoRA dropout",
        description="Dropout applied inside the LoRA adapters.",
    )


@dataclass
class QwenConfig:
    """Qwen vision-language reasoner config."""

    model_id: str = STR_FIELD(
        value="Qwen/Qwen3-VL-4B-Instruct",
        default_value="Qwen/Qwen3-VL-4B-Instruct",
        display_name="Qwen model id",
        description="Hugging Face model id or local directory for Qwen3-VL-4B.",
    )
    seg_token: str = STR_FIELD(
        value="[SEG]",
        default_value="[SEG]",
        display_name="segmentation token",
        description="Added token whose hidden state is projected into the SAM3 prompt space.",
    )
    freeze_vision: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="freeze vision tower",
        description="Flag to freeze the Qwen vision tower.",
    )
    dtype: str = STR_FIELD(
        value="bfloat16",
        default_value="bfloat16",
        display_name="dtype",
        description="Compute dtype for the Qwen weights.",
        valid_options=",".join(["float32", "bfloat16", "float16"])
    )
    attn_implementation: str = STR_FIELD(
        value="sdpa",
        default_value="sdpa",
        display_name="attention implementation",
        description="Attention kernel used by the Qwen backbone.",
        valid_options=",".join(["sdpa", "eager", "flash_attention_2"])
    )
    use_lora: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="use LoRA",
        description="Train LoRA adapters instead of the full Qwen weights.",
    )
    lora: LoraConfig = DATACLASS_FIELD(
        LoraConfig(),
        display_name="LoRA",
        description="Hyper parameters for the LoRA adapters.",
    )


@dataclass
class Sam3Config:
    """Frozen SAM3 mask-decoder config."""

    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        display_name="SAM3 checkpoint",
        description="""
        Path to a local SAM3 checkpoint. Null downloads the weights from the
        hub when load_from_hf is set.""",
    )
    load_from_hf: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="load SAM3 from HF",
        description="Download the SAM3 weights from the hub when no checkpoint is given.",
    )
    resolution: int = INT_FIELD(
        value=1008,
        default_value=1008,
        valid_min=32,
        display_name="SAM3 resolution",
        description="Square resolution the input views are resized to for SAM3.",
    )


@dataclass
class SamBridgeConfig:
    """``[SEG]``-to-SAM3 prompt projector config."""

    sam_prompt_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        display_name="SAM prompt dim",
        description="Width of the SAM3 language-prompt tensor.",
    )
    hidden_dim: int = INT_FIELD(
        value=4096,
        default_value=4096,
        valid_min=1,
        display_name="projector hidden dim",
        description="Hidden width of the ``[SEG]`` -> SAM prompt MLP.",
    )
    dropout: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        display_name="projector dropout",
        description="Dropout inside the projector MLP.",
    )


@dataclass
class ReasoningVariantConfig:
    """Architecture of the reasoning-segmentation variant."""

    qwen: QwenConfig = DATACLASS_FIELD(
        QwenConfig(),
        display_name="Qwen reasoner",
        description="Configuration hyper parameters for the Qwen VLM reasoner.",
    )
    sam3: Sam3Config = DATACLASS_FIELD(
        Sam3Config(),
        display_name="SAM3",
        description="Configuration hyper parameters for the frozen SAM3 mask decoder.",
    )
    sam_bridge: SamBridgeConfig = DATACLASS_FIELD(
        SamBridgeConfig(),
        display_name="SAM bridge",
        description="Configuration hyper parameters for the ``[SEG]`` prompt projector.",
    )
    point_mask_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        display_name="point mask threshold",
        description="Foreground threshold applied to the SAM3 mask before lifting to 3D.",
    )
    point_conf_threshold: Optional[float] = FLOAT_FIELD(
        value=None,
        display_name="point confidence threshold",
        description="""
        Optional VGGT confidence gate on the lifted points. Null keeps every
        geometrically valid point.""",
    )


@dataclass
class NVPanoptix3Dv2ModelConfig:
    """NVPanoptix3Dv2 model config.

    Both variants share the frozen VGGT backbone and the metric scale head
    under :code:`backbone`; :code:`model_type` selects which variant-specific
    block is read and which Lightning module is built.
    """

    model_type: str = STR_FIELD(
        value="panoptic",
        default_value="panoptic",
        display_name="model type",
        description="""
        Which NVPanoptix3Dv2 variant to build:
        * :code:`panoptic`  -- open-vocabulary panoptic segmentation.
        * :code:`reasoning` -- Qwen-driven reasoning segmentation.""",
        valid_options=",".join(["panoptic", "reasoning"]),
        popular="yes",
    )
    img_size: int = INT_FIELD(
        value=518,
        default_value=518,
        valid_min=32,
        display_name="image size",
        description="Image size the VGGT backbone is instantiated for.",
    )
    patch_size: int = INT_FIELD(
        value=14,
        default_value=14,
        valid_min=1,
        display_name="patch size",
        description="Patch size of the VGGT backbone.",
    )
    embed_dim: int = INT_FIELD(
        value=1024,
        default_value=1024,
        valid_min=1,
        display_name="embedding dim",
        description="Token width of the VGGT backbone.",
    )
    backbone: BackboneConfig = DATACLASS_FIELD(
        BackboneConfig(),
        display_name="backbone",
        description="""
        Configuration hyper parameters for the frozen VGGT backbone and the
        metric scale head. Shared by both variants.""",
    )
    panoptic: PanopticVariantConfig = DATACLASS_FIELD(
        PanopticVariantConfig(),
        display_name="panoptic variant",
        description="Architecture of the panoptic variant. Read when model_type is panoptic.",
    )
    reasoning: ReasoningVariantConfig = DATACLASS_FIELD(
        ReasoningVariantConfig(),
        display_name="reasoning variant",
        description="Architecture of the reasoning variant. Read when model_type is reasoning.",
    )
