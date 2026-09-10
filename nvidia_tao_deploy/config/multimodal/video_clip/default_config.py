# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP (InternVideo2-CLIP) deploy experiment configuration."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from omegaconf import MISSING

from nvidia_tao_deploy.config.common.common_config import (
    CommonExperimentConfig,
    GenTrtEngineConfig,
    TrtConfig,
)
from nvidia_tao_deploy.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    DICT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)


# =============================================================================
# Model Config
# =============================================================================
@dataclass
class VideoCLIPModelConfig:
    """Video-CLIP model configuration (subset needed for deploy preprocessing)."""

    type: str = STR_FIELD(
        value="internvideo2-clip-l14",
        default_value="internvideo2-clip-l14",
        description="Video-CLIP model type.",
        display_name="Model Type",
    )
    image_size: int = INT_FIELD(
        value=224,
        default_value=224,
        description="Square input resolution H=W for the vision tower. "
                    "Overridden by the engine's image input shape at runtime.",
        display_name="Image Size",
    )
    canonicalize_text: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Apply lowercase + punctuation-removal before tokenization. "
                    "Kept False for retrieval (matches training).",
        display_name="Canonicalize Text",
    )


# =============================================================================
# Dataset Config
# =============================================================================
@dataclass
class VideoCLIPEvalDataConfig:
    """Validation data for multi-relevance text->video retrieval eval."""

    gt_queries: str = STR_FIELD(
        value=MISSING,
        default_value=MISSING,
        description="Explicit-relevance eval file (e.g. domain_test_*.json) with "
                    "'gallery' and 'queries' (each query record: query, chunk_id, "
                    "slice, relevant_clip_ids).",
        display_name="Ground-truth Queries",
    )
    metadata: str = STR_FIELD(
        value=MISSING,
        default_value=MISSING,
        description="vadr1_chunks metadata JSON used to resolve each gallery "
                    "chunk_id to its video path and time window.",
        display_name="Chunk Metadata",
    )
    video_root: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Optional root prepended to relative video paths.",
        display_name="Video Root",
    )
    path_prefix_mapping: Dict[str, str] = DICT_FIELD(
        {},
        description="Optional mapping from original path prefixes to local "
                    "prefixes, applied before video_root.",
        display_name="Path Prefix Mapping",
    )
    batch_size: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        description="Clips per batch when embedding the gallery.",
        display_name="Batch Size",
    )
    num_workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=0,
        description="Reserved (single-process decord loading is used).",
        display_name="Number of Workers",
    )


@dataclass
class VideoCLIPDatasetConfig:
    """Dataset configuration for Video-CLIP deploy evaluation."""

    val: VideoCLIPEvalDataConfig = DATACLASS_FIELD(
        VideoCLIPEvalDataConfig(),
        description="Validation/retrieval dataset configuration.",
    )


# =============================================================================
# Inference / Eval Config
# =============================================================================
@dataclass
class VideoCLIPInferenceEvalConfig:
    """Configuration for Video-CLIP TRT inference and evaluation."""

    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to the combined TensorRT engine.",
        display_name="TRT Engine Path",
    )
    batch_size: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        description="Batch size for embedding extraction.",
        display_name="Batch Size",
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Directory to save results.",
        display_name="Results Directory",
    )
    text_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Optional text file (one prompt per line) for text-embedding "
                    "inference.",
        display_name="Text File",
    )
    num_gpus: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="Number of GPUs (TRT eval runs on a single device).",
        display_name="Number of GPUs",
    )
    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        default_value=[0],
        description="GPU device IDs.",
        display_name="GPU IDs",
    )


# =============================================================================
# TRT Engine Config
# =============================================================================
@dataclass
class VideoCLIPTrtConfig(TrtConfig):
    """Video-CLIP TensorRT configuration."""

    data_type: str = STR_FIELD(
        value="fp32",
        default_value="fp32",
        valid_options="fp32,fp16",
        description="TensorRT precision: FP32 or FP16. FP16 automatically "
                    "keeps numerically sensitive InternVideo2 vision block "
                    "normalization reductions in FP32.",
        display_name="Data Type",
    )
    strongly_typed: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Build a strongly-typed TensorRT engine that honors the "
                    "explicit precision of a mixed-precision ONNX (e.g. from "
                    "ModelOpt AutoCast, which keeps the vision RMSNorm in FP32). "
                    "Required for a pin-free FP16 engine: a weakly-typed FP16 "
                    "build re-lowers the RMSNorm reduction and collapses accuracy. "
                    "Use with data_type=fp32 (graph carries the precision).",
        display_name="Strongly Typed",
    )
    max_batch_size: int = INT_FIELD(
        value=16,
        default_value=16,
        valid_min=1,
        description="Maximum batch size in the TRT optimization profile.",
        display_name="Maximum batch size",
    )


@dataclass
class VideoCLIPGenTrtEngineConfig(GenTrtEngineConfig):
    """Video-CLIP TRT engine generation config."""

    tensorrt: VideoCLIPTrtConfig = DATACLASS_FIELD(VideoCLIPTrtConfig())


# =============================================================================
# Experiment Config
# =============================================================================
@dataclass
class VideoCLIPExperimentConfig(CommonExperimentConfig):
    """Video-CLIP deploy experiment config."""

    model_name: Optional[str] = STR_FIELD(
        value="video_clip",
        default_value="video_clip",
        description="Name of model for task invocation.",
        display_name="Model Name",
    )
    model: VideoCLIPModelConfig = DATACLASS_FIELD(
        VideoCLIPModelConfig(),
        description="Model config.",
    )
    dataset: VideoCLIPDatasetConfig = DATACLASS_FIELD(
        VideoCLIPDatasetConfig(),
        description="Dataset config.",
    )
    evaluate: VideoCLIPInferenceEvalConfig = DATACLASS_FIELD(
        VideoCLIPInferenceEvalConfig(),
        description="Evaluation config.",
    )
    inference: VideoCLIPInferenceEvalConfig = DATACLASS_FIELD(
        VideoCLIPInferenceEvalConfig(),
        description="Inference config.",
    )
    gen_trt_engine: VideoCLIPGenTrtEngineConfig = DATACLASS_FIELD(
        VideoCLIPGenTrtEngineConfig(),
        description="TensorRT engine generation config.",
    )
