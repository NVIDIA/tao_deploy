# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the dataset."""

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
class PhotometricAugmentationConfig:
    """Geometry-safe photometric augmentation config.

    One recipe is sampled per multi-view sample and applied identically to
    every view, so no spatial operation or geometric target is modified.
    Training only.
    """

    enabled: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="enable photometric augmentation",
        description="Flag to enable RGB-only training augmentation.",
    )
    color_jitter_prob: float = FLOAT_FIELD(
        value=0.8,
        default_value=0.8,
        valid_min=0.0,
        valid_max=1.0,
        display_name="color jitter probability",
        description="Probability of applying brightness/contrast/saturation jitter.",
    )
    brightness: float = FLOAT_FIELD(
        value=0.10,
        default_value=0.10,
        valid_min=0.0,
        display_name="brightness",
        description="Maximum relative brightness jitter.",
    )
    contrast: float = FLOAT_FIELD(
        value=0.10,
        default_value=0.10,
        valid_min=0.0,
        display_name="contrast",
        description="Maximum relative contrast jitter.",
    )
    saturation: float = FLOAT_FIELD(
        value=0.15,
        default_value=0.15,
        valid_min=0.0,
        display_name="saturation",
        description="Maximum relative saturation jitter.",
    )
    gamma_exposure_prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        display_name="gamma/exposure probability",
        description="Probability of applying the gamma and exposure jitter.",
    )
    gamma: float = FLOAT_FIELD(
        value=0.10,
        default_value=0.10,
        valid_min=0.0,
        valid_max=1.0,
        display_name="gamma",
        description="Maximum gamma jitter. Must be in [0, 1).",
    )
    exposure_ev: float = FLOAT_FIELD(
        value=0.15,
        default_value=0.15,
        valid_min=0.0,
        display_name="exposure EV",
        description="Maximum exposure jitter in EV stops.",
    )
    grayscale_prob: float = FLOAT_FIELD(
        value=0.05,
        default_value=0.05,
        valid_min=0.0,
        valid_max=1.0,
        display_name="grayscale probability",
        description="Probability of converting a sample to grayscale.",
    )


@dataclass
class PanopticDatasetConfig:
    """Dataset config for the panoptic variant (preprocessed ScanNet++ trees)."""

    train_preprocessed_root: Optional[str] = STR_FIELD(
        value=None,
        display_name="training preprocessed root",
        description="""
        Preprocessed ScanNet++ training root containing
        :code:`all_metadata.npz`, :code:`categories.json`, and the scene
        directories. The metadata file includes the training pair table.""",
    )
    val_preprocessed_root: Optional[str] = STR_FIELD(
        value=None,
        display_name="validation preprocessed root",
        description="""
        Preprocessed ScanNet++ validation root containing
        :code:`all_metadata.npz`, :code:`categories.json`, and the scene
        directories. The metadata file includes the validation pair table.""",
    )
    photometric_augmentation: PhotometricAugmentationConfig = DATACLASS_FIELD(
        PhotometricAugmentationConfig(),
        display_name="photometric augmentation",
        description="Configuration parameters for the training-only RGB augmentation.",
    )
    num_views: int = INT_FIELD(
        value=5,
        default_value=5,
        valid_min=2,
        display_name="number of views",
        description="Number of views per multi-view sample.",
        popular="yes",
    )
    randomize_view_order: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="randomize view order",
        description="""
        Training only. Select either central anchor as VGGT reference view 0,
        then shuffle the remaining views. Validation and test ignore this flag
        and use stable index-seeded sampling.""",
    )
    num_workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=0,
        display_name="num workers",
        description="Number of dataloader workers for every split.",
    )
    pairs_per_scene: int = INT_FIELD(
        value=50,
        default_value=50,
        valid_min=1,
        display_name="pairs per scene",
        description="Samples drawn per scene from the primary root.",
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        display_name="batch size",
        description="Number of multi-view samples per GPU.",
        popular="yes",
    )
    resolution: List[List[int]] = LIST_FIELD(
        arrList=[[518, 518], [518, 378], [518, 322]],
        default_value=[[518, 518], [518, 378], [518, 322]],
        display_name="resolution buckets",
        description="""
        Aspect-ratio buckets as [H, W] pairs. One bucket is selected per sample
        from the anchor view's aspect ratio; a single entry reproduces
        fixed-resolution behaviour.""",
    )


@dataclass
class ReasoningDatasetConfig:
    """Dataset config for the reasoning variant (JSONL reasoning manifests)."""

    train_manifest: Optional[str] = STR_FIELD(
        value=None,
        display_name="train manifest",
        description="""
        Path to the training JSONL manifest. Each line holds the view image
        paths, the RGB-encoded panoptic PNGs, the instruction/answer pair, and
        the target instance id.""",
    )
    val_manifest: Optional[str] = STR_FIELD(
        value=None,
        display_name="val manifest",
        description="Path to the validation JSONL manifest. Null disables validation.",
    )
    require_seg_token: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="require SEG token",
        description="""
        Drop training records whose answer has a target instance but no
        :code:`[SEG]` token, which would otherwise yield no mask supervision.""",
    )
    resolution: List[int] = LIST_FIELD(
        arrList=[518, 518],
        default_value=[518, 518],
        display_name="resolution",
        description="Square letterbox size [H, W] every view is resized to, matching VGGT.",
    )
    num_views: int = INT_FIELD(
        value=5,
        default_value=5,
        valid_min=1,
        display_name="number of views",
        description="Number of views per sample. 1 reproduces the single-view setting.",
        popular="yes",
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        display_name="batch size",
        description="Number of multi-view samples per GPU.",
        popular="yes",
    )
    num_workers: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=0,
        display_name="num workers",
        description="Number of dataloader workers for every split.",
    )
    depth_scale: float = FLOAT_FIELD(
        value=1000.0,
        default_value=1000.0,
        math_cond="> 0.0",
        display_name="depth scale",
        description="""
        Stored depth units per metre (ScanNet++: 1000). Per-record values in
        the manifest override this fallback.""",
    )


@dataclass
class NVPanoptix3Dv2DatasetConfig:
    """Data config.

    Each variant owns a self-contained sub-block; :code:`model.model_type`
    selects which one the data module reads.
    """

    panoptic: PanopticDatasetConfig = DATACLASS_FIELD(
        PanopticDatasetConfig(),
        display_name="panoptic dataset",
        description="Dataset for the panoptic variant. Read when model_type is panoptic.",
    )
    reasoning: ReasoningDatasetConfig = DATACLASS_FIELD(
        ReasoningDatasetConfig(),
        display_name="reasoning dataset",
        description="Dataset for the reasoning variant. Read when model_type is reasoning.",
    )
