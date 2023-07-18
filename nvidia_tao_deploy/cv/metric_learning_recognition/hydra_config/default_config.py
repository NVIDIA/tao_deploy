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

"""Default config file."""

from typing import Optional, List, Dict
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class GaussianBlur:
    """Gaussian Blur configuration template."""

    enabled: bool = True
    kernel: List[int] = field(default_factory=lambda: [15, 15])
    sigma: List[float] = field(default_factory=lambda: [0.3, 0.7])


@dataclass
class ColorAugmentation:
    """Color Augmentation configuration template."""

    enabled: bool = True
    brightness: float = 0.5
    contrast: float = 0.3
    saturation: float = 0.1
    hue: float = 0.1


@dataclass
class DatasetConfig:
    """Metric Learning Recognition Dataset configuration template."""

    train_dataset: Optional[str] = None
    val_dataset: Optional[Dict[str, str]] = None
    workers: int = 8
    class_map: Optional[str] = None  # TODO: add class_map support
    pixel_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    pixel_std: List[float] = field(default_factory=lambda: [0.226, 0.226, 0.226])
    prob: float = 0.5
    re_prob: float = 0.5
    gaussian_blur: GaussianBlur = GaussianBlur()
    color_augmentation: ColorAugmentation = ColorAugmentation()
    random_rotation: bool = False
    num_instance: int = 4


@dataclass
class ModelConfig:
    """Metric Learning Recognition model configuration for training, testing & validation."""

    backbone: str = "resnet_50"
    pretrain_choice: str = "imagenet"
    pretrained_model_path: Optional[str] = None
    input_channels: int = 3
    input_width: int = 224
    input_height: int = 224
    feat_dim: int = 256


@dataclass
class EvalConfig:
    """Evaluation experiment configuration template."""

    trt_engine: Optional[str] = None
    checkpoint: Optional[str] = None
    gpu_id: int = 0
    batch_size: int = 64
    topk: int = 1
    report_accuracy_per_class: bool = True
    results_dir: Optional[str] = None


@dataclass
class InferenceConfig:
    """Inference experiment configuration template."""

    trt_engine: Optional[str] = None
    checkpoint: Optional[str] = None
    input_path: str = MISSING  # a image file or a folder
    inference_input_type: str = "image_folder"  # possible values are "image_folder" and "classification_foler"
    gpu_id: int = 0
    topk: int = 1
    batch_size: int = 64
    results_dir: Optional[str] = None


@dataclass
class CalibrationConfig:
    """Calibration config."""

    cal_cache_file: Optional[str] = None
    cal_batch_size: int = 1
    cal_batches: int = 1
    cal_image_dir: Optional[List[str]] = field(default_factory=lambda: [])


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = "FP32"
    workspace_size: int = 1024
    min_batch_size: int = 1
    opt_batch_size: int = 1
    max_batch_size: int = 1
    calibration: CalibrationConfig = CalibrationConfig()


@dataclass
class TrtEngineConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = None
    gpu_id: int = 0
    onnx_file: str = MISSING
    trt_engine: Optional[str] = None
    batch_size: int = -1
    verbose: bool = False
    tensorrt: TrtConfig = TrtConfig()


@dataclass
class ExperimentConfig:
    """Experiment config."""

    model: ModelConfig = ModelConfig()
    evaluate: EvalConfig = EvalConfig()
    dataset: DatasetConfig = DatasetConfig()
    inference: InferenceConfig = InferenceConfig()
    gen_trt_engine: TrtEngineConfig = TrtEngineConfig()
    results_dir: Optional[str] = None
