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

"""Default config file"""

from typing import List, Optional
from dataclasses import dataclass, field
from omegaconf import MISSING

from nvidia_tao_deploy.cv.common.config.mlops import ClearMLConfig, WandBConfig


@dataclass
class RegConfig:
    """Regularizer config."""

    type: str = 'L2'
    scope: List[str] = field(default_factory=lambda: ['conv2d', 'dense'])
    weight_decay: float = 0.000015


@dataclass
class BNConfig:
    """Batchnorm config."""

    momentum: float = 0.9
    epsilon: float = 1e-5


@dataclass
class OptimConfig:
    """Optimizer config."""

    optimizer: str = 'sgd'
    lr: float = 0.05
    decay: float = 0.0001
    epsilon: float = 0.0001
    rho: float = 0.5
    beta_1: float = 0.99
    beta_2: float = 0.99
    momentum: float = 0.99
    nesterov: bool = True


@dataclass
class LRConfig:
    """Learning rate config."""

    scheduler: str = 'cosine'  # soft_anneal, step
    learning_rate: float = 0.05
    soft_start: float = 0.05
    annealing_points: List[float] = field(default_factory=lambda: [0.33, 0.66, 0.88])
    annealing_divider: float = 10
    min_lr_ratio: float = 0.00003
    gamma: float = 0.000015
    step_size: int = 10


@dataclass
class TrainConfig:
    """Train config."""

    qat: bool = False
    checkpoint: str = ''
    checkpoint_interval: int = 1
    batch_size_per_gpu: int = 64
    num_epochs: int = 100
    n_workers: int = 10
    random_seed: int = 42
    label_smoothing: float = 0.01
    reg_config: RegConfig = RegConfig()
    bn_config: BNConfig = BNConfig()
    lr_config: LRConfig = LRConfig()
    optim_config: OptimConfig = OptimConfig()
    wandb: WandBConfig = WandBConfig(
        name="classification",
        tags=["classification", "training", "tao-toolkit"]
    )
    clearml: ClearMLConfig = ClearMLConfig(
        task="classification_train",
        tags=["classification", "training", "tao-toolkit"]
    )
    results_dir: Optional[str] = None


@dataclass
class AugmentConfig:
    """Augment config."""

    enable_random_crop: bool = True
    enable_center_crop: bool = True
    enable_color_augmentation: bool = False
    disable_horizontal_flip: bool = False
    mixup_alpha: float = 0


@dataclass
class DataConfig:
    """Data config."""

    train_dataset_path: str = MISSING
    val_dataset_path: str = MISSING
    preprocess_mode: str = 'caffe'
    image_mean: List[float] = field(default_factory=lambda: [103.939, 116.779, 123.68])
    augmentation: AugmentConfig = AugmentConfig()
    num_classes: int = MISSING


@dataclass
class ModelConfig:
    """Model config."""

    backbone: str = 'resnet_18'
    input_width: int = 224
    input_height: int = 224
    input_channels: int = 3
    input_image_depth: int = 8
    use_batch_norm: bool = True
    use_bias: bool = False
    use_pooling: bool = True
    all_projections: bool = False
    freeze_bn: bool = False
    freeze_blocks: List[int] = field(default_factory=lambda: [])
    retain_head: bool = False
    dropout: float = 0.0
    resize_interpolation_method: str = 'bilinear'  # 'bicubic'
    activation_type: Optional[str] = None  # only used in efficientnets
    byom_model: str = ''


@dataclass
class EvalConfig:
    """Eval config."""

    dataset_path: str = MISSING
    checkpoint: str = MISSING
    trt_engine: Optional[str] = None
    batch_size: int = 64
    n_workers: int = 64
    top_k: int = 3
    classmap: str = ""
    results_dir: Optional[str] = None


@dataclass
class ExportConfig:
    """Export config."""

    checkpoint: str = MISSING
    onnx_file: str = MISSING
    results_dir: Optional[str] = None


@dataclass
class CalibrationConfig:
    """Calibration config."""

    cal_image_dir: str = ""
    cal_cache_file: str = ""
    cal_batch_size: int = 1
    cal_batches: int = 1
    cal_data_file: str = ""


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = "fp32"
    max_workspace_size: int = 2  # in Gb
    min_batch_size: int = 1
    opt_batch_size: int = 1
    max_batch_size: int = 1
    calibration: CalibrationConfig = CalibrationConfig()


@dataclass
class GenTrtEngineConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = None
    onnx_file: str = MISSING
    trt_engine: Optional[str] = None
    tensorrt: TrtConfig = TrtConfig()


@dataclass
class InferConfig:
    """Inference config."""

    checkpoint: str = MISSING
    trt_engine: Optional[str] = None
    image_dir: str = MISSING
    classmap: str = MISSING
    results_dir: Optional[str] = None


@dataclass
class PruneConfig:
    """Pruning config."""

    checkpoint: str = MISSING
    byom_model_path: Optional[str] = None
    normalizer: str = 'max'
    results_dir: Optional[str] = None
    equalization_criterion: str = 'union'
    granularity: int = 8
    threshold: float = MISSING
    min_num_filters: int = 16
    excluded_layers: List[str] = field(default_factory=lambda: [])


@dataclass
class ExperimentConfig:
    """Experiment config."""

    train: TrainConfig = TrainConfig()
    dataset: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    evaluate: EvalConfig = EvalConfig()
    export: ExportConfig = ExportConfig()
    inference: InferConfig = InferConfig()
    prune: PruneConfig = PruneConfig()
    gen_trt_engine: GenTrtEngineConfig = GenTrtEngineConfig()
    results_dir: str = MISSING
    encryption_key: Optional[str] = None
    data_format: str = 'channels_first'
