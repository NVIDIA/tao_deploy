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
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_deploy.config.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)
from nvidia_tao_deploy.cv.common.config.mlops import ClearMLConfig, WandBConfig


@dataclass
class RegConfig:
    """Regularizer config."""

    type: str = STR_FIELD(value="L2", automl_enabled="TRUE")
    scope: List[str] = LIST_FIELD(["conv2d", "dense"])
    weight_decay: float = FLOAT_FIELD(value=0.000015, valid_min=0, valid_max="inf")


@dataclass
class BNConfig:
    """Batchnorm config."""

    momentum: float = FLOAT_FIELD(value=0.9, valid_min="1.00E-10", valid_max="0.99")
    epsilon: float = FLOAT_FIELD(value=1e-5, valid_min="1.00E-10", valid_max="inf")


@dataclass
class OptimConfig:
    """Optimizer config."""

    optimizer: str = STR_FIELD(value="sgd")
    lr: float = FLOAT_FIELD(
        value=0.05, automl_enabled="TRUE", valid_min=0, valid_max="inf"
    )
    decay: float = FLOAT_FIELD(value=0.0001, valid_min=0, valid_max=1)
    epsilon: float = FLOAT_FIELD(value=0.0001, valid_min=0, valid_max=1)
    rho: float = FLOAT_FIELD(value=0.5, valid_min=0, valid_max=1)
    beta_1: float = FLOAT_FIELD(
        value=0.99, automl_enabled="TRUE", valid_min=0, valid_max=1
    )
    beta_2: float = FLOAT_FIELD(value=0.99, valid_min=0, valid_max=1)
    momentum: float = FLOAT_FIELD(value=0.99, valid_min=0, valid_max=1)
    nesterov: bool = BOOL_FIELD(value=True, automl_enabled="TRUE")


@dataclass
class LRConfig:
    """Learning rate config."""

    scheduler: str = STR_FIELD(value="cosine")  # soft_anneal, step
    learning_rate: float = FLOAT_FIELD(value=0.05)
    soft_start: float = FLOAT_FIELD(value=0.05)
    annealing_points: List[float] = LIST_FIELD([0.33, 0.66, 0.88])
    annealing_divider: float = FLOAT_FIELD(value=10)
    min_lr_ratio: float = FLOAT_FIELD(value=0.00003)
    gamma: float = FLOAT_FIELD(value=0.000015)
    step_size: int = INT_FIELD(value=10)


@dataclass
class TrainConfig:
    """Train config."""

    qat: bool = BOOL_FIELD(value=False)
    resume_training_checkpoint_path: str = STR_FIELD(value="")
    checkpoint: str = STR_FIELD(value="")
    checkpoint_interval: int = INT_FIELD(
        value=1, default_value=1, valid_min=1, valid_max="inf"
    )
    batch_size_per_gpu: int = INT_FIELD(value=64)
    num_epochs: int = INT_FIELD(
        value=100, default_value=80, valid_min=1, valid_max="inf"
    )
    n_workers: int = INT_FIELD(value=10, default_value=10, valid_min=1, valid_max="inf")
    random_seed: int = INT_FIELD(
        value=42, default_value=42, valid_min=1, valid_max="inf"
    )
    label_smoothing: float = FLOAT_FIELD(
        value=0.01, default_value=0.01, valid_min=0, valid_max=1
    )
    reg_config: RegConfig = DATACLASS_FIELD(RegConfig())
    bn_config: BNConfig = DATACLASS_FIELD(BNConfig())
    lr_config: LRConfig = DATACLASS_FIELD(LRConfig())
    optim_config: OptimConfig = DATACLASS_FIELD(OptimConfig())
    wandb: WandBConfig = DATACLASS_FIELD(
        WandBConfig(
            name="classification", tags=["classification", "training", "tao-toolkit"]
        )
    )
    clearml: ClearMLConfig = DATACLASS_FIELD(
        ClearMLConfig(
            task="classification_train",
            tags=["classification", "training", "tao-toolkit"],
        )
    )
    results_dir: Optional[str] = STR_FIELD(value=None)


@dataclass
class AugmentConfig:
    """Augment config."""

    enable_random_crop: bool = BOOL_FIELD(value=True)
    enable_center_crop: bool = BOOL_FIELD(value=True)
    enable_color_augmentation: bool = BOOL_FIELD(value=False)
    disable_horizontal_flip: bool = BOOL_FIELD(value=False)
    mixup_alpha: float = FLOAT_FIELD(value=0, valid_min=0, valid_max=1)


@dataclass
class DataConfig:
    """Data config."""

    train_dataset_path: str = STR_FIELD(value=MISSING)
    val_dataset_path: str = STR_FIELD(value=MISSING)
    preprocess_mode: str = STR_FIELD(value="caffe", valid_options="caffe,torch,tf")
    image_mean: List[float] = LIST_FIELD([103.939, 116.779, 123.68])
    augmentation: AugmentConfig = DATACLASS_FIELD(AugmentConfig())
    num_classes: int = INT_FIELD(value=20, valid_min=2, valid_max="inf")


@dataclass
class ModelConfig:
    """Model config."""

    backbone: str = STR_FIELD(
        value="efficientnet-b0",
        valid_options="efficientnet-b0,efficientnet-b1,efficientnet-b2,efficientnet-b3,efficientnet-b4,efficientnet-b5",
    )
    input_width: int = INT_FIELD(value=256, valid_min=32)
    input_height: int = INT_FIELD(value=256, valid_min=32)
    input_channels: int = INT_FIELD(value=3, valid_options="1,3")
    input_image_depth: int = INT_FIELD(value=8, valid_options="8,16")
    use_batch_norm: bool = BOOL_FIELD(value=True)
    use_bias: bool = BOOL_FIELD(value=False)
    use_pooling: bool = BOOL_FIELD(value=True)
    all_projections: bool = BOOL_FIELD(value=False)
    freeze_bn: bool = BOOL_FIELD(value=False)
    freeze_blocks: List[int] = LIST_FIELD([])
    retain_head: bool = BOOL_FIELD(value=False)
    dropout: float = FLOAT_FIELD(value=0, valid_min=0, valid_max=1)
    resize_interpolation_method: str = STR_FIELD(
        value="bilinear", valid_options="bilinear,bicubic"
    )  # 'bicubic'
    activation_type: Optional[str] = STR_FIELD(
        None, valid_options="None,relu"
    )  # only used in efficientnets
    byom_model: str = STR_FIELD(value="")


@dataclass
class EvalConfig:
    """Eval config."""

    dataset_path: str = STR_FIELD(value=MISSING)
    checkpoint: str = STR_FIELD(value=MISSING)
    trt_engine: Optional[str] = STR_FIELD(value=None)
    batch_size: int = INT_FIELD(value=64, default_value=1, valid_min=1, valid_max="inf")
    n_workers: int = INT_FIELD(value=64, default_value=1, valid_min=1, valid_max="inf")
    top_k: int = INT_FIELD(value=3, default_value=1, valid_min=1, valid_max="inf")
    classmap: str = STR_FIELD(value="")
    results_dir: Optional[str] = STR_FIELD(value=None)


@dataclass
class ExportConfig:
    """Export config."""

    checkpoint: str = STR_FIELD(value=MISSING)
    onnx_file: str = STR_FIELD(value=MISSING)
    results_dir: Optional[str] = STR_FIELD(value=None)


@dataclass
class CalibrationConfig:
    """Calibration config."""

    cal_image_dir: str = STR_FIELD(value="")
    cal_cache_file: str = STR_FIELD(value="")
    cal_batch_size: int = INT_FIELD(value=1, default_value=16)
    cal_batches: int = INT_FIELD(value=1, default_value=10)
    cal_data_file: str = STR_FIELD(value="")


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = STR_FIELD(value="fp32", valid_options="fp32,int8,fp16")
    max_workspace_size: int = INT_FIELD(value=2)  # in Gb
    min_batch_size: int = INT_FIELD(value=1)
    opt_batch_size: int = INT_FIELD(value=1)
    max_batch_size: int = INT_FIELD(value=1)
    calibration: CalibrationConfig = DATACLASS_FIELD(CalibrationConfig())


@dataclass
class GenTrtEngineConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = STR_FIELD(value=None)
    onnx_file: str = STR_FIELD(value=MISSING)
    trt_engine: Optional[str] = STR_FIELD(None)
    tensorrt: TrtConfig = DATACLASS_FIELD(TrtConfig())


@dataclass
class InferConfig:
    """Inference config."""

    checkpoint: str = STR_FIELD(value=MISSING)
    trt_engine: Optional[str] = STR_FIELD(None)
    image_dir: str = STR_FIELD(value=MISSING)
    classmap: str = STR_FIELD(value=MISSING)
    results_dir: Optional[str] = STR_FIELD(value=None)


@dataclass
class PruneConfig:
    """Pruning config."""

    checkpoint: str = STR_FIELD(value=MISSING)
    byom_model_path: Optional[str] = STR_FIELD(None)
    normalizer: str = STR_FIELD(value="max", valid_options="max,L2")
    results_dir: Optional[str] = STR_FIELD(value=None)
    equalization_criterion: str = STR_FIELD(
        value="union",
        required="no",
        valid_options="union,intersection,arithmetic_mean,geometric_mean",
    )
    granularity: int = INT_FIELD(value=8, required="no", valid_min=8)
    threshold: float = FLOAT_FIELD(
        value=0.1, default_value=0.1, required="no", valid_min=0, valid_max="inf"
    )
    min_num_filters: int = INT_FIELD(
        value=16, default_value=16, valid_min=8, valid_max="inf"
    )
    excluded_layers: List[str] = LIST_FIELD([])


@dataclass
class MPIConfig:
    """MPI configuration"""

    nccl_ib_hca: str = STR_FIELD(value="mlx5_4,mlx5_6,mlx5_8,mlx5_10")
    nccl_socket_ifname: str = STR_FIELD(value="^lo,docker")


@dataclass
class ExperimentConfig:
    """Experiment config."""

    num_gpus: int = INT_FIELD(value=1)
    gpu_ids: List[int] = LIST_FIELD(arrList=[0])
    results_dir: str = STR_FIELD(value="/results")
    encryption_key: Optional[str] = STR_FIELD(None)
    data_format: str = STR_FIELD(
        value="channels_first", valid_options="channels_first,channels_last"
    )
    train: TrainConfig = DATACLASS_FIELD(TrainConfig())
    dataset: DataConfig = DATACLASS_FIELD(DataConfig())
    model: ModelConfig = DATACLASS_FIELD(ModelConfig())
    evaluate: EvalConfig = DATACLASS_FIELD(EvalConfig())
    export: ExportConfig = DATACLASS_FIELD(ExportConfig())
    inference: InferConfig = DATACLASS_FIELD(InferConfig())
    prune: PruneConfig = DATACLASS_FIELD(PruneConfig())
    gen_trt_engine: GenTrtEngineConfig = DATACLASS_FIELD(GenTrtEngineConfig())
    mpi_args: MPIConfig = DATACLASS_FIELD(MPIConfig())
    cuda_blocking: bool = BOOL_FIELD(
        value=False,
        description="Debug flag to add CUDA_LAUNCH_BLOCKING=1 to the command calls.",
    )
    multi_node: bool = BOOL_FIELD(
        value=False, description="Flag to enable to run multi-node training."
    )
    num_processes: int = INT_FIELD(
        value=-1,
        description="The number of horovod child processes to be spawned. Default is -1 (equal to num_gpus).",
    )
