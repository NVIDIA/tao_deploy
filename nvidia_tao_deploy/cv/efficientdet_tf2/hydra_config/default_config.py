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

from nvidia_tao_deploy.cv.common.config.mlops import ClearMLConfig, WandBConfig
from nvidia_tao_deploy.config.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class LoaderConfig:
    """Dataloader config."""

    shuffle_buffer: int = INT_FIELD(value=10000, valid_min=1, valid_max="inf")
    cycle_length: int = INT_FIELD(value=32, valid_min=1, valid_max="inf")
    block_length: int = INT_FIELD(value=16, valid_min=1, valid_max="inf")
    shuffle_file: bool = BOOL_FIELD(value=True)
    prefetch_size: int = INT_FIELD(value=2, valid_min=1, valid_max="inf")


@dataclass
class LRConfig:
    """LR config."""

    name: str = STR_FIELD(
        value="cosine", valid_options="cosine,soft_anneal"
    )  # soft_anneal
    warmup_epoch: int = INT_FIELD(
        value=5, default_value=2, valid_min=0, valid_max="inf"
    )
    warmup_init: float = FLOAT_FIELD(
        value=0.0001, valid_min=0, valid_max="inf", automl_enabled="TRUE"
    )
    learning_rate: float = FLOAT_FIELD(
        value=0.2, valid_min=0, valid_max="inf", automl_enabled="TRUE"
    )
    annealing_epoch: int = INT_FIELD(value=10)


@dataclass
class OptConfig:
    """Optimizer config."""

    name: str = STR_FIELD(value="sgd", valid_options="sgd,adam")
    momentum: float = FLOAT_FIELD(value=0.9, valid_min=0, valid_max=1)


@dataclass
class TrainConfig:
    """Train config."""

    init_epoch: int = INT_FIELD(value=0)
    resume_training_checkpoint_path: str = STR_FIELD(value="")
    optimizer: OptConfig = DATACLASS_FIELD(OptConfig())
    lr_schedule: LRConfig = DATACLASS_FIELD(LRConfig())
    num_examples_per_epoch: int = INT_FIELD(value=120000, valid_min=1, valid_max="inf")
    batch_size: int = INT_FIELD(value=8, valid_min=1, valid_max="inf")
    num_epochs: int = INT_FIELD(value=300, valid_min=1, valid_max="inf")
    checkpoint: str = STR_FIELD(value="", default_value="")
    random_seed: int = INT_FIELD(value=42, valid_min=1, valid_max="inf")
    l1_weight_decay: float = FLOAT_FIELD(value=0.0, valid_min=0, valid_max=1)
    l2_weight_decay: float = FLOAT_FIELD(
        value=0.00004, valid_min=0, valid_max="inf", automl_enabled="TRUE"
    )
    amp: bool = BOOL_FIELD(value=False, default_value="TRUE")
    pruned_model_path: str = STR_FIELD(value="")
    moving_average_decay: float = FLOAT_FIELD(
        value=0.9999, valid_min=0, valid_max=1, automl_enabled="TRUE"
    )
    clip_gradients_norm: float = FLOAT_FIELD(value=10.0, valid_min=0, valid_max="inf")
    skip_checkpoint_variables: str = STR_FIELD(value="-predict*")
    checkpoint_interval: int = INT_FIELD(
        value=10, default_value=5, valid_min=1, valid_max="inf"
    )
    image_preview: bool = BOOL_FIELD(value=True)
    qat: bool = BOOL_FIELD(value=False)
    label_smoothing: float = FLOAT_FIELD(value=0.0, valid_min=0, valid_max=1)
    box_loss_weight: float = FLOAT_FIELD(value=50.0, valid_min=0, valid_max="inf")
    iou_loss_type: str = STR_FIELD(value="")
    iou_loss_weight: float = FLOAT_FIELD(value=1.0, valid_min=0, valid_max="inf")
    wandb: WandBConfig = DATACLASS_FIELD(
        WandBConfig(
            name="efficientdet", tags=["efficientdet", "training", "tao-toolkit"]
        )
    )
    clearml: ClearMLConfig = DATACLASS_FIELD(
        ClearMLConfig(
            task="efficientdet_train", tags=["efficientdet", "training", "tao-toolkit"]
        )
    )
    results_dir: Optional[str] = STR_FIELD(value=None)


@dataclass
class ModelConfig:
    """Model config."""

    name: str = STR_FIELD(
        value="efficientdet-d0",
        valid_options="efficientdet-d0,efficientdet-d1,efficientdet-d2,efficientdet-d3,efficientdet-d4,efficientdet-d5",
    )
    aspect_ratios: str = STR_FIELD(value="[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]")
    anchor_scale: int = INT_FIELD(value=4, valid_min=1, valid_max="inf")
    min_level: int = INT_FIELD(value=3, valid_options="3")
    max_level: int = INT_FIELD(value=7, valid_options="7")
    num_scales: int = INT_FIELD(value=3, valid_min=1, valid_max="inf")
    freeze_bn: bool = BOOL_FIELD(value=False)
    freeze_blocks: List[int] = LIST_FIELD([])
    input_width: int = INT_FIELD(value=512)
    input_height: int = INT_FIELD(value=512)


@dataclass
class AugmentationConfig:
    """Augmentation config."""

    rand_hflip: bool = BOOL_FIELD(value=True)
    random_crop_min_scale: float = FLOAT_FIELD(value=0.1, valid_min=0, valid_max=1)
    random_crop_max_scale: float = FLOAT_FIELD(value=2, valid_min=1, valid_max="inf")
    auto_color_distortion: bool = BOOL_FIELD(value=False)
    auto_translate_xy: bool = BOOL_FIELD(value=False)


@dataclass
class DataConfig:
    """Data config."""

    train_tfrecords: List[str] = LIST_FIELD([])
    train_dirs: List[str] = LIST_FIELD([])  # TODO
    val_tfrecords: List[str] = LIST_FIELD([])
    val_dirs: List[str] = LIST_FIELD([])  # TODO
    val_json_file: str = STR_FIELD(value="", default_value="")
    num_classes: int = INT_FIELD(value=91)
    max_instances_per_image: int = INT_FIELD(
        value=200, default_value=100, valid_min=1, valid_max="inf"
    )
    skip_crowd_during_training: bool = BOOL_FIELD(value=True)
    use_fake_data: bool = BOOL_FIELD(value=False)
    loader: LoaderConfig = DATACLASS_FIELD(LoaderConfig())
    augmentation: AugmentationConfig = DATACLASS_FIELD(AugmentationConfig())


@dataclass
class EvalConfig:
    """Eval config."""

    batch_size: int = INT_FIELD(value=8, default_value=16, valid_min=1, valid_max="inf")
    num_samples: int = INT_FIELD(value=5000, valid_min=1, valid_max="inf")
    max_detections_per_image: int = INT_FIELD(value=100, valid_min=1, valid_max="inf")
    label_map: str = STR_FIELD(value="")
    max_nms_inputs: int = INT_FIELD(value=5000, valid_min=1, valid_max="inf")
    checkpoint: str = STR_FIELD(value="")
    trt_engine: Optional[str] = STR_FIELD(value=None)
    start_eval_epoch: int = INT_FIELD(value=1, valid_min=1, valid_max="inf")
    sigma: float = FLOAT_FIELD(value=0.5, valid_min=0, valid_max=1)
    results_dir: Optional[str] = STR_FIELD(value=None)


@dataclass
class ExportConfig:
    """Export config."""

    batch_size: int = INT_FIELD(value=8, valid_min=1, valid_max="inf")
    dynamic_batch_size: bool = BOOL_FIELD(value=True)
    min_score_thresh: float = FLOAT_FIELD(value=0.01, valid_min=0, valid_max=1)
    checkpoint: str = STR_FIELD(value=MISSING)
    onnx_file: str = STR_FIELD(value=MISSING)
    results_dir: Optional[str] = STR_FIELD(value=None)


@dataclass
class CalibrationConfig:
    """Calibration config."""

    cal_image_dir: str = STR_FIELD(value="")
    cal_cache_file: str = STR_FIELD(value="")
    cal_batch_size: int = INT_FIELD(value=1, valid_min=1, valid_max="inf")
    cal_batches: int = INT_FIELD(value=1, valid_min=1, valid_max="inf")


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = STR_FIELD(value="fp32", valid_options="fp32,int8,fp16")
    max_workspace_size: int = INT_FIELD(value=2, valid_min=1, valid_max="inf")  # in Gb
    min_batch_size: int = INT_FIELD(value=1, valid_min=1, valid_max="inf")
    opt_batch_size: int = INT_FIELD(value=1, valid_min=1, valid_max="inf")
    max_batch_size: int = INT_FIELD(value=1, valid_min=1, valid_max="inf")
    calibration: CalibrationConfig = DATACLASS_FIELD(CalibrationConfig())


@dataclass
class GenTrtEngineConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = STR_FIELD(value=None)
    onnx_file: str = STR_FIELD(value=MISSING)
    trt_engine: Optional[str] = STR_FIELD(value=None)
    tensorrt: TrtConfig = DATACLASS_FIELD(TrtConfig())


@dataclass
class InferenceConfig:
    """Inference config."""

    checkpoint: str = STR_FIELD(value=MISSING)
    trt_engine: Optional[str] = STR_FIELD(value=None)
    image_dir: str = STR_FIELD(value=MISSING)
    results_dir: Optional[str] = STR_FIELD(value=None)
    dump_label: bool = BOOL_FIELD(value=False)
    batch_size: int = INT_FIELD(value=1, valid_min=1, valid_max="inf")
    min_score_thresh: float = FLOAT_FIELD(value=0.3, valid_min=0, valid_max=1)
    label_map: str = STR_FIELD(value="")
    max_boxes_to_draw: int = INT_FIELD(value=100, valid_min=1, valid_max="inf")


@dataclass
class PruneConfig:
    """Pruning config."""

    checkpoint: str = STR_FIELD(value=MISSING)
    normalizer: str = STR_FIELD(value="max", valid_options="max,L2")
    results_dir: Optional[str] = STR_FIELD(None)
    equalization_criterion: str = STR_FIELD(
        value="union",
        required="no",
        valid_options="union,intersection,arithmetic_mean,geometric_mean",
    )
    granularity: int = INT_FIELD(value=8, required="no", valid_min=8)
    threshold: float = FLOAT_FIELD(
        value=0.2, default_value=0.2, required="no", valid_min=0, valid_max="inf"
    )
    min_num_filters: int = INT_FIELD(
        value=16, default_value=8, valid_min=8, valid_max="inf"
    )
    excluded_layers: List[str] = LIST_FIELD([])


@dataclass
class DatasetConvertConfig:
    """Dataset Convert config."""

    image_dir: str = STR_FIELD(value=MISSING)
    annotations_file: str = STR_FIELD(value=MISSING)
    results_dir: Optional[str] = STR_FIELD(value=None)
    tag: str = STR_FIELD(value="")
    num_shards: int = INT_FIELD(value=256)
    include_masks: bool = BOOL_FIELD(value=False)


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
    encryption_key: Optional[str] = STR_FIELD(value=None)
    train: TrainConfig = DATACLASS_FIELD(TrainConfig())
    model: ModelConfig = DATACLASS_FIELD(ModelConfig())
    evaluate: EvalConfig = DATACLASS_FIELD(EvalConfig())
    dataset: DataConfig = DATACLASS_FIELD(DataConfig())
    export: ExportConfig = DATACLASS_FIELD(ExportConfig())
    inference: InferenceConfig = DATACLASS_FIELD(InferenceConfig())
    prune: PruneConfig = DATACLASS_FIELD(PruneConfig())
    dataset_convert: DatasetConvertConfig = DATACLASS_FIELD(DatasetConvertConfig())
    gen_trt_engine: GenTrtEngineConfig = DATACLASS_FIELD(GenTrtEngineConfig())
    data_format: str = STR_FIELD(value="channels_last")
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
