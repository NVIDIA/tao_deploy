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
class LoaderConfig:
    """Dataloader config."""

    shuffle_buffer: int = 10000
    cycle_length: int = 32
    block_length: int = 16
    shuffle_file: bool = True
    prefetch_size: int = 2


@dataclass
class LRConfig:
    """LR config."""

    name: str = 'cosine'  # soft_anneal
    warmup_epoch: int = 5
    warmup_init: float = 0.0001
    learning_rate: float = 0.2
    annealing_epoch: int = 10


@dataclass
class OptConfig:
    """Optimizer config."""

    name: str = 'sgd'
    momentum: float = 0.9


@dataclass
class TrainConfig:
    """Train config."""

    init_epoch: int = 0
    optimizer: OptConfig = OptConfig()
    lr_schedule: LRConfig = LRConfig()
    num_examples_per_epoch: int = 120000
    batch_size: int = 8
    num_epochs: int = 300
    checkpoint: str = ""
    random_seed: int = 42
    l1_weight_decay: float = 0.0
    l2_weight_decay: float = 0.00004
    amp: bool = False
    pruned_model_path: str = ''
    moving_average_decay: float = 0.9999
    clip_gradients_norm: float = 10.0
    skip_checkpoint_variables: str = ''
    checkpoint_interval: int = 10
    image_preview: bool = True
    qat: bool = False
    label_smoothing: float = 0.0
    box_loss_weight: float = 50.0
    iou_loss_type: str = ""
    iou_loss_weight: float = 1.0

    wandb: WandBConfig = WandBConfig(
        name="efficientdet",
        tags=["efficientdet", "training", "tao-toolkit"]
    )
    clearml: ClearMLConfig = ClearMLConfig(
        task="efficientdet_train",
        tags=["efficientdet", "training", "tao-toolkit"]
    )
    results_dir: Optional[str] = None


@dataclass
class ModelConfig:
    """Model config."""

    name: str = 'efficientdet-d0'
    aspect_ratios: str = '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
    anchor_scale: int = 4
    min_level: int = 3
    max_level: int = 7
    num_scales: int = 3
    freeze_bn: bool = False
    freeze_blocks: List[int] = field(default_factory=lambda: [])
    input_width: int = 512
    input_height: int = 512


@dataclass
class AugmentationConfig:
    """Augmentation config."""

    rand_hflip: bool = True
    random_crop_min_scale: float = 0.1
    random_crop_max_scale: float = 2
    auto_color_distortion: bool = False
    auto_translate_xy: bool = False


@dataclass
class DataConfig:
    """Data config."""

    train_tfrecords: List[str] = field(default_factory=lambda: [])
    train_dirs: List[str] = field(default_factory=lambda: [])  # TODO
    val_tfrecords: List[str] = field(default_factory=lambda: [])
    val_dirs: List[str] = field(default_factory=lambda: [])  # TODO
    val_json_file: str = ""
    num_classes: int = 91
    max_instances_per_image: int = 200
    skip_crowd_during_training: bool = True
    use_fake_data: bool = False
    loader: LoaderConfig = LoaderConfig()
    augmentation: AugmentationConfig = AugmentationConfig()


@dataclass
class EvalConfig:
    """Eval config."""

    batch_size: int = 8
    num_samples: int = 5000
    max_detections_per_image: int = 100
    label_map: str = ''
    max_nms_inputs: int = 5000
    checkpoint: str = ''
    trt_engine: Optional[str] = None
    start_eval_epoch: int = 1
    sigma: float = 0.5
    results_dir: Optional[str] = None


@dataclass
class ExportConfig:
    """Export config."""

    batch_size: int = 8
    dynamic_batch_size: bool = True
    min_score_thresh: float = 0.01
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
class InferenceConfig:
    """Inference config."""

    checkpoint: str = MISSING
    trt_engine: Optional[str] = None
    image_dir: str = MISSING
    results_dir: Optional[str] = None
    dump_label: bool = False
    batch_size: int = 1
    min_score_thresh: float = 0.3
    label_map: str = ''
    max_boxes_to_draw: int = 100


@dataclass
class PruneConfig:
    """Pruning config."""

    checkpoint: str = MISSING
    normalizer: str = 'max'
    results_dir: Optional[str] = None
    equalization_criterion: str = 'union'
    granularity: int = 8
    threshold: float = MISSING
    min_num_filters: int = 16
    excluded_layers: List[str] = field(default_factory=lambda: [])


@dataclass
class DatasetConvertConfig:
    """Dataset Convert config."""

    image_dir: str = MISSING
    annotations_file: str = MISSING
    results_dir: str = MISSING
    tag: str = ''
    num_shards: int = 256
    include_masks: bool = False


@dataclass
class ExperimentConfig:
    """Experiment config."""

    train: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()
    evaluate: EvalConfig = EvalConfig()
    dataset: DataConfig = DataConfig()
    export: ExportConfig = ExportConfig()
    inference: InferenceConfig = InferenceConfig()
    prune: PruneConfig = PruneConfig()
    dataset_convert: DatasetConvertConfig = DatasetConvertConfig()
    gen_trt_engine: GenTrtEngineConfig = GenTrtEngineConfig()
    encryption_key: Optional[str] = None
    data_format: str = 'channels_last'
    results_dir: str = MISSING
