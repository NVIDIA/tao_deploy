# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Inference script for PointPillars."""
import os
from pathlib import Path

from nvidia_tao_core.config.pointpillars.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.utils.path_utils import expand_path
from nvidia_tao_deploy.cv.pointpillars.dataloader.pc_dataset import build_dataloader
from nvidia_tao_deploy.cv.pointpillars.inferencer import PointPillarsInferencer, TrtModelWrapper
from nvidia_tao_deploy.cv.pointpillars.utils import eval_utils


def infer_single_ckpt_trt(
    model, test_loader, cfg,
    infer_output_dir
):
    """Inference with TensorRT engine."""
    return eval_utils.infer_one_epoch_trt(
        cfg, model, test_loader,
        result_dir=infer_output_dir,
        save_to_file=cfg.inference.save_to_file
    )


spec_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "specs")


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=spec_root, config_name="inference", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Main function."""
    if not os.path.exists(cfg.inference.trt_engine):
        raise FileNotFoundError(f"TensorRT engine not found: {cfg.inference.trt_engine}")
    if cfg.inference.results_dir is None:
        raise OSError("Either provide output_dir in config file or provide output_dir as a CLI argument")
    output_dir = Path(expand_path(cfg.inference.results_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    infer_output_dir = output_dir / 'infer'
    infer_output_dir.mkdir(parents=True, exist_ok=True)
    # log to file
    test_loader = build_dataloader(
        dataset_cfg=cfg.dataset,
        class_names=cfg.dataset.class_names,
        logger=None,
        info_path=cfg.dataset.data_info_path
    )
    inferencer = PointPillarsInferencer(
        cfg.inference.trt_engine,
        batch_size=cfg.inference.batch_size
    )
    model_wrapper = TrtModelWrapper(
        cfg,
        inferencer
    )
    infer_single_ckpt_trt(
        model_wrapper, test_loader, cfg,
        infer_output_dir
    )


if __name__ == '__main__':
    main()
