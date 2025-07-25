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

"""DINO convert onnx model to TRT engine."""

import logging
import os

from nvidia_tao_core.config.dino.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.initialize_experiments import initialize_gen_trt_engine_experiment
from nvidia_tao_deploy.cv.deformable_detr.engine_builder import DDETRDetEngineBuilder
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.utils.decoding import decode_model


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="gen_trt_engine", schema=ExperimentConfig
)
@monitor_status(name='dino', mode='gen_trt_engine')
def main(cfg: ExperimentConfig) -> None:
    """Convert encrypted uff or onnx model to TRT engine."""
    # decrypt etlt
    tmp_onnx_file, file_format = decode_model(cfg.gen_trt_engine.onnx_file, cfg.encryption_key)

    engine_builder_kwargs, create_engine_kwargs = initialize_gen_trt_engine_experiment(cfg)

    workspace_size = cfg.gen_trt_engine.tensorrt.workspace_size

    # INT8 related configs
    img_std = cfg.dataset.augmentation.input_std

    # For ViT-L, override workspace to be larger.
    if cfg.model.backbone == "vit_large_dinov2" and workspace_size < 24080:
        logger.warning("Overriding workspace_size from {} to 20480 due to ViT's model size".format(workspace_size))
        workspace_size = 20480

    builder = DDETRDetEngineBuilder(**engine_builder_kwargs,
                                    workspace=workspace_size // 1024,  # DINO config is not in GB
                                    img_std=img_std)

    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(**create_engine_kwargs)


if __name__ == '__main__':
    main()
