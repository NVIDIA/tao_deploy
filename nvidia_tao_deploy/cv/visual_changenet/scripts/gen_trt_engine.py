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

"""Visual ChangeNet convert onnx model to TRT engine."""

import logging
import os
import tempfile

from nvidia_tao_deploy.cv.visual_changenet.engine_builder import ChangeNetEngineBuilder
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.visual_changenet.hydra_config.default_config import ExperimentConfig

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="gen_trt_engine", schema=ExperimentConfig
)
@monitor_status(name='visual_changenet', mode='gen_trt_engine')
def main(cfg: ExperimentConfig) -> None:
    """Convert encrypted uff or onnx model to TRT engine."""
    if cfg.gen_trt_engine.results_dir:
        results_dir = cfg.gen_trt_engine.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "gen_trt_engine")
    os.makedirs(results_dir, exist_ok=True)

    engine_file = cfg.gen_trt_engine.trt_engine

    data_type = cfg.gen_trt_engine.tensorrt.data_type
    workspace_size = cfg.gen_trt_engine.tensorrt.workspace_size
    min_batch_size = cfg.gen_trt_engine.tensorrt.min_batch_size
    opt_batch_size = cfg.gen_trt_engine.tensorrt.opt_batch_size
    max_batch_size = cfg.gen_trt_engine.tensorrt.max_batch_size
    batch_size = cfg.gen_trt_engine.batch_size

    # For ViT-L, override workspace to be larger. #TODO: Check if needed.
    if cfg.model.backbone == "vit_large_dinov2" and workspace_size < 24080:
        logger.warning("Overriding workspace_size from {} to 20480 due to ViT's model size".format(workspace_size))
        workspace_size = 20480

    if engine_file is None:
        engine_handle, temp_engine_path = tempfile.mkstemp()
        os.close(engine_handle)
        output_engine_path = temp_engine_path
    else:
        output_engine_path = engine_file

    engine_builder_kwargs = {
        "verbose": cfg.gen_trt_engine.verbose,
        "workspace": workspace_size,
        "min_batch_size": min_batch_size,
        "opt_batch_size": opt_batch_size,
        "max_batch_size": max_batch_size,
        "batch_size": batch_size
    }

    builder = ChangeNetEngineBuilder(**engine_builder_kwargs)
    builder.create_network(cfg.gen_trt_engine.onnx_file, 'onnx')
    builder.create_engine(
        output_engine_path,
        data_type,
    )

    print("Export finished successfully.")


if __name__ == '__main__':
    main()
