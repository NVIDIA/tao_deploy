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

"""Mask2former convert onnx model to TRT engine."""

import logging
import os

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.mask2former.engine_builder import Mask2formerEngineBuilder
from nvidia_tao_deploy.cv.mask2former.hydra_config.default_config import ExperimentConfig

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="gen_trt_engine", schema=ExperimentConfig
)
@monitor_status(name='mask2former', mode='gen_trt_engine')
def main(cfg: ExperimentConfig) -> None:
    """Convert onnx model to TRT engine."""
    if cfg.gen_trt_engine.results_dir:
        results_dir = cfg.gen_trt_engine.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "gen_trt_engine")
    os.makedirs(results_dir, exist_ok=True)

    engine_file = cfg.gen_trt_engine.trt_engine
    data_type = cfg.gen_trt_engine.tensorrt.data_type
    assert data_type.lower() in ['fp32', 'fp16'], "Only FP32 and FP16 are supported."
    workspace_size = cfg.gen_trt_engine.tensorrt.workspace_size
    min_batch_size = cfg.gen_trt_engine.tensorrt.min_batch_size
    opt_batch_size = cfg.gen_trt_engine.tensorrt.opt_batch_size
    max_batch_size = cfg.gen_trt_engine.tensorrt.max_batch_size
    batch_size = cfg.gen_trt_engine.batch_size
    num_channels = cfg.gen_trt_engine.input_channel
    input_width = cfg.gen_trt_engine.input_width
    input_height = cfg.gen_trt_engine.input_height

    if batch_size is None or batch_size == -1:
        input_batch_size = 1
        is_dynamic = True
    else:
        input_batch_size = batch_size
        is_dynamic = False

    builder = Mask2formerEngineBuilder(
        workspace=workspace_size // 1024,
        input_dims=(input_batch_size, num_channels, input_height, input_width),
        is_dynamic=is_dynamic,
        min_batch_size=min_batch_size,
        opt_batch_size=opt_batch_size,
        max_batch_size=max_batch_size,
        img_std=None,
        verbose=True)

    builder.create_network(cfg.gen_trt_engine.onnx_file)
    layer_precisions = {
        "/sem_seg_head/predictor/transformer_cross_attention_layers": "fp32",
        "/post_processor": "fp32"
    }
    builder.create_engine(
        engine_file,
        data_type,
        layers_precision=layer_precisions
    )

    print(f"TensorRT engine was saved at {cfg.gen_trt_engine.trt_engine}.")


if __name__ == '__main__':
    main()
