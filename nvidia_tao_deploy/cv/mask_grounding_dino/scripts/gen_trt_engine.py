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

"""Mask Grounding DINO convert onnx model to TRT engine."""

import logging
import os
import tempfile

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner

from nvidia_tao_deploy.cv.grounding_dino.engine_builder import GDINODetEngineBuilder
from nvidia_tao_deploy.cv.mask_grounding_dino.hydra_config.default_config import ExperimentConfig


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="gen_trt_engine", schema=ExperimentConfig
)
@monitor_status(name='mask_grounding_dino', mode='gen_trt_engine')
def main(cfg: ExperimentConfig) -> None:
    """Convert onnx model to TRT engine."""
    if cfg.gen_trt_engine.results_dir:
        results_dir = cfg.gen_trt_engine.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "gen_trt_engine")
    os.makedirs(results_dir, exist_ok=True)

    # decrypt etlt
    tmp_onnx_file = cfg.gen_trt_engine.onnx_file

    engine_file = cfg.gen_trt_engine.trt_engine

    data_type = cfg.gen_trt_engine.tensorrt.data_type
    workspace_size = cfg.gen_trt_engine.tensorrt.workspace_size
    min_batch_size = cfg.gen_trt_engine.tensorrt.min_batch_size
    opt_batch_size = cfg.gen_trt_engine.tensorrt.opt_batch_size
    max_batch_size = cfg.gen_trt_engine.tensorrt.max_batch_size
    batch_size = cfg.gen_trt_engine.batch_size
    num_channels = cfg.gen_trt_engine.input_channel
    input_width = cfg.gen_trt_engine.input_width
    input_height = cfg.gen_trt_engine.input_height
    max_text_len = cfg.model.max_text_len
    img_std = cfg.dataset.augmentation.input_std

    if batch_size is None or batch_size == -1:
        input_batch_size = 1
        is_dynamic = True
    else:
        input_batch_size = batch_size
        is_dynamic = False

    if engine_file is not None:
        if engine_file is None:
            engine_handle, temp_engine_path = tempfile.mkstemp()
            os.close(engine_handle)
            output_engine_path = temp_engine_path
        else:
            output_engine_path = engine_file

        builder = GDINODetEngineBuilder(workspace=workspace_size // 1024,  # DINO config is not in GB
                                        input_dims=(input_batch_size, num_channels, input_height, input_width),
                                        is_dynamic=is_dynamic,
                                        max_text_len=max_text_len,
                                        min_batch_size=min_batch_size,
                                        opt_batch_size=opt_batch_size,
                                        max_batch_size=max_batch_size,
                                        img_std=img_std)

        builder.create_network(tmp_onnx_file, "onnx")
        builder.create_engine(
            output_engine_path,
            data_type
        )

    logging.info("The TensorRT engine was saved at: %s.", output_engine_path)


if __name__ == '__main__':
    main()
