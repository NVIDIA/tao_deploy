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
import tempfile

from nvidia_tao_deploy.cv.deformable_detr.engine_builder import DDETRDetEngineBuilder
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.dino.hydra_config.default_config import ExperimentConfig
from nvidia_tao_deploy.utils.decoding import decode_model
from nvidia_tao_deploy.engine.builder import NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH


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
    if cfg.gen_trt_engine.results_dir is not None:
        results_dir = cfg.gen_trt_engine.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "gen_trt_engine")
    os.makedirs(results_dir, exist_ok=True)

    # decrypt etlt
    tmp_onnx_file, file_format = decode_model(cfg.gen_trt_engine.onnx_file, cfg.encryption_key)

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

    # INT8 related configs
    img_std = cfg.dataset.augmentation.input_std
    calib_input = list(cfg.gen_trt_engine.tensorrt.calibration.get('cal_image_dir', []))
    calib_cache = cfg.gen_trt_engine.tensorrt.calibration.get('cal_cache_file', None)

    if batch_size is None or batch_size == -1:
        input_batch_size = 1
        is_dynamic = True
    else:
        input_batch_size = batch_size
        is_dynamic = False

    # TODO: Remove this when we upgrade to DLFW 23.04+
    trt_version_number = NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH
    if data_type.lower() == "fp16" and trt_version_number < 8600:
        logger.warning("[WARNING]: LayerNorm has overflow issue in FP16 upto TensorRT version 8.5 "
                       "which can lead to mAP drop compared to FP32.\n"
                       "[WARNING]: Please re-export ONNX using opset 17 and use TensorRT version 8.6.\n")

    if engine_file is not None:
        if engine_file is None:
            engine_handle, temp_engine_path = tempfile.mkstemp()
            os.close(engine_handle)
            output_engine_path = temp_engine_path
        else:
            output_engine_path = engine_file

        builder = DDETRDetEngineBuilder(workspace=workspace_size // 1024,  # DINO config is not in GB
                                        input_dims=(input_batch_size, num_channels, input_height, input_width),
                                        is_dynamic=is_dynamic,
                                        min_batch_size=min_batch_size,
                                        opt_batch_size=opt_batch_size,
                                        max_batch_size=max_batch_size,
                                        img_std=img_std)

        builder.create_network(tmp_onnx_file, file_format)
        builder.create_engine(
            output_engine_path,
            data_type,
            calib_input=calib_input,
            calib_cache=calib_cache,
            calib_num_images=cfg.gen_trt_engine.tensorrt.calibration.cal_batch_size * cfg.gen_trt_engine.tensorrt.calibration.cal_batches,
            calib_batch_size=cfg.gen_trt_engine.tensorrt.calibration.cal_batch_size
        )

    logging.info("Export finished successfully.")


if __name__ == '__main__':
    main()
