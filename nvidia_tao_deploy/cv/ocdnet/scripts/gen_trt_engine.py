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

"""OCDNet convert model to TRT engine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from nvidia_tao_deploy.utils.decoding import decode_model
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.ocdnet.config.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.ocdnet.engine_builder import OCDNetEngineBuilder

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="gen_trt_engine", schema=ExperimentConfig
)
@monitor_status(name="ocdnet", mode="gen_trt_engine")
def main(cfg: ExperimentConfig) -> None:
    """Convert the onnx model to TRT engine."""
    if cfg.gen_trt_engine.results_dir is not None:
        results_dir = cfg.gen_trt_engine.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "gen_trt_engine")
    os.makedirs(results_dir, exist_ok=True)

    tmp_onnx_file, file_format = decode_model(cfg['gen_trt_engine']['onnx_file'])

    engine_file = cfg['gen_trt_engine']['trt_engine']

    data_type = cfg['gen_trt_engine']['tensorrt']['data_type']
    workspace_size = cfg['gen_trt_engine']['tensorrt']['workspace_size']
    min_batch_size = cfg['gen_trt_engine']['tensorrt']['min_batch_size']
    opt_batch_size = cfg['gen_trt_engine']['tensorrt']['opt_batch_size']
    max_batch_size = cfg['gen_trt_engine']['tensorrt']['max_batch_size']
    input_height = cfg['gen_trt_engine']['height']
    input_width = cfg['gen_trt_engine']['width']
    img_mode = cfg['gen_trt_engine']['img_mode']

    cal_image_dir = cfg['gen_trt_engine']['tensorrt']['calibration']['cal_image_dir']
    cal_cache_file = cfg['gen_trt_engine']['tensorrt']['calibration']['cal_cache_file']
    cal_batch_size = cfg['gen_trt_engine']['tensorrt']['calibration']['cal_batch_size']
    cal_num_batches = cfg['gen_trt_engine']['tensorrt']['calibration']['cal_num_batches']

    if engine_file:
        if data_type == "int8":
            if not os.path.isdir(cal_image_dir):
                raise FileNotFoundError(
                    f"Calibration image directory {cal_image_dir} not found."
                )
            if len(os.listdir(cal_image_dir)) == 0:
                raise FileNotFoundError(
                    f"Calibration image directory {cal_image_dir} is empty."
                )
            if cal_num_batches <= 0:
                raise ValueError(
                    f"Calibration number of batches {cal_num_batches} is non-positive."
                )
            if cal_batch_size <= 0:
                raise ValueError(
                    f"Calibration batch size {cal_batch_size} is non-positive."
                )
            if len(os.listdir(cal_image_dir)) < cal_num_batches * cal_batch_size:
                raise ValueError(
                    f"Calibration images should be large than {cal_num_batches} * {cal_batch_size}."
                )

        builder = OCDNetEngineBuilder(input_width,
                                      input_height,
                                      img_mode,
                                      workspace=workspace_size,
                                      min_batch_size=min_batch_size,
                                      opt_batch_size=opt_batch_size,
                                      max_batch_size=max_batch_size,
                                      )
        builder.create_network(tmp_onnx_file, file_format)
        builder.create_engine(
            engine_file,
            data_type,
            calib_data_file=None,
            calib_input=cal_image_dir,
            calib_cache=cal_cache_file,
            calib_num_images=cal_batch_size * cal_num_batches,
            calib_batch_size=cal_batch_size)

    logging.info("Generate TensorRT engine and calibration cache file successfully.")


if __name__ == '__main__':
    main()
