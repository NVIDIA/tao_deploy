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

"""EfficientDet convert etlt model to TRT engine."""

import logging
import os
import tempfile

from nvidia_tao_deploy.cv.efficientdet_tf2.engine_builder import EfficientDetEngineBuilder
from nvidia_tao_deploy.cv.efficientdet_tf2.hydra_config.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.common.utils import update_results_dir
from nvidia_tao_deploy.utils.decoding import decode_model

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment_spec", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for TRT engine conversion."""
    cfg = update_results_dir(cfg, 'gen_trt_engine')
    run_conversion(cfg=cfg)


@monitor_status(name='efficientdet_tf2', mode='gen_trt_engine')
def run_conversion(cfg: ExperimentConfig) -> None:
    """EfficientDet TRT convert."""
    # decrypt etlt or use onnx
    tmp_onnx_file, file_format = decode_model(cfg.gen_trt_engine.onnx_file, cfg.encryption_key)

    if cfg.gen_trt_engine.trt_engine is not None or cfg.gen_trt_engine.tensorrt.data_type == 'int8':
        if cfg.gen_trt_engine.trt_engine is None:
            engine_handle, temp_engine_path = tempfile.mkstemp()
            os.close(engine_handle)
            output_engine_path = temp_engine_path
        else:
            output_engine_path = cfg.gen_trt_engine.trt_engine

        builder = EfficientDetEngineBuilder(verbose=True,
                                            workspace=cfg.gen_trt_engine.tensorrt.max_workspace_size,
                                            max_batch_size=cfg.gen_trt_engine.tensorrt.max_batch_size,
                                            is_qat=cfg.train.qat)
        builder.create_network(tmp_onnx_file,
                               dynamic_batch_size=(cfg.gen_trt_engine.tensorrt.min_batch_size,
                                                   cfg.gen_trt_engine.tensorrt.opt_batch_size,
                                                   cfg.gen_trt_engine.tensorrt.max_batch_size),
                               file_format=file_format)
        builder.create_engine(
            output_engine_path,
            cfg.gen_trt_engine.tensorrt.data_type,
            calib_input=cfg.gen_trt_engine.tensorrt.calibration.cal_image_dir,
            calib_cache=cfg.gen_trt_engine.tensorrt.calibration.cal_cache_file,
            calib_num_images=cfg.gen_trt_engine.tensorrt.calibration.cal_batch_size * cfg.gen_trt_engine.tensorrt.calibration.cal_batches,
            calib_batch_size=cfg.gen_trt_engine.tensorrt.calibration.cal_batch_size
        )

    logging.info("Export finished successfully.")


if __name__ == '__main__':
    main()
