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

"""Classification convert etlt model to TRT engine."""

import logging
import os
import sys
import tempfile

from nvidia_tao_core.config.classification_tf2.default_config import ExperimentConfig

from nvidia_tao_deploy.utils.decoding import decode_model
from nvidia_tao_deploy.cv.classification_tf1.engine_builder import ClassificationEngineBuilder
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment_spec", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for TRT engine generation."""
    # Deprecated: DLFW 25.01 doesn't support tensorflow_quantization
    if sys.version_info >= (3, 12):
        logger.warning("DeprecationWarning: QAT is not supported after DLFW 25.01. Using normal training.")
        cfg.train.qat = False

    run_conversion(cfg=cfg)


@monitor_status(name='classification_tf2', mode='gen_trt_engine')
def run_conversion(cfg: ExperimentConfig) -> None:
    """Classification TRT convert."""
    # decrypt etlt or use onnx
    tmp_onnx_file, file_format = decode_model(cfg.gen_trt_engine.onnx_file, cfg.encryption_key)

    if cfg.gen_trt_engine.trt_engine is not None or cfg.gen_trt_engine.tensorrt.data_type == 'int8':
        if cfg.gen_trt_engine.trt_engine is None:
            engine_handle, temp_engine_path = tempfile.mkstemp()
            os.close(engine_handle)
            output_engine_path = temp_engine_path
        else:
            output_engine_path = cfg.gen_trt_engine.trt_engine

        builder = ClassificationEngineBuilder(verbose=True,
                                              workspace=cfg.gen_trt_engine.tensorrt.max_workspace_size,
                                              min_batch_size=cfg.gen_trt_engine.tensorrt.min_batch_size,
                                              opt_batch_size=cfg.gen_trt_engine.tensorrt.opt_batch_size,
                                              max_batch_size=cfg.gen_trt_engine.tensorrt.max_batch_size,
                                              is_qat=cfg.train.qat,
                                              data_format=cfg.data_format,  # channels_first
                                              preprocess_mode=cfg.dataset.preprocess_mode)
        builder.create_network(tmp_onnx_file, file_format)
        builder.create_engine(
            output_engine_path,
            cfg.gen_trt_engine.tensorrt.data_type,
            calib_data_file=cfg.gen_trt_engine.tensorrt.calibration.cal_data_file,
            calib_input=cfg.gen_trt_engine.tensorrt.calibration.cal_image_dir,
            calib_cache=cfg.gen_trt_engine.tensorrt.calibration.cal_cache_file,
            calib_num_images=cfg.gen_trt_engine.tensorrt.calibration.cal_batch_size * cfg.gen_trt_engine.tensorrt.calibration.cal_batches,
            calib_batch_size=cfg.gen_trt_engine.tensorrt.calibration.cal_batch_size,
            tf2=True)


if __name__ == '__main__':
    main()
