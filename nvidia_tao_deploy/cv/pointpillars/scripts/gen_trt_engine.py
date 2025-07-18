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

"""Convert PointPillars model to TensorRT engine."""
import logging
import os

from nvidia_tao_core.config.pointpillars.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.pointpillars.engine_builder import (
    PointPillarsEngineBuilder
)
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.utils.decoding import decode_model


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "specs")


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=spec_root, config_name="gen_trt_engine", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Main function."""
    # INT8 is not yet fully supported, raise error if one tries to use it
    if cfg.gen_trt_engine.data_type.lower() not in ["fp16", "fp32"]:
        raise ValueError("PointPillars only support TensorRT engine in FP32/FP16 data type.")
    print('Converting the model to TensorRT engine...')
    if cfg.gen_trt_engine.onnx_file is None:
        raise OSError("Please provide gen_trt_engine.onnx_file in config file")
    if not os.path.isfile(cfg.gen_trt_engine.onnx_file):
        raise FileNotFoundError(f"Input ONNX model {cfg.gen_trt_engine.onnx_file} does not exist")
    # Warn the user if an exported file already exists.
    assert not os.path.exists(cfg.gen_trt_engine.save_engine), "Default engine file {} already "\
        "exists".format(cfg.gen_trt_engine.save_engine)
    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(cfg.gen_trt_engine.save_engine))
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    tmp_onnx_file, file_format = decode_model(cfg.gen_trt_engine.onnx_file)
    # Save TRT engine
    builder = PointPillarsEngineBuilder(
        batch_size=cfg.gen_trt_engine.batch_size
    )
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(
        cfg.gen_trt_engine.save_engine,
        cfg.gen_trt_engine.data_type,
        calib_data_file=None,
        calib_input=None,
        calib_cache=None,
        calib_num_images=None,
        calib_batch_size=None,
        layers_precision=None
    )
    logging.info("Generate TensorRT engine and calibration cache file successfully.")


if __name__ == '__main__':
    main()
