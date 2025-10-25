# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""DepthNet TensorRT engine generation script.

This script converts ONNX models to optimized TensorRT engines for DepthNet inference.
It supports model decryption, TensorRT optimization, and engine serialization for
deployment on NVIDIA GPUs.

The engine generation process includes:
- ONNX model decryption and loading
- TensorRT network creation and optimization
- Engine serialization and storage
- Configurable optimization parameters
- Support for different precision modes
"""

import logging
import os

from nvidia_tao_core.config.depth_net.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.initialize_experiments import initialize_gen_trt_engine_experiment
from nvidia_tao_deploy.engine.builder import EngineBuilder
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
@monitor_status(name='depth_net', mode='gen_trt_engine')
def main(cfg: ExperimentConfig) -> None:
    """
    Convert ONNX model to optimized TensorRT engine for DepthNet inference.

    This function performs the complete TensorRT engine generation process for
    DepthNet models. It handles model decryption, TensorRT optimization, and
    engine serialization for deployment on NVIDIA GPUs.

    The process includes:
    - Decryption of encrypted ONNX models using provided encryption keys
    - TensorRT network creation from ONNX model
    - Optimization with configurable precision and workspace settings
    - Engine serialization and storage for deployment
    - Support for different optimization profiles and precision modes

    Args:
        cfg (ExperimentConfig): Configuration object containing all engine generation
            parameters including model paths, TensorRT settings, and optimization
            preferences.

    Raises:
        FileNotFoundError: If the ONNX model file does not exist.
        ValueError: If configuration parameters are invalid.
        RuntimeError: If TensorRT engine generation fails.

    Example:
        The function is typically called through the command line interface:
        ```bash
        python gen_trt_engine.py gen_trt_engine.onnx_file=/path/to/model.onnx
        ```

    Output:
        - Optimized TensorRT engine is saved to the specified output path
        - Engine file can be used for high-performance inference
        - Console output includes optimization progress and completion status
    """
    # decrypt etlt
    tmp_onnx_file, file_format = decode_model(cfg.gen_trt_engine.onnx_file, cfg.encryption_key)

    engine_builder_kwargs, create_engine_kwargs = initialize_gen_trt_engine_experiment(cfg)

    workspace_size = cfg.gen_trt_engine.tensorrt.workspace_size

    builder = EngineBuilder(**engine_builder_kwargs,
                            workspace=workspace_size)
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(**create_engine_kwargs)


if __name__ == '__main__':
    main()
