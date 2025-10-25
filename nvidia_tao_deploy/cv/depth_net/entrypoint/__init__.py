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

"""Entrypoint module for DepthNet operations.

This module provides the main command-line interface for DepthNet operations in the
TAO Deploy framework. It serves as the primary entry point for all DepthNet-related
tasks including model conversion, inference, and evaluation.

The module exports the main entrypoint function that handles:
- Command-line argument parsing
- Task routing to appropriate submodules
- Integration with the TAO framework's hydra-based configuration system

Supported Operations:
- gen_trt_engine: Convert ONNX models to TensorRT engines
- inference: Perform high-performance inference on images
- evaluate: Evaluate model performance with comprehensive metrics

Usage:
    The module is typically used through the command-line interface:
    ```bash
    depth_net <operation> --config-file config.yaml
    ```

Dependencies:
    - argparse for command-line parsing
    - nvidia_tao_deploy.cv.common.entrypoint for hydra integration
    - nvidia_tao_deploy.cv.depth_net.scripts for operation implementations
"""
