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

"""TAO Deploy DepthNet scripts module.

This module contains the main operational scripts for DepthNet deployment and evaluation.
It provides command-line interfaces for model conversion, inference, and evaluation
tasks using TensorRT optimization.

Available Scripts:
- gen_trt_engine.py: Convert ONNX models to TensorRT engines
- inference.py: Perform high-performance inference on images
- evaluate.py: Evaluate model performance with comprehensive metrics

Each script supports:
- Hydra-based configuration management
- TensorRT optimization for NVIDIA GPUs
- Batch processing capabilities
- Comprehensive logging and monitoring
- Integration with TAO framework

Usage:
    Scripts can be executed directly or through the main depth_net entrypoint:
    ```bash
    # Direct execution
    python scripts/inference.py --config-file config.yaml

    # Through entrypoint
    depth_net inference --config-file config.yaml
    ```
"""
