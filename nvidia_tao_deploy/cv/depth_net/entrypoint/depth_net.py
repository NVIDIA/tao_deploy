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

"""TAO Deploy command line wrapper for DepthNet operations.

This module provides the main command-line interface for DepthNet operations including
inference, evaluation, and TensorRT engine generation. It serves as the entry point
for all DepthNet-related tasks in the TAO Deploy framework.

Supported operations:
- gen_trt_engine: Convert ONNX models to TensorRT engines
- inference: Perform high-performance inference on images
- evaluate: Evaluate model performance with ground truth data
"""

import argparse
from nvidia_tao_deploy.cv.depth_net import scripts

from nvidia_tao_deploy.cv.common.entrypoint.entrypoint_hydra import get_subtasks, launch, command_line_parser


def get_subtask_list():
    """
    Get the list of available subtasks by inspecting the scripts package.

    This function dynamically discovers all available DepthNet operations by
    examining the scripts module. It returns a list of subtask names that can
    be used with the command-line interface.

    Returns:
        list: List of available subtask names for DepthNet operations.

    Example:
        >>> subtasks = get_subtask_list()
        >>> print(f"Available tasks: {subtasks}")
        # Output: ['gen_trt_engine', 'inference', 'evaluate']
    """
    return get_subtasks(scripts)


def main():
    """
    Main entry point for DepthNet command-line operations.

    This function sets up the command-line argument parser and handles the
    execution of DepthNet subtasks. It provides a unified interface for all
    DepthNet operations including model conversion, inference, and evaluation.

    The function supports the following operations:
    - gen_trt_engine: Convert ONNX models to optimized TensorRT engines
    - inference: Perform depth estimation on input images
    - evaluate: Evaluate model performance with comprehensive metrics

    Command-line usage:
        ```bash
        # Generate TensorRT engine
        depth_net gen_trt_engine --config-file config.yaml

        # Perform inference
        depth_net inference --config-file config.yaml

        # Evaluate model
        depth_net evaluate --config-file config.yaml
        ```

    Raises:
        SystemExit: If command-line arguments are invalid or operation fails.
        ImportError: If required dependencies are not available.
    """
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "depth_net",
        add_help=True,
        description="Train Adapt Optimize Deploy entrypoint for DepthNet"
    )

    # Obtain the list of substasks
    subtasks = get_subtask_list()

    args, unknown_args = command_line_parser(parser, subtasks)

    # Parse the arguments and launch the subtask.
    launch(vars(args), unknown_args, subtasks, network="depth_net")


if __name__ == '__main__':
    main()
