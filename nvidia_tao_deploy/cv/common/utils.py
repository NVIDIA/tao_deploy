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

"""Helper functions."""

import logging
import os

import onnx

logger = logging.getLogger(__name__)

TASKS = {"gen_trt_engine": "gen_trt_engine",
         "evaluate": "trt_evaluate",
         "inference": "trt_inference"}


def update_results_dir(cfg, task):
    """Update global results_dir based on task.results_dir.

    This function should be called at the beginning of a pipeline script.

    Args:
        cfg (Hydra config): Config object loaded by Hydra
        task (str): TAO pipeline name (gen_trt_engine, evaluate, inference)
    Return:
        Updated cfg
    """
    if cfg[task]['results_dir']:
        cfg['results_dir'] = cfg[task]['results_dir']
    elif cfg['results_dir']:
        cfg['results_dir'] = os.path.join(cfg['results_dir'], TASKS[task])
        cfg[task]['results_dir'] = cfg['results_dir']
    else:
        raise ValueError(f"You need to set at least one of following fields: results_dir, {task}.results_dir")
    print(f"{TASKS[task].capitalize()} results will be saved at: {cfg['results_dir']}")

    return cfg


def is_qdq_quantized_onnx(onnx_file_path):
    """Check if an ONNX model contains QDQ (QuantizeLinear/DequantizeLinear) quantization nodes.

    This function detects if an ONNX model has been quantized using the QDQ (Quantize-Dequantize)
    format, which includes operators like QuantizeLinear, DequantizeLinear, and other Q-prefixed
    linear operators.

    Parameters
    ----------
    onnx_file_path : str
        Path to the ONNX model file.

    Returns
    -------
    bool
        True if the model contains QDQ quantization nodes, False otherwise.
        Returns False if unable to parse the ONNX model.
    """
    try:
        # Load the ONNX model
        model = onnx.load(onnx_file_path)

        # Check for quantization-specific operators
        quantization_ops = {'QuantizeLinear', 'DequantizeLinear', 'QLinearConv',
                            'QLinearMatMul', 'QLinearAdd', 'QLinearMul'}

        # Iterate through all nodes in the model graph
        for node in model.graph.node:
            if node.op_type in quantization_ops:
                logger.info("Detected quantization operator '%s' in ONNX model. "
                            "Model is quantized.", node.op_type)
                return True

        logger.info("No QDQ quantization operators found in ONNX model.")
        return False
    except (FileNotFoundError, IOError) as e:
        logger.warning("ONNX file not accessible: %s. Assuming not quantized.", e)
        return False
    except Exception as e:
        logger.warning("Failed to parse ONNX model: %s. Assuming not quantized.", e)
        return False
