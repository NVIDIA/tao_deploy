# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a TensorRT engine for an NVPanoptix3Dv2 panoptic ONNX export."""

import logging
import math
import os

from nvidia_tao_deploy.config.nvpanoptix3d_v2.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.common.initialize_experiments import (
    initialize_gen_trt_engine_experiment,
)
from nvidia_tao_deploy.cv.nvpanoptix3d_v2.engine_builder import (
    NVPanoptix3Dv2EngineBuilder,
)


logging.basicConfig(
    format="%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s",
    level="INFO",
)
logger = logging.getLogger(__name__)
SPEC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANOPTIC = "panoptic"


def workspace_megabytes_to_gibibytes(workspace_size_mb):
    """Convert the config's MB workspace value to builder GiB units.

    The common TAO schema exposes workspace in MB while ``EngineBuilder``
    accepts whole GiB. Rounding up preserves small, valid user requests rather
    than accidentally passing a zero-byte TensorRT workspace.
    """
    if workspace_size_mb <= 0:
        raise ValueError(
            f"gen_trt_engine.tensorrt.workspace_size must be positive, got {workspace_size_mb}."
        )
    return max(1, math.ceil(workspace_size_mb / 1024))


def validate_batch_profile(trt_config):
    """Reject invalid dynamic-batch profile ordering before TensorRT runs."""
    min_batch = trt_config.min_batch_size
    opt_batch = trt_config.opt_batch_size
    max_batch = trt_config.max_batch_size
    if min_batch <= 0 or opt_batch <= 0 or max_batch <= 0:
        raise ValueError(
            "TensorRT min/opt/max batch sizes must all be positive, got "
            f"{min_batch}, {opt_batch}, {max_batch}."
        )
    if not min_batch <= opt_batch <= max_batch:
        raise ValueError(
            "TensorRT batch profile must satisfy min_batch_size <= "
            "opt_batch_size <= max_batch_size, got "
            f"{min_batch} <= {opt_batch} <= {max_batch}."
        )


def run_engine_builder(cfg):
    """Build and serialize the TensorRT engine described by ``cfg``."""
    model_type = str(cfg.model.model_type)
    if model_type != PANOPTIC:
        raise ValueError(
            f"TensorRT deployment supports model.model_type='{PANOPTIC}' only, "
            f"got '{model_type}'."
        )

    onnx_file = cfg.gen_trt_engine.onnx_file
    engine_file = cfg.gen_trt_engine.trt_engine
    if not onnx_file:
        raise ValueError("gen_trt_engine.onnx_file must point to the exported ONNX graph.")
    if not os.path.isfile(onnx_file):
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_file}")
    if not engine_file:
        raise ValueError("gen_trt_engine.trt_engine must specify the output engine path.")
    if os.path.realpath(onnx_file) == os.path.realpath(engine_file):
        raise ValueError("The TensorRT engine path must not overwrite the input ONNX graph.")
    batch_size = cfg.gen_trt_engine.batch_size
    if batch_size == 0 or batch_size < -1:
        raise ValueError(
            "gen_trt_engine.batch_size must be positive or -1 for dynamic batch, "
            f"got {batch_size}."
        )

    trt_config = cfg.gen_trt_engine.tensorrt
    validate_batch_profile(trt_config)
    workspace_gib = workspace_megabytes_to_gibibytes(trt_config.workspace_size)

    engine_builder_kwargs, create_engine_kwargs = initialize_gen_trt_engine_experiment(cfg)
    builder = NVPanoptix3Dv2EngineBuilder(
        **engine_builder_kwargs,
        workspace=workspace_gib,
    )

    # Parse by path so TensorRT can resolve the external data files emitted by
    # the >2 GB NVPanoptix3Dv2 export without materializing another ONNX proto.
    builder.create_network(onnx_file, file_format="onnx")
    builder.create_engine(**create_engine_kwargs)
    return engine_file


@hydra_runner(
    config_path=os.path.join(SPEC_ROOT, "specs"),
    config_name="gen_trt_engine",
    schema=ExperimentConfig,
)
@monitor_status(name="nvpanoptix3d_v2", mode="gen_trt_engine")
def main(cfg: ExperimentConfig) -> None:
    """Convert a panoptic ONNX export to a TensorRT engine."""
    engine_file = run_engine_builder(cfg)
    logger.info("TensorRT engine was saved at %s.", engine_file)


if __name__ == "__main__":
    main()
