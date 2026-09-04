# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT engine builder for the NVPanoptix3Dv2 panoptic variant."""

import logging

from nvidia_tao_deploy.cv.nvpanoptix3d_v2.onnx_sanitizer import (
    prepare_onnx_for_tensorrt,
)
from nvidia_tao_deploy.engine.builder import EngineBuilder


logger = logging.getLogger(__name__)

INPUT_NAME = "images"
REQUIRED_OUTPUT_NAMES = frozenset(("pred_logits", "pred_masks"))


class NVPanoptix3Dv2EngineBuilder(EngineBuilder):
    """Parse and build the exported NVPanoptix3Dv2 panoptic ONNX graph.

    The shared :class:`EngineBuilder` owns TensorRT parsing, optimization
    profiles, precision selection, timing caches, and serialization. This
    specialization protects that generic path with the NVPanoptix3Dv2 export
    contract: one ``images`` input in ``[B, S, 3, H, W]`` layout, with only the
    batch axis allowed to be dynamic, and the two required panoptic outputs.
    """

    def create_network(self, model_path, file_format="onnx"):
        """Parse an ONNX graph and validate the panoptic tensor interface.

        Args:
            model_path: Path to the ONNX graph. External weight data is
                resolved by TensorRT relative to this file.
            file_format: Must be ``"onnx"``.

        Raises:
            ValueError: If the graph does not match the panoptic export
                contract.
        """
        if str(file_format).lower() != "onnx":
            raise ValueError("NVPanoptix3Dv2 deployment supports ONNX input only.")

        report = prepare_onnx_for_tensorrt(model_path)
        logger.info(
            "Building the panoptic TensorRT engine from sanitized graph %s. "
            "Derived metric outputs removed from the engine: %s",
            report.output_path,
            ", ".join(sorted(report.dropped_outputs)) or "none",
        )
        super().create_network(report.output_path, file_format="onnx")
        self.validate_panoptic_network()

    def validate_panoptic_network(self):
        """Validate input layout and required output names after ONNX parsing."""
        if self.network.num_inputs != 1:
            raise ValueError(
                "NVPanoptix3Dv2 panoptic ONNX must have exactly one input "
                f"named '{INPUT_NAME}', but parsed {self.network.num_inputs}."
            )

        model_input = self.network.get_input(0)
        if model_input.name != INPUT_NAME:
            raise ValueError(
                "NVPanoptix3Dv2 panoptic ONNX input must be named "
                f"'{INPUT_NAME}', got '{model_input.name}'."
            )

        input_shape = tuple(int(dim) for dim in model_input.shape)
        if len(input_shape) != 5:
            raise ValueError(
                "NVPanoptix3Dv2 panoptic ONNX input must have rank 5 in "
                f"[B, S, 3, H, W] layout, got shape {input_shape}."
            )

        batch, num_views, channels, height, width = input_shape
        if batch == 0 or batch < -1:
            raise ValueError(
                "NVPanoptix3Dv2 batch dimension must be a positive static "
                f"value or -1, got {batch}."
            )
        if num_views < 2:
            raise ValueError(
                "NVPanoptix3Dv2 exports require at least two static views; "
                f"got {num_views} in shape {input_shape}."
            )
        if channels != 3:
            raise ValueError(
                "NVPanoptix3Dv2 expects three RGB channels at input axis 2; "
                f"got {channels} in shape {input_shape}."
            )
        if height <= 0 or width <= 0:
            raise ValueError(
                "NVPanoptix3Dv2 exports require static positive spatial "
                f"dimensions; got {height}x{width}."
            )

        output_names = {
            self.network.get_output(index).name
            for index in range(self.network.num_outputs)
        }
        missing_outputs = REQUIRED_OUTPUT_NAMES - output_names
        if missing_outputs:
            raise ValueError(
                "The ONNX graph is not an NVPanoptix3Dv2 panoptic export; "
                "missing required output(s): " + ", ".join(sorted(missing_outputs))
            )

        self.num_views = num_views
        self.input_height = height
        self.input_width = width
        logger.info(
            "Validated NVPanoptix3Dv2 panoptic graph: input %s, outputs %s",
            input_shape,
            ", ".join(sorted(output_names)),
        )
