# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare NVPanoptix3Dv2 panoptic ONNX graphs for TensorRT.

The PyTorch export contains two constructs that are valid ONNX but unsuitable
for the TensorRT panoptic engine:

* The metric-scale head sorts every image pixel to calculate quantiles. At the
  exported 518x518 resolution that becomes TopK with K=268324, beyond
  TensorRT's TopK limit. Metric outputs are derived from raw geometry, so the
  TensorRT graph keeps ``depth``, ``world_points`` and ``pose_enc`` and leaves
  ``metric_depth``, ``metric_points`` and ``intrinsics`` to postprocessing.
* The mask decoder uses advanced in-place indexing to clear fully masked
  attention rows. PyTorch lowers it through NonZero and ScatterND with a
  data-dependent update count. It is rewritten to an equivalent broadcast
  Where operation.

The source ONNX graph is never overwritten. A small derived protobuf is saved
beside it so all relative external-weight references remain valid.
"""

from dataclasses import dataclass
import logging
import os
import tempfile
from typing import FrozenSet, List, Set

import numpy as np
import onnx
from onnx import helper, numpy_helper


logger = logging.getLogger(__name__)

DERIVED_SUFFIX = ".tensorrt_panoptic.onnx"
METRIC_OUTPUT_NAMES = frozenset(("metric_depth", "metric_points", "intrinsics"))
MASK_SCATTER_PREFIX = "/model/panoptic_decoder/mask_transformer/ScatterND"


@dataclass(frozen=True)
class SanitizationReport:
    """Summary of transformations applied to a derived ONNX graph."""

    output_path: str
    dropped_outputs: FrozenSet[str]
    rewritten_mask_updates: int
    removed_nodes: int


def rewrite_mask_updates(model):
    """Replace data-dependent mask ScatterND updates with broadcast Where."""
    has_mask_updates = any(
        node.op_type == "ScatterND" and node.name.startswith(MASK_SCATTER_PREFIX)
        for node in model.graph.node
    )
    if not has_mask_updates:
        return 0

    nodes_by_name = {node.name: node for node in model.graph.node}
    axes_name = "nvpanoptix3d_v2_trt_unsqueeze_last_axis"
    false_name = "nvpanoptix3d_v2_trt_false"
    existing_initializers = {item.name for item in model.graph.initializer}
    if axes_name not in existing_initializers:
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray([-1], dtype=np.int64), name=axes_name)
        )
    if false_name not in existing_initializers:
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(False, dtype=np.bool_), name=false_name)
        )

    rewritten_nodes: List[onnx.NodeProto] = []
    rewrite_count = 0
    for node in model.graph.node:
        is_mask_update = (
            node.op_type == "ScatterND" and
            node.name.startswith(MASK_SCATTER_PREFIX)
        )
        if not is_mask_update:
            rewritten_nodes.append(node)
            continue

        nonzero_name = node.name.replace("/ScatterND", "/NonZero", 1)
        nonzero_node = nodes_by_name.get(nonzero_name)
        if nonzero_node is None or nonzero_node.op_type != "NonZero":
            raise ValueError(
                f"Expected matching NonZero node '{nonzero_name}' for mask "
                f"update '{node.name}'."
            )

        data_input = node.input[0]
        fully_masked_condition = nonzero_node.input[0]
        expanded_condition = f"{node.name}/TensorRTSafeCondition_output_0"
        rewritten_nodes.append(
            helper.make_node(
                "Unsqueeze",
                inputs=[fully_masked_condition, axes_name],
                outputs=[expanded_condition],
                name=f"{node.name}/TensorRTSafeCondition",
            )
        )
        rewritten_nodes.append(
            helper.make_node(
                "Where",
                inputs=[expanded_condition, false_name, data_input],
                outputs=list(node.output),
                name=f"{node.name}/TensorRTSafeWhere",
            )
        )
        rewrite_count += 1

    model.graph.ClearField("node")
    model.graph.node.extend(rewritten_nodes)
    return rewrite_count


def drop_metric_outputs(model):
    """Remove derived metric outputs while retaining raw geometry outputs."""
    dropped = {
        output.name
        for output in model.graph.output
        if output.name in METRIC_OUTPUT_NAMES
    }
    retained_outputs = [
        output
        for output in model.graph.output
        if output.name not in METRIC_OUTPUT_NAMES
    ]
    model.graph.ClearField("output")
    model.graph.output.extend(retained_outputs)
    return frozenset(dropped)


def prune_dead_graph(model):
    """Remove nodes and initializers that cannot reach a graph output."""
    required_tensors: Set[str] = {output.name for output in model.graph.output}
    retained_nodes = []
    for node in reversed(model.graph.node):
        if any(output in required_tensors for output in node.output):
            retained_nodes.append(node)
            required_tensors.update(name for name in node.input if name)
    retained_nodes.reverse()

    removed_nodes = len(model.graph.node) - len(retained_nodes)
    model.graph.ClearField("node")
    model.graph.node.extend(retained_nodes)

    retained_initializers = [
        item for item in model.graph.initializer if item.name in required_tensors
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(retained_initializers)

    retained_value_info = [
        item for item in model.graph.value_info if item.name in required_tensors
    ]
    model.graph.ClearField("value_info")
    model.graph.value_info.extend(retained_value_info)
    return removed_nodes


def derived_path(onnx_path):
    """Return the deterministic derived graph path beside ``onnx_path``."""
    stem, extension = os.path.splitext(os.path.realpath(onnx_path))
    if extension.lower() != ".onnx":
        raise ValueError(f"Expected an .onnx model path, got: {onnx_path}")
    return f"{stem}{DERIVED_SUFFIX}"


def prepare_onnx_for_tensorrt(onnx_path):
    """Create and return a TensorRT-safe derivative of ``onnx_path``.

    External tensor payloads are deliberately not loaded or rewritten. The
    derived protobuf remains in the source directory and references the same
    sibling files as the original graph.
    """
    source_path = os.path.realpath(onnx_path)
    output_path = derived_path(source_path)
    model = onnx.load(source_path, load_external_data=False)
    original_node_count = len(model.graph.node)

    rewritten_mask_updates = rewrite_mask_updates(model)
    dropped_outputs = drop_metric_outputs(model)
    removed_nodes = prune_dead_graph(model)

    remaining_ops = {node.op_type for node in model.graph.node}
    if "TopK" in remaining_ops:
        raise ValueError(
            "NVPanoptix3Dv2 TensorRT graph still contains TopK after removing "
            "the unsupported metric-output branch."
        )
    if "NonZero" in remaining_ops:
        raise ValueError(
            "NVPanoptix3Dv2 TensorRT graph still contains a data-dependent "
            "NonZero operation after mask-update rewriting."
        )

    metadata = {item.key: item.value for item in model.metadata_props}
    metadata["nvidia_tao_deploy.nvpanoptix3d_v2_tensorrt_sanitized"] = "1"
    model.ClearField("metadata_props")
    helper.set_model_props(model, metadata)

    output_dir = os.path.dirname(output_path)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_path)}.",
        suffix=".tmp",
        dir=output_dir,
    )
    os.close(file_descriptor)
    try:
        onnx.save_model(model, temporary_path)
        # Validate by path so ONNX resolves external tensor payloads relative
        # to the source directory shared by the temporary and final graphs.
        onnx.checker.check_model(temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

    report = SanitizationReport(
        output_path=output_path,
        dropped_outputs=dropped_outputs,
        rewritten_mask_updates=rewritten_mask_updates,
        removed_nodes=removed_nodes,
    )
    logger.info(
        "Prepared TensorRT ONNX graph %s: dropped outputs=%s, rewritten mask "
        "updates=%d, pruned nodes=%d, nodes=%d->%d",
        output_path,
        ", ".join(sorted(dropped_outputs)) or "none",
        rewritten_mask_updates,
        removed_nodes,
        original_node_count,
        len(model.graph.node),
    )
    return report
