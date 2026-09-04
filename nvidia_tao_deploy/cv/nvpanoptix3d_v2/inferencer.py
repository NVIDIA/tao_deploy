# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT inference and panoptic postprocessing for NVPanoptix3Dv2."""

import numpy as np

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import do_inference


INPUT_NAME = "images"
REQUIRED_OUTPUT_NAMES = frozenset(("pred_logits", "pred_masks"))


def sigmoid(values):
    """Return a numerically stable NumPy sigmoid."""
    values = np.asarray(values)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def resize_masks(mask_probabilities, target_hw):
    """Resize ``[B, S, Q, H, W]`` masks with linear interpolation."""
    mask_probabilities = np.asarray(mask_probabilities)
    if mask_probabilities.ndim != 5:
        raise ValueError(
            "mask_probabilities must have shape [B,S,Q,H,W], got "
            f"{mask_probabilities.shape}"
        )
    target_height, target_width = (int(value) for value in target_hw)
    if target_height <= 0 or target_width <= 0:
        raise ValueError(
            f"target_hw must contain positive dimensions, got {target_hw}"
        )
    height, width = mask_probabilities.shape[-2:]
    if height <= 0 or width <= 0:
        raise ValueError(
            "Mask height and width must be positive, got "
            f"{mask_probabilities.shape}"
        )
    if (height, width) == (target_height, target_width):
        return mask_probabilities
    horizontal_positions = (
        (np.arange(target_width, dtype=np.float32) + 0.5) *
        width / target_width - 0.5
    )
    left_indices = np.floor(horizontal_positions).astype(np.int64)
    right_indices = left_indices + 1
    horizontal_weights = horizontal_positions - left_indices
    left_indices = np.clip(left_indices, 0, width - 1)
    right_indices = np.clip(right_indices, 0, width - 1)
    vertical_positions = (
        (np.arange(target_height, dtype=np.float32) + 0.5) *
        height / target_height - 0.5
    )
    top_indices = np.floor(vertical_positions).astype(np.int64)
    bottom_indices = top_indices + 1
    vertical_weights = vertical_positions - top_indices
    top_indices = np.clip(top_indices, 0, height - 1)
    bottom_indices = np.clip(bottom_indices, 0, height - 1)
    vertical_weights = vertical_weights.reshape(1, target_height, 1)

    batch_size, num_views, num_queries = mask_probabilities.shape[:3]
    resized = np.empty(
        (batch_size, num_views, num_queries, target_height, target_width),
        dtype=np.float32,
    )
    for batch_index in range(batch_size):
        for view_index in range(num_views):
            masks = mask_probabilities[batch_index, view_index].astype(
                np.float32, copy=False
            )
            left_values = masks[..., left_indices]
            horizontal = left_values + (
                masks[..., right_indices] - left_values
            ) * horizontal_weights
            top_values = horizontal[..., top_indices, :]
            resized[batch_index, view_index] = top_values + (
                horizontal[..., bottom_indices, :] - top_values
            ) * vertical_weights
    return resized


def postprocess_panoptic(
    outputs,
    target_hw,
    label_mode="sigmoid",
    cls_threshold=0.1,
    mask_threshold=0.25,
    overlap_threshold=0.5,
):
    """Convert named TensorRT outputs into multi-view panoptic maps.

    Args:
        outputs: Mapping containing ``pred_logits`` and ``pred_masks``, plus
            optional ``pred_objectness``.
        target_hw: Output ``(height, width)``.
        label_mode: Classification activation, ``sigmoid`` or ``softmax``.
        cls_threshold: Minimum query confidence.
        mask_threshold: Per-pixel mask probability threshold.
        overlap_threshold: Minimum retained-to-original mask area ratio.

    Returns:
        One dictionary per batch sample. Each dictionary contains an int32
        ``pan`` array ``[S, H, W]`` and a ``segments_info`` list.
    """
    missing = REQUIRED_OUTPUT_NAMES.difference(outputs)
    if missing:
        raise KeyError(
            "Missing TensorRT panoptic output(s): " +
            ", ".join(sorted(missing))
        )
    if label_mode not in {"sigmoid", "softmax"}:
        raise ValueError(
            f"label_mode must be 'sigmoid' or 'softmax', got {label_mode!r}"
        )

    mask_logits = np.asarray(outputs["pred_masks"])
    class_logits = np.asarray(outputs["pred_logits"])
    if mask_logits.ndim != 5 or class_logits.ndim != 3:
        raise ValueError(
            "Expected pred_masks [B,S,Q,H,W] and pred_logits [B,Q,C], "
            f"got {mask_logits.shape} and {class_logits.shape}"
        )
    batch_size, num_views, num_queries = mask_logits.shape[:3]
    if class_logits.shape[:2] != (batch_size, num_queries):
        raise ValueError(
            "pred_logits and pred_masks batch/query dimensions disagree: "
            f"{class_logits.shape} vs {mask_logits.shape}"
        )

    objectness = outputs.get("pred_objectness")
    if objectness is not None:
        objectness = np.asarray(objectness)
        if objectness.shape != (batch_size, num_queries):
            raise ValueError(
                "pred_objectness must have shape [B,Q], got "
                f"{objectness.shape}"
            )
        objectness = sigmoid(objectness)

    mask_probabilities = resize_masks(sigmoid(mask_logits), target_hw)
    target_height, target_width = mask_probabilities.shape[-2:]
    results = []
    for batch_index in range(batch_size):
        logits = class_logits[batch_index]
        if label_mode == "sigmoid":
            probabilities = sigmoid(logits)
            labels = probabilities.argmax(axis=-1)
            scores = probabilities[
                np.arange(num_queries), labels
            ]
            if objectness is not None:
                scores = scores * objectness[batch_index]
            keep = scores > cls_threshold
        else:
            shifted = logits - logits.max(axis=-1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            labels = probabilities.argmax(axis=-1)
            scores = probabilities[
                np.arange(num_queries), labels
            ]
            if objectness is not None:
                scores = scores * objectness[batch_index]
            no_object_class = class_logits.shape[-1] - 1
            keep = (labels != no_object_class) & (scores > cls_threshold)

        kept_scores = scores[keep]
        kept_labels = labels[keep]
        kept_masks = mask_probabilities[batch_index][:, keep].transpose(
            1, 0, 2, 3
        )
        panoptic_map = np.zeros(
            (num_views, target_height, target_width), dtype=np.int32
        )
        segments_info = []
        if kept_masks.shape[0] == 0:
            results.append(
                {"pan": panoptic_map, "segments_info": segments_info}
            )
            continue

        weighted_masks = kept_scores[:, None, None, None] * kept_masks
        assigned_queries = weighted_masks.argmax(axis=0)
        for query_index, (category_id, score) in enumerate(
            zip(kept_labels, kept_scores)
        ):
            original_area = int((kept_masks[query_index] >= 0.5).sum())
            mask = (
                (assigned_queries == query_index) &
                (kept_masks[query_index] >= mask_threshold)
            )
            mask_area = int(mask.sum())
            if (
                mask_area == 0 or
                original_area == 0 or
                mask_area / original_area < overlap_threshold
            ):
                continue
            segment_id = len(segments_info) + 1
            panoptic_map[mask] = segment_id
            segments_info.append(
                {
                    "id": segment_id,
                    "category_id": int(category_id),
                    "score": float(score),
                    "area": mask_area,
                }
            )
        results.append(
            {"pan": panoptic_map, "segments_info": segments_info}
        )
    return results


def outputs_to_dict(outputs):
    """Reshape TensorRT host buffers and key them by output tensor name."""
    result = {}
    for output in outputs:
        shape = tuple(int(dim) for dim in output.numpy_shape)
        volume = int(np.prod(shape))
        if volume <= 0 or output.host.size < volume:
            raise ValueError(
                f"Invalid runtime shape {shape} for output {output.name}"
            )
        result[output.name] = np.reshape(output.host[:volume], shape)
    return result


class NVPanoptix3Dv2Inferencer(TRTInferencer):
    """Execute an NVPanoptix3Dv2 panoptic TensorRT engine."""

    def __init__(
        self,
        engine_path,
        input_shape=None,
        batch_size=None,
        data_format="channel_first",
    ):
        """Load the engine and allocate its input and output buffers."""
        if data_format != "channel_first":
            raise ValueError(
                "NVPanoptix3Dv2 uses [B,S,3,H,W] channel-first input"
            )
        if input_shape is None and batch_size is None:
            batch_size = 1
        if input_shape is not None:
            input_shape = tuple(int(dim) for dim in input_shape)
            if len(input_shape) != 5 or input_shape[0] <= 0:
                raise ValueError(
                    "input_shape must be a resolved [B,S,3,H,W] shape, got "
                    f"{input_shape}"
                )
            if batch_size is not None and int(batch_size) != input_shape[0]:
                raise ValueError(
                    f"batch_size={batch_size} conflicts with input_shape "
                    f"{input_shape}"
                )
            self.allocated_batch_size = input_shape[0]
        else:
            self.allocated_batch_size = int(batch_size)
            if self.allocated_batch_size <= 0:
                raise ValueError(
                    f"batch_size must be positive, got {batch_size}"
                )
        super().__init__(
            engine_path,
            input_shape=input_shape,
            batch_size=batch_size,
            data_format=data_format,
            reshape=False,
        )
        if len(self.input_tensors) != 1:
            raise ValueError(
                "NVPanoptix3Dv2 requires exactly one TensorRT input"
            )
        input_tensor = self.input_tensors[0]
        if input_tensor.tensor_name != INPUT_NAME:
            raise ValueError(
                f"NVPanoptix3Dv2 input must be '{INPUT_NAME}', got "
                f"'{input_tensor.tensor_name}'"
            )
        if len(input_tensor.tensor_shape) != 5:
            raise ValueError(
                "NVPanoptix3Dv2 input must have shape [B,S,3,H,W], got "
                f"{input_tensor.tensor_shape}"
            )
        output_names = {
            output.tensor_name for output in self.output_tensors
        }
        missing = REQUIRED_OUTPUT_NAMES.difference(output_names)
        if missing:
            raise ValueError(
                "Engine is missing required output(s): " +
                ", ".join(sorted(missing))
            )

    def validate_images(self, images):
        """Validate an image batch against the engine input contract."""
        images = np.asarray(images)
        if images.ndim != 5:
            raise ValueError(
                f"images must have shape [B,S,3,H,W], got {images.shape}"
            )
        engine_shape = tuple(self.input_tensors[0].tensor_shape)
        allocated_batch_size = getattr(
            self, "allocated_batch_size", images.shape[0]
        )
        if images.shape[0] > allocated_batch_size:
            raise ValueError(
                f"Batch size {images.shape[0]} exceeds the allocated TensorRT "
                f"batch size {allocated_batch_size}"
            )
        for axis, (actual, expected) in enumerate(
            zip(images.shape, engine_shape)
        ):
            if expected > 0 and actual != expected:
                raise ValueError(
                    f"images axis {axis} must be {expected}, got {actual}"
                )
        profile = getattr(
            self.input_tensors[0], "optimization_profile", None
        )
        if profile is not None:
            minimum, _, maximum = profile
            for axis, actual in enumerate(images.shape):
                if not minimum[axis] <= actual <= maximum[axis]:
                    raise ValueError(
                        f"images axis {axis}={actual} is outside TensorRT "
                        f"profile [{minimum[axis]}, {maximum[axis]}]"
                    )
        return np.ascontiguousarray(images)

    def infer(self, images):
        """Execute one preprocessed ``[B,S,3,H,W]`` image batch.

        Returns:
            Dictionary of NumPy arrays keyed by TensorRT output name.
        """
        images = self.validate_images(images)
        input_tensor = self.input_tensors[0]
        if getattr(input_tensor, "optimization_profile", None) is not None:
            accepted = self.context.set_input_shape(
                input_tensor.tensor_name, tuple(images.shape)
            )
            if accepted is False:
                raise ValueError(
                    f"TensorRT rejected input shape {images.shape}"
                )

        self._copy_input_to_host([images])
        raw_outputs = do_inference(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
            batch_size=images.shape[0],
            execute_v2=self.execute_async,
            return_raw=True,
        )
        for output in raw_outputs:
            output.numpy_shape = tuple(
                int(dim)
                for dim in self.context.get_tensor_shape(output.name)
            )
        return outputs_to_dict(raw_outputs)

    @staticmethod
    def postprocess(
        outputs,
        target_hw,
        label_mode="sigmoid",
        cls_threshold=0.1,
        mask_threshold=0.25,
        overlap_threshold=0.5,
    ):
        """Run panoptic postprocessing on named raw engine outputs."""
        return postprocess_panoptic(
            outputs,
            target_hw=target_hw,
            label_mode=label_mode,
            cls_threshold=cls_threshold,
            mask_threshold=mask_threshold,
            overlap_threshold=overlap_threshold,
        )


__all__ = [
    "NVPanoptix3Dv2Inferencer",
    "outputs_to_dict",
    "postprocess_panoptic",
]
