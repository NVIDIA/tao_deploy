# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ScanNet++ data loader for NVPanoptix3Dv2 TensorRT inference."""

import json
import math
import os
import random

import numpy as np
from PIL import Image


class NVPanoptix3Dv2DataLoader:
    """Build deterministic multi-view batches from preprocessed ScanNet++.

    The loader consumes the same ``all_metadata.npz`` representation as the
    PyTorch model. Images are principal-point centered, resized with preserved
    aspect ratio, padded white, converted to RGB ``[0, 1]``, and returned in
    ``[B, S, 3, H, W]`` layout.

    Args:
        preprocessed_root: Directory containing ``all_metadata.npz``,
            ``categories.json``, and one image directory per scene.
        shape: TensorRT input shape ``[B, S, 3, H, W]``. A dynamic batch axis
            may be given as ``-1`` when ``batch_size`` is specified.
        dtype: NumPy dtype of the returned image tensor.
        batch_size: Batch size used when ``shape[0]`` is dynamic.
        pairs_per_scene: Number of deterministic samples drawn per scene.
        seed: Base seed for pair and neighboring-view selection.
        max_samples: Optional limit on the number of samples.
    """

    def __init__(
        self,
        preprocessed_root,
        shape,
        dtype=np.float32,
        batch_size=None,
        pairs_per_scene=1,
        seed=42,
        max_samples=None,
    ):
        self.preprocessed_root = os.path.realpath(
            os.path.expanduser(str(preprocessed_root))
        )
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = np.dtype(dtype)
        if not np.issubdtype(self.dtype, np.floating):
            raise ValueError(
                f"NVPanoptix3Dv2 image dtype must be floating point, got {self.dtype}"
            )
        self.pairs_per_scene = int(pairs_per_scene)
        self.seed = int(seed)
        self.batch_index = 0

        self.validate_input_shape(batch_size)
        self.load_metadata()
        self.build_pair_indices()

        self.num_samples = len(self.scenes) * self.pairs_per_scene
        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples <= 0:
                raise ValueError(
                    f"max_samples must be positive, got {max_samples}"
                )
            self.num_samples = min(self.num_samples, max_samples)
        self.num_batches = math.ceil(self.num_samples / self.batch_size)

    def validate_input_shape(self, batch_size):
        """Validate and resolve the TensorRT input dimensions."""
        if len(self.shape) != 5:
            raise ValueError(
                "NVPanoptix3Dv2 input shape must be [B, S, 3, H, W], "
                f"got {self.shape}"
            )
        shape_batch, self.num_views, channels, self.height, self.width = self.shape
        if channels != 3:
            raise ValueError(
                f"NVPanoptix3Dv2 expects three RGB channels, got {channels}"
            )
        if self.num_views < 2 or self.height <= 0 or self.width <= 0:
            raise ValueError(
                "View count, height, and width must be positive static "
                f"dimensions, got {self.shape}"
            )
        if shape_batch == -1:
            if batch_size is None or int(batch_size) <= 0:
                raise ValueError(
                    "A positive batch_size is required for a dynamic input shape"
                )
            self.batch_size = int(batch_size)
            self.dynamic_batch = True
        elif shape_batch > 0:
            if batch_size is not None and int(batch_size) != shape_batch:
                raise ValueError(
                    f"batch_size={batch_size} conflicts with shape {self.shape}"
                )
            self.batch_size = shape_batch
            self.dynamic_batch = False
        else:
            raise ValueError(
                f"Batch dimension must be positive or -1, got {shape_batch}"
            )
        if self.pairs_per_scene <= 0:
            raise ValueError(
                f"pairs_per_scene must be positive, got {self.pairs_per_scene}"
            )

    def load_metadata(self):
        """Load and validate the ScanNet++ metadata and class vocabulary."""
        metadata_path = os.path.join(
            self.preprocessed_root, "all_metadata.npz"
        )
        categories_path = os.path.join(
            self.preprocessed_root, "categories.json"
        )
        for path in (metadata_path, categories_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

        with open(categories_path, "r", encoding="utf-8") as handle:
            self.categories = json.load(handle)
        if not isinstance(self.categories, list) or not all(
            isinstance(category, dict) and category.get("name")
            for category in self.categories
        ):
            raise ValueError(
                f"Invalid category list in {categories_path}"
            )
        self.classes = [category["name"] for category in self.categories]

        with np.load(metadata_path, allow_pickle=True) as metadata:
            required = {"scenes", "sceneids", "images", "intrinsics", "pairs"}
            missing = required.difference(metadata.files)
            if missing:
                raise ValueError(
                    f"Missing metadata arrays in {metadata_path}: "
                    f"{', '.join(sorted(missing))}"
                )
            scenes = np.asarray(metadata["scenes"])
            self.scene_ids = metadata["sceneids"].astype(np.int64)
            images = np.asarray(metadata["images"])
            self.intrinsics = metadata["intrinsics"].astype(np.float32)
            pairs = np.asarray(metadata["pairs"])
        self.scenes = [str(scene) for scene in scenes]
        self.image_ids = [str(image) for image in images]

        image_count = len(self.image_ids)
        if not self.scenes or image_count == 0:
            raise ValueError(f"Empty ScanNet++ metadata in {metadata_path}")
        if not (
            len(self.scene_ids) == image_count == len(self.intrinsics)
        ):
            raise ValueError(
                "sceneids, images, and intrinsics must have equal lengths"
            )
        if pairs.ndim != 2 or pairs.shape[1] < 2:
            raise ValueError(
                f"pairs must have shape [N, >=2], got {pairs.shape}"
            )
        self.pairs = pairs[:, :2].astype(np.int64)

    def build_pair_indices(self):
        """Build per-image neighbors and per-scene pair lookup tables."""
        image_count = len(self.image_ids)
        self.pairs_per_image = [set() for _ in range(image_count)]
        self.scene_pair_indices = [[] for _ in self.scenes]
        for pair_index, (first, second) in enumerate(self.pairs):
            if not (
                0 <= first < image_count and 0 <= second < image_count
            ):
                raise ValueError(
                    f"Pair {pair_index} contains invalid indices: "
                    f"{first}, {second}"
                )
            first_scene = int(self.scene_ids[first])
            second_scene = int(self.scene_ids[second])
            if first_scene != second_scene:
                raise ValueError(
                    f"Pair {pair_index} crosses ScanNet++ scenes"
                )
            if not 0 <= first_scene < len(self.scenes):
                raise ValueError(
                    f"Image {first} has invalid scene index {first_scene}"
                )
            self.pairs_per_image[first].add(int(second))
            self.pairs_per_image[second].add(int(first))
            self.scene_pair_indices[first_scene].append(pair_index)

        empty_scenes = [
            self.scenes[index]
            for index, pair_indices in enumerate(self.scene_pair_indices)
            if not pair_indices
        ]
        if empty_scenes:
            raise ValueError(
                "Every scene must contain at least one pair; missing for: " +
                ", ".join(empty_scenes)
            )

    def scene_directory(self, image_index):
        """Return the scene directory for an image metadata index."""
        scene_index = int(self.scene_ids[image_index])
        return os.path.join(
            self.preprocessed_root, self.scenes[scene_index]
        )

    def image_path(self, image_index):
        """Return the RGB path for an image metadata index."""
        return os.path.join(
            self.scene_directory(image_index),
            "images",
            f"{self.image_ids[image_index]}.jpg",
        )

    def select_views(self, first, second, rng):
        """Select the deterministic multi-view tuple around an anchor pair."""
        selected = [int(first), int(second)]
        selected_set = set(selected)
        candidates = sorted(
            self.pairs_per_image[first] | self.pairs_per_image[second]
        )
        rng.shuffle(candidates)
        for candidate in candidates:
            if len(selected) >= self.num_views:
                break
            if candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
        while len(selected) < self.num_views:
            selected.append(selected[len(selected) % len(selected)])
        return selected[:self.num_views]

    def preprocess_image(self, image_index):
        """Load one view and reproduce the PyTorch spatial preprocessing."""
        path = self.image_path(image_index)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        with Image.open(path) as source:
            image = source.convert("RGB")
        intrinsics = self.intrinsics[image_index].copy()
        image_width, image_height = image.size
        center_x = int(round(float(intrinsics[0, 2])))
        center_y = int(round(float(intrinsics[1, 2])))
        margin_x = min(center_x, image_width - center_x)
        margin_y = min(center_y, image_height - center_y)
        if margin_x <= 0 or margin_y <= 0:
            raise ValueError(
                f"Invalid principal point ({center_x}, {center_y}) for "
                f"image {path} with size {image.size}"
            )

        left, top = center_x - margin_x, center_y - margin_y
        right, bottom = center_x + margin_x, center_y + margin_y
        image = image.crop((left, top, right, bottom))
        intrinsics[0, 2] -= left
        intrinsics[1, 2] -= top

        crop_width, crop_height = image.size
        scale = min(
            self.width / crop_width,
            self.height / crop_height,
        )
        resized_width = max(
            1, min(self.width, round(crop_width * scale))
        )
        resized_height = max(
            1, min(self.height, round(crop_height * scale))
        )
        image = image.resize(
            (resized_width, resized_height), Image.LANCZOS
        )
        intrinsics[0, :] *= resized_width / crop_width
        intrinsics[1, :] *= resized_height / crop_height

        pad_left = (self.width - resized_width) // 2
        pad_top = (self.height - resized_height) // 2
        canvas = Image.new(
            "RGB", (self.width, self.height), (255, 255, 255)
        )
        canvas.paste(image, (pad_left, pad_top))
        intrinsics[0, 2] += pad_left
        intrinsics[1, 2] += pad_top

        image_array = np.asarray(canvas, dtype=np.float32) / 255.0
        image_array = image_array.transpose(2, 0, 1).astype(
            self.dtype, copy=False
        )
        return image_array, intrinsics, path

    def get_sample(self, sample_index):
        """Load one deterministic scene sample and its identifying metadata."""
        if not 0 <= sample_index < self.num_samples:
            raise IndexError(sample_index)
        scene_index = sample_index % len(self.scenes)
        rng = random.Random(self.seed + sample_index)
        pair_index = rng.choice(self.scene_pair_indices[scene_index])
        first, second = self.pairs[pair_index]
        view_indices = self.select_views(first, second, rng)

        images = []
        adjusted_intrinsics = []
        image_paths = []
        for image_index in view_indices:
            image, intrinsics, path = self.preprocess_image(image_index)
            images.append(image)
            adjusted_intrinsics.append(intrinsics)
            image_paths.append(path)

        sample = np.stack(images)
        metadata = {
            "scene_id": self.scenes[scene_index],
            "view_ids": [self.image_ids[index] for index in view_indices],
            "image_paths": image_paths,
            "intrinsics": np.stack(adjusted_intrinsics),
            "input_size": (self.height, self.width),
        }
        return sample, metadata

    def __len__(self):
        """Return the number of TensorRT batches."""
        return self.num_batches

    def __iter__(self):
        """Reset batch iteration."""
        self.batch_index = 0
        return self

    def __next__(self):
        """Return one image batch and metadata for its real samples."""
        if self.batch_index >= self.num_batches:
            raise StopIteration

        start = self.batch_index * self.batch_size
        stop = min(start + self.batch_size, self.num_samples)
        output_batch_size = (
            stop - start if self.dynamic_batch else self.batch_size
        )
        batch = np.zeros(
            (
                output_batch_size,
                self.num_views,
                3,
                self.height,
                self.width,
            ),
            dtype=self.dtype,
        )
        metadata = []
        for batch_offset, sample_index in enumerate(range(start, stop)):
            batch[batch_offset], sample_metadata = self.get_sample(sample_index)
            metadata.append(sample_metadata)
        self.batch_index += 1
        return batch, metadata

    def get_batch(self):
        """Return an iterator over ``(images, metadata)`` batches."""
        return iter(self)


__all__ = ["NVPanoptix3Dv2DataLoader"]
