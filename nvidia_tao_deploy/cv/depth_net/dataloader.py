# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""DepthNet data loader based on annotation file lists.

Overview
--------
DepthNetDataLoader is a lightweight, dependency-free batcher tailored for
DepthNet TensorRT inference/evaluation. It reads simple annotation text files
and produces preprocessed batches compatible with DepthNet engines.

Supported annotation formats (per-line)
--------------------------------------
- Monocular, no GT:                      ``<image>``
- Monocular, with GT:                    ``<image> <gt_depth>``
- Stereo, no GT:                         ``<left> <right>``
- Stereo, with GT:                       ``<left> <right> <gt_depth>``

Notes
-----
- Lines starting with ``#`` or empty lines are ignored.
- All paths should be absolute for robustness.
- Ground-truth depth supports ``.pfm`` and ``.png`` encodings (see ``read_gt_depth``).

Input/Output
------------
- Input shape must be a 4-tuple matching the engine input: either ``(B, C, H, W)``
  or ``(B, H, W, C)``; the loader infers layout and preprocesses accordingly.
- Preprocessing: resize to target size, normalize via ImageNet stats, and format
  to channel-first if needed.
- Yielded batches:
  - Mono: ``np.ndarray`` with shape ``shape``; returns ``(batch, image_paths, scales)``
  - Stereo: ``dict`` with keys ``left_image`` and ``right_image`` (both shaped ``shape``);
    returns ``(batch_dict, left_image_paths, scales)``
  - In evaluation mode, an additional ``gt_depths`` array is yielded as the 4th item.

Usage examples
--------------
Monocular inference
~~~~~~~~~~~~~~~~~~~
>>> loader = DepthNetDataLoader(
...     data_sources=[{"dataset_name": "RelativeMonoDataset", "data_file": "/abs/list.txt"}],
...     shape=(4, 3, 518, 924),
...     dtype=np.float32,
...     preprocessor="DepthNet",
...     evaluation=False,
... )
>>> for batch, img_paths, scales in loader.get_batch():
...     preds = trt_infer.infer(batch)

Stereo evaluation
~~~~~~~~~~~~~~~~~
>>> loader = DepthNetDataLoader(
...     data_sources=[{"dataset_name": "GenericDataset", "data_file": "/abs/list.txt"}],
...     shape=(2, 3, 320, 736),
...     dtype=np.float32,
...     preprocessor="DepthNet",
...     evaluation=True,
... )
>>> for batches, img_paths, scales, gt_depths in loader.get_batch():
...     preds = trt_infer.infer(batches)

Limitations
-----------
- This loader focuses on deterministic preprocessing for deploy-time usage.
- It performs minimal validation; ensure your annotation files are consistent.
"""

import os
from typing import List, Tuple
import numpy as np
from PIL import Image

from nvidia_tao_deploy.inferencer.preprocess_input import preprocess_input
from nvidia_tao_deploy.cv.deformable_detr.dataloader import resize
from nvidia_tao_deploy.cv.depth_net.utils import read_gt_depth


class DepthNetDataLoader:
    """DepthNet dataloader that parses annotation files and yields preprocessed batches.

    Expected data_sources format (from YAML):
    - data_sources: list of dicts with keys:
        - dataset_name: e.g., "GenericDataset" (stereo) or "RelativeMonoDataset" (mono)
        - data_file: path to a text file with one sample per line

    Each line in ``data_file`` can be one of:
    - mono without GT:  "image"
    - mono with GT:     "image gt"
    - stereo with GT:   "left right gt"
    - stereo without GT:"left right"
    Fields are whitespace-separated. Lines beginning with '#' or empty lines are ignored.
    """

    def __init__(
        self,
        data_sources: List[dict],
        shape: Tuple[int, int, int, int],
        dtype,
        preprocessor: str = "DepthNet",
        img_std: List[float] = [0.229, 0.224, 0.225],
        img_mean: List[float] = [0.485, 0.456, 0.406],
        evaluation: bool = False,
    ) -> None:
        """Initialize the DepthNetDataLoader.

        Args:
            data_sources (List[dict]): List of data source dicts. Each must contain
                a key ``data_file`` that points to an annotation text file.
            shape (Tuple[int, int, int, int]): Target input tensor shape. Supports
                ``(B, C, H, W)`` or ``(B, H, W, C)``.
            dtype: Numpy dtype to cast image tensors to (e.g., ``np.float32``).
            preprocessor (str): Preprocessor type. Only ``"DepthNet"`` is supported.
            img_std (List[float]): Per-channel standard deviation used for normalization.
            img_mean (List[float]): Per-channel mean used for normalization.
            evaluation (bool): If True, yields ground-truth depth arrays as the
                fourth item of each batch tuple when annotations include GT.

        Raises:
            FileNotFoundError: If any provided ``data_file`` path does not exist.
            ValueError: If no samples are found or list lengths are inconsistent.
        """
        self.shape = shape
        self.dtype = dtype
        self.preprocessor = preprocessor
        self.img_std = img_std
        self.img_mean = img_mean
        self.evaluation = evaluation

        left_list: List[str] = []
        right_list: List[str] = []
        gt_list: List[str] = []

        for data_source in data_sources:
            data_file = data_source.get("data_file")
            if data_file is None:
                continue
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Provided data_file {data_file} does not exist!")
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) == 1:
                        # mono, no GT
                        left_list.append(parts[0])
                    elif len(parts) == 2:
                        # mono with GT OR stereo without GT
                        # Heuristic: if second token looks like an image extension, treat as stereo
                        if os.path.splitext(parts[1])[1].lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                            left_list.append(parts[0])
                            right_list.append(parts[1])
                        else:
                            left_list.append(parts[0])
                            gt_list.append(parts[1])
                    elif len(parts) >= 3:
                        # stereo with GT (ignore extra tokens)
                        left_list.append(parts[0])
                        right_list.append(parts[1])
                        gt_list.append(parts[2])
                    else:
                        continue

        # Store lists
        self._left_list = left_list
        self._right_list = right_list
        self._gt_list = gt_list

        # Determine layout from shape
        assert len(shape) == 4
        self.batch_size = shape[0]
        if shape[1] in [1, 3]:
            self.height, self.width, self.channel = shape[2], shape[3], shape[1]
            self.channels_first = True
        else:
            self.height, self.width, self.channel = shape[1], shape[2], shape[3]
            self.channels_first = False

        # Validate lengths
        if len(self._left_list) == 0:
            raise ValueError("No samples found in provided data_files.")
        if len(self._right_list) not in (0, len(self._left_list)):
            raise ValueError("Number of right images does not match left images.")
        if self.evaluation and len(self._gt_list) not in (0, len(self._left_list)):
            raise ValueError("Number of GT depth files does not match left images.")

        # Build batches as indices
        self.num_images = len(self._left_list)
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self._batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self._batches.append(list(range(start, end)))

    def _preprocess_left(self, image_path: str):
        """Load and preprocess an image to the target network input format.

        The image is loaded as RGB, resized to ``(self.height, self.width)``,
        normalized with ImageNet statistics, and formatted according to the
        channel layout inferred from ``shape``.

        Args:
            image_path (str): Absolute path to the input image.

        Returns:
            Tuple[np.ndarray, Tuple[float, float]]: A tuple of
                (preprocessed_image, scale), where ``preprocessed_image`` has the
                same layout as ``shape`` and ``scale`` is a tuple
                ``(orig_h / new_h, orig_w / new_w)`` for post-resize restoration.
        """
        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image, dtype=self.dtype)
        orig_h, orig_w = image.shape[:2]
        image, _ = resize(image, None, size=(self.height, self.width))
        image = preprocess_input(
            image,
            data_format='channels_first',
            img_mean=self.img_mean,
            img_std=self.img_std,
            mode='torch',
        )
        new_h, new_w = image.shape[:2]
        scale = (orig_h / new_h, orig_w / new_w)
        if self.channels_first and self.channel != 1:
            image = np.transpose(image, (2, 0, 1))
        return image, scale

    def get_batch(self):
        """Iterate over preprocessed batches.

        Yields:
            - Monocular (evaluation=False):
                ``Tuple[np.ndarray, List[str], List[Tuple[float, float]]]``
            - Monocular (evaluation=True):
                ``Tuple[np.ndarray, List[str], List[Tuple[float, float]], np.ndarray]``
            - Stereo (evaluation=False):
                ``Tuple[Dict[str, np.ndarray], List[str], List[Tuple[float, float]]]``
                where the dict has keys ``"left_image"`` and ``"right_image"``.
            - Stereo (evaluation=True):
                ``Tuple[Dict[str, np.ndarray], List[str], List[Tuple[float, float]], np.ndarray]``

        Notes:
            - ``scales`` correspond to the left image for stereo batches.
            - The last batch is not padded; it may contain fewer than ``batch_size`` samples.
        """
        for batch_indices in self._batches:
            # stereo
            if len(self._right_list) > 0:
                batch_data = {
                    "left_image": np.zeros(self.shape, dtype=self.dtype),
                    "right_image": np.zeros(self.shape, dtype=self.dtype),
                }
                batch_scales = [None] * len(batch_indices)
                batch_left_paths = []
                batch_gt = []
                for i, idx in enumerate(batch_indices):
                    left_img, scale = self._preprocess_left(self._left_list[idx])
                    right_img, _ = self._preprocess_left(self._right_list[idx])
                    batch_data["left_image"][i] = left_img
                    batch_data["right_image"][i] = right_img
                    batch_scales[i] = scale
                    batch_left_paths.append(self._left_list[idx])
                    if self.evaluation and len(self._gt_list) > 0:
                        batch_gt.append(read_gt_depth(self._gt_list[idx]))
                if self.evaluation and len(self._gt_list) > 0:
                    yield batch_data, batch_left_paths, batch_scales, np.array(batch_gt)
                else:
                    yield batch_data, batch_left_paths, batch_scales
            else:
                # mono
                batch_data = np.zeros(self.shape, dtype=self.dtype)
                batch_scales = [None] * len(batch_indices)
                batch_left_paths = []
                batch_gt = []
                for i, idx in enumerate(batch_indices):
                    left_img, scale = self._preprocess_left(self._left_list[idx])
                    batch_data[i] = left_img
                    batch_scales[i] = scale
                    batch_left_paths.append(self._left_list[idx])
                    if self.evaluation and len(self._gt_list) > 0:
                        batch_gt.append(read_gt_depth(self._gt_list[idx]))
                if self.evaluation and len(self._gt_list) > 0:
                    yield batch_data, batch_left_paths, batch_scales, np.array(batch_gt)
                else:
                    yield batch_data, batch_left_paths, batch_scales
