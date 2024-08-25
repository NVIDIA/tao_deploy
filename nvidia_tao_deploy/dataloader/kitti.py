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

"""KITTI loader."""

from abc import ABC, abstractmethod
import logging
import os
import numpy as np
from PIL import Image

from nvidia_tao_deploy.cv.common.constants import VALID_IMAGE_EXTENSIONS


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    level="DEBUG")
logger = logging.getLogger(__name__)


class KITTILoader(ABC):
    """Base Class for KITTI dataloader"""

    def __init__(self,
                 shape,
                 image_dirs,
                 label_dirs,
                 mapping_dict,
                 exclude_difficult=True,
                 batch_size=10,
                 is_inference=False,
                 image_mean=None,
                 data_format="channels_first",
                 image_depth=8,
                 dtype=None):
        """Init.

        Args:
            shape (list): list of input dimension that is either (c, h, w) or (h, w, c) format.
            image_dirs (list): list of image directories.
            label_dirs (list): list of label directories.
            mapping_dict (dict): class mapping. e.g. {'Person': 'person', 'Crowd': 'person'}
            exclude_difficult (bool): whether to include difficult samples.
            batch_size (int): size of the batch.
            is_inference (bool): If True, no labels will be returned
            image_mean (list): image mean used for preprocessing.
            data_format (str): Data format of the input. (Default: channels_first)
            image_depth(int): Bit depth of images(8 or 16).
            dtype (str): data type to cast to
        """
        assert len(image_dirs) == len(label_dirs), "Mismatch in the length of image and label dirs!"
        self.image_paths = []
        self.label_paths = []

        self.is_inference = is_inference
        self.dtype = dtype

        # mapping class to 1-based integer
        self.mapping_dict = mapping_dict
        classes = sorted({str(x).lower() for x in mapping_dict.values()})
        self.classes = dict(zip(classes, range(1, len(classes) + 1)))
        self.class_mapping = {key.lower(): self.classes[str(val.lower())]
                              for key, val in mapping_dict.items()}

        # use numpy array to accelerate
        self._add_source(image_dirs, label_dirs)
        self.image_paths = np.array(self.image_paths)
        self.label_paths = np.array(self.label_paths)
        self.data_inds = np.arange(len(self.image_paths))

        self.batch_size = batch_size
        if data_format == "channels_first":
            self.num_channels, self.height, self.width = shape
        else:
            self.height, self.width, self.num_channels = shape
        self.image_depth = image_depth

        self.exclude_difficult = exclude_difficult

        self.image_mean = image_mean
        self.n_samples = len(self.data_inds)
        self.n_batches = int(len(self.image_paths) // self.batch_size)
        assert self.n_batches > 0, "empty image dir or batch size too large!"

    def _add_source(self, image_folders, label_folders):
        """Add Kitti sources."""
        for image_folder, label_folder in zip(image_folders, label_folders):
            img_paths = os.listdir(image_folder)
            if not self.is_inference:
                label_paths = set(os.listdir(label_folder))
            else:
                label_paths = []

            for img_path in img_paths:
                # Only add valid items to paths
                filename, _ = os.path.splitext(img_path)
                if img_path.lower().endswith(VALID_IMAGE_EXTENSIONS) and filename + '.txt' in label_paths:
                    self.image_paths.append(os.path.join(image_folder, img_path))
                    self.label_paths.append(os.path.join(label_folder, filename + '.txt'))
                elif img_path.lower().endswith(VALID_IMAGE_EXTENSIONS) and self.is_inference:
                    self.image_paths.append(os.path.join(image_folder, img_path))

    def __len__(self):
        """Get length of Sequence."""
        return self.n_batches

    def _load_gt_image(self, image_path):
        """Load GT image from file."""
        img = Image.open(image_path)
        if self.num_channels == 3:
            img = img.convert('RGB')  # Color Image
        else:
            if self.image_depth == 16:
                img = img.convert('I')  # PIL int32 mode for 16-bit images
            else:
                img = img.convert('L')  # Grayscale Image
        return img

    def _load_gt_label(self, label_path):
        """Load Kitti labels.

        Returns:
            [class_idx, is_difficult, x_min, y_min, x_max, y_max]
        """
        with open(label_path, 'r', encoding="utf-8") as f:
            entries = f.read().strip().split('\n')
        results = []
        for entry in entries:
            items = entry.strip().split()
            if len(items) < 9:
                continue
            items[0] = items[0].lower()
            if items[0] not in self.class_mapping:
                continue
            label = [self.class_mapping[items[0]], 1 if int(
                items[2]) != 0 else 0, *items[4:8]]
            results.append([float(x) for x in label])

        return np.array(results).reshape(-1, 6)

    def _filter_invalid_labels(self, labels):
        """filter out invalid labels.

        Arg:
            labels: size (N, 6).
        Returns:
            labels: size (M, 6), filtered bboxes with clipped boxes.
        """
        x_coords = labels[:, [-4, -2]]
        x_coords = np.clip(x_coords, 0, self.width - 1)
        labels[:, [-4, -2]] = x_coords
        y_coords = labels[:, [-3, -1]]
        y_coords = np.clip(y_coords, 0, self.height - 1)
        labels[:, [-3, -1]] = y_coords

        # exclude invalid boxes
        x_cond = labels[:, -2] - labels[:, -4] > 1e-3
        y_cond = labels[:, -1] - labels[:, -3] > 1e-3

        return labels[x_cond & y_cond]

    def _get_single_item_raw(self, idx):
        """Load single image and its label.

        Returns:
            image (PIL.image): image object in original resolution
            label (np.array): [class_idx, is_difficult, x_min, y_min, x_max, y_max]
                               with normalized coordinates
        """
        image = self._load_gt_image(self.image_paths[self.data_inds[idx]])
        if self.is_inference:
            label = np.zeros((1, 6))  # Random array to label
        else:
            label = self._load_gt_label(self.label_paths[self.data_inds[idx]])

        return image, label

    @abstractmethod
    def preprocessing(self, image, label):
        """Perform preprocessing on image and label."""
        pass

    def _get_single_processed_item(self, idx):
        """Load and process single image and its label."""
        image, label = self._get_single_item_raw(idx)
        image, label = self.preprocessing(image, label)
        return image, label

    def _batch_post_processing(self, images, labels):
        """Post processing for a batch."""
        images = np.array(images)
        # For num_channels=1 case, we assume that additional dimension was created in preprocessing.

        if self.num_channels == 3:
            images = images[..., [2, 1, 0]]  # RGB -> BGR

        images = images.transpose(0, 3, 1, 2)  # channels_last -> channels_first

        if self.num_channels == 3:
            if self.image_mean:
                bb, gg, rr = self.image_mean
            else:
                bb, gg, rr = 103.939, 116.779, 123.68
            # subtract imagenet mean
            images -= np.array([[[[bb]], [[gg]], [[rr]]]])

        else:
            if self.image_mean:
                bb = self.image_mean  # grayscale only contains one value
            elif self.image_depth == 8:
                bb = 117.3786
            elif self.image_depth == 16:
                # 117.3786 * 256
                bb = 30048.9216
            else:
                raise ValueError(
                    f"Unsupported image depth: {self.image_depth}, should be 8 or 16, "
                    "please check `augmentation_config.output_depth` in spec file"
                )

            # subtract imagenet mean
            images -= np.array([[[bb]]])

        # try to make labels a numpy array
        is_make_array = True
        x_shape = None
        for x in labels:
            if not isinstance(x, np.ndarray):
                is_make_array = False
                break
            if x_shape is None:
                x_shape = x.shape
            elif x_shape != x.shape:
                is_make_array = False
                break

        if is_make_array:
            labels = np.array(labels)

        return images, labels

    def __iter__(self):
        """Iterate."""
        self.n = 0
        return self

    def __next__(self):
        """Load a full batch."""
        images = []
        labels = []
        if self.n < self.n_batches:
            for idx in range(self.n * self.batch_size,
                             (self.n + 1) * self.batch_size):
                image, label = self._get_single_processed_item(idx)
                images.append(image)
                labels.append(label)
            self.n += 1

            return self._batch_post_processing(images, labels)
        raise StopIteration
