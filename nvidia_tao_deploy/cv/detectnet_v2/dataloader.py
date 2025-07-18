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

"""DetectNetv2 loader."""

import numpy as np
from PIL import Image

from nvidia_tao_deploy.dataloader.kitti import KITTILoader


class DetectNetKITTILoader(KITTILoader):
    """DetectNetv2 Dataloader."""

    def __init__(self,
                 **kwargs):
        """Init."""
        super().__init__(**kwargs)
        self.image_size = []

    def _load_gt_image(self, image_path):
        """Load GT image from file."""
        img = Image.open(image_path).convert('RGB')
        return img, img.size  # DNv2 requires original image size

    def _get_single_item_raw(self, idx):
        """Load single image and its label.

        Returns:
            image (PIL.image): image object in original resolution
            label (np.array): [class_idx, is_difficult, x_min, y_min, x_max, y_max]
                               with normalized coordinates
        """
        image, image_size = self._load_gt_image(self.image_paths[self.data_inds[idx]])
        if self.is_inference:
            label = np.zeros((1, 6))  # Random array to label
        else:
            label = self._load_gt_label(self.label_paths[self.data_inds[idx]])

        return image, label, image_size

    def _get_single_processed_item(self, idx):
        """Load and process single image and its label."""
        image, label, image_size = self._get_single_item_raw(idx)
        image, label = self.preprocessing(image, label)
        return image, label, image_size

    def __next__(self):
        """Load a full batch."""
        images = []
        labels = []
        image_sizes = []
        if self.n < self.n_batches:
            for idx in range(self.n * self.batch_size,
                             (self.n + 1) * self.batch_size):
                image, label, image_size = self._get_single_processed_item(idx)
                images.append(image)
                labels.append(label)
                image_sizes.append(image_size)
            self.n += 1
            self.image_size.append(image_sizes)
            return self._batch_post_processing(images, labels)
        raise StopIteration

    def preprocessing(self, image, label):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.

        This includes padding, resizing, normalization, data type casting, and transposing.

        Args:
            image (PIL.image): The Pillow image on disk to load.
            label (np.array): labels

        Returns:
            image (np.array): A numpy array holding the image sample, ready to be concatenated
                              into the rest of the batch
            label (np.array): labels
        """
        image = image.resize((self.width, self.height), Image.LANCZOS)
        image = np.asarray(image, dtype=self.dtype) / 255.0

        # Handle Grayscale
        if self.num_channels == 1:
            image = np.expand_dims(image, axis=2)

        image = np.transpose(image, (2, 0, 1))

        # Round
        label = np.round(label, decimals=0)

        # Filter out invalid labels
        label = self._filter_invalid_labels(label)

        return image, label

    def _batch_post_processing(self, images, labels):
        """Post processing for a batch."""
        images = np.array(images)

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

    def _filter_invalid_labels(self, labels):
        """filter out invalid labels.

        Arg:
            labels: size (N, 6).
        Returns:
            labels: size (M, 6), filtered bboxes with clipped boxes.
        """
        return labels
