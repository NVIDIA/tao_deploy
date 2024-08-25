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

"""COCO Loader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod
import os
import numpy as np

from pycocotools.coco import COCO


class COCOLoader(ABC):
    """Base Class for COCO dataloader"""

    def __init__(self, val_json_file, shape, dtype, batch_size=1, data_format='channels_last', image_dir=None, eval_samples=None):
        """Init.

        Args:
            val_json_file (str): validation json file.
            shape (list): shape of the network.
            dtype (str): data type.
            data_format (str): data format (default: channels_last).
            image_dir (str): directory where images are located.
            eval_samples (str): total number of samples to evaluate.
        """
        self.image_dir = image_dir
        self.coco = COCO(val_json_file)
        self.image_ids = self.coco.getImgIds()
        self.n_samples = eval_samples or len(self.image_ids)
        self.batch_size = batch_size
        self.n_batches = self.n_samples // self.batch_size
        assert self.n_batches > 0, "empty image dir or batch size too large!"

        self.load_classes()
        if data_format == "channels_last":
            self.height = shape[1]
            self.width = shape[2]
        else:
            self.height = shape[2]
            self.width = shape[3]
        self.dtype = dtype
        self.data_format = data_format

    @abstractmethod
    def preprocess_image(self, image_path):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.

        This may include padding, resizing, normalization, data type casting, and transposing.
        """
        pass

    def load_classes(self):
        """create class mapping."""
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes) + 1
            self.classes[c['name']] = len(self.classes) + 1

    def coco_label_to_label(self, coco_label):
        """coco label to label mapping."""
        return self.coco_labels_inverse[coco_label]

    def _load_gt_image(self, image_index):
        """Load image."""
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        if os.path.exists(image_info['file_name']):
            # Read absolute path from the annotation
            path = image_info['file_name']
        else:
            # Override the root directory to user provided image_dir
            path = os.path.join(self.image_dir, image_info['file_name'])
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image path {path} does not exist!")
        return self.preprocess_image(path), image_info['id']

    def _load_gt_label(self, image_index):
        """Load COCO labels.

        Returns:
            [class_idx, is_difficult, x_min, y_min, x_max, y_max]
            where is_diffcult is hardcoded to 0 in the current COCO GT labels.
        """
        # get image info
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        h = image_info['height']
        w = image_info['width']
        # image_id = image_info['id']

        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = {
            'labels': np.empty((0,)),
            'bboxes': np.empty((0, 4)),
            'masks': [],
        }

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            # return empty annotations
            return np.empty((0, 6)), [self.height, self.width, h, w]

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for _, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotations['labels'] = np.concatenate(
                [annotations['labels'], [self.coco_label_to_label(a['category_id'])]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][1],
                a['bbox'][0],
                a['bbox'][1] + a['bbox'][3],
                a['bbox'][0] + a['bbox'][2],
            ]]], axis=0)

        labels = np.expand_dims(annotations['labels'], axis=-1)
        return np.concatenate(
            (annotations['bboxes'], np.full_like(labels, 0), np.full_like(labels, -1), labels),
            axis=1), [self.height, self.width, h, w]

    def _get_single_processed_item(self, idx):
        """Load and process single image and its label."""
        gt_image_info, image_id = self._load_gt_image(idx)
        gt_image, gt_scale = gt_image_info
        gt_label = self._load_gt_label(idx)
        return gt_image, gt_scale, image_id, gt_label

    def __iter__(self):
        """Iterate."""
        self.n = 0
        return self

    def __next__(self):
        """Load a full batch."""
        images = []
        labels = []
        image_ids = []
        scales = []
        if self.n < self.n_batches:
            for idx in range(self.n * self.batch_size,
                             (self.n + 1) * self.batch_size):
                image, scale, image_id, label = self._get_single_processed_item(idx)
                images.append(image)
                labels.append(label)
                image_ids.append(image_id)
                scales.append(scale)
            self.n += 1
            return images, scales, image_ids, labels
        raise StopIteration

    def __len__(self):
        """Return length."""
        return int(np.ceil(self.n_samples / self.batch_size))
