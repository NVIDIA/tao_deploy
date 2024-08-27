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

"""COCO Panoptic Loader."""

import os
import numpy as np
import json
from PIL import Image, ImageOps
from panopticapi.utils import rgb2id
import logging
logger = logging.getLogger(__name__)


class COCOPanopticLoader:
    """Base Class for COCO Panoptic dataloader"""

    def __init__(self, ann_path, img_dir, panoptic_dir,
                 shape, dtype,
                 contiguous_id=False,
                 batch_size=1, data_format='channels_last',
                 downsampling_rate=1,
                 eval_samples=None):
        """Init.

        Args:
            ann_path (str): annotation file.
            img_dir (str): image directory.
            panoptic_dir (str): segmentation image directory.
            shape (list): shape of the network.
            dtype (str): data type.
            data_format (str): data format (default: channels_last).
            root_dir (str): root directory to where images are located.
            eval_samples (str): total number of samples to evaluate.
        """
        self.batch_size = batch_size
        self.ann_path = ann_path
        self.img_dir = img_dir
        self.panoptic_dir = panoptic_dir
        self.contiguous_id = contiguous_id
        self.segm_downsampling_rate = downsampling_rate

        self.load_json()
        self.get_category_mapping()
        self.n_batches = (len(self.raw_annot['annotations']) + self.batch_size - 1) // self.batch_size
        assert self.n_batches > 0, "empty image dir or batch size too large!"

        if data_format == "channels_last":
            self.height = shape[1]
            self.width = shape[2]
        else:
            self.height = shape[2]
            self.width = shape[3]
        self.dtype = dtype
        self.data_format = data_format

    def load_json(self):
        """Load json annotation file."""
        with open(self.ann_path, 'r', encoding='utf-8') as f:
            self.raw_annot = json.load(f)  # 'images', 'annotations', 'categories'

        self.id2img = {}
        for img in self.raw_annot['images']:
            self.id2img[img["id"]] = img

    def get_category_mapping(self):
        """Map category index in json to 1 based index."""
        self.thing_dataset_id_to_contiguous_id = {}
        self.stuff_dataset_id_to_contiguous_id = {}
        for i, cat in enumerate(self.raw_annot['categories']):
            if cat["isthing"]:
                self.thing_dataset_id_to_contiguous_id[cat["id"]] = i + 1

            # in order to use sem_seg evaluator
            self.stuff_dataset_id_to_contiguous_id[cat["id"]] = i + 1

    def __len__(self):
        """Dataset size."""
        return self.n_batches

    def __iter__(self):
        """Iterate."""
        self.n = 0
        return self

    def __next__(self):
        """Load a full batch."""
        images = []
        segms = []
        if self.n < self.n_batches:
            for idx in range(self.n * self.batch_size,
                             (self.n + 1) * self.batch_size):
                image, segm = self._get_item(idx)
                images.append(image)
                segms.append(segm)
            self.n += 1
            return np.array(images), np.array(segms)
        raise StopIteration

    def _get_item(self, idx):
        # load image and label
        ann = self.raw_annot['annotations'][idx]
        filename = self.id2img[ann['image_id']]['file_name']
        target_size = (self.width, self.height)
        img = self.get_image(filename, root_dir=self.img_dir,
                             target_size=target_size)
        pan_segm = self.get_mask(ann['file_name'], self.panoptic_dir, mode="RGB",
                                 target_size=target_size)
        pan_segm = rgb2id(pan_segm)

        labels = []
        masks = []
        segm = np.zeros_like(pan_segm)
        for segment_info in ann["segments_info"]:
            cat_id = segment_info["category_id"]
            if self.contiguous_id:
                cat_id = self.stuff_dataset_id_to_contiguous_id[cat_id]
            if not segment_info["iscrowd"]:
                labels.append(cat_id)
                masks.append(pan_segm == segment_info["id"])
                segm[pan_segm == segment_info["id"]] = cat_id

        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            segm = np.zeros((self.height // self.segm_downsampling_rate, self.width // self.segm_downsampling_rate))
        else:
            segm = Image.fromarray(segm).resize(
                (self.width // self.segm_downsampling_rate, self.height // self.segm_downsampling_rate),
                resample=Image.NEAREST)
            segm = np.asarray(segm)
        return self.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), segm

    def get_image(self, file_name, root_dir=None, target_size=None):
        """Load image.

        Args:
            file_name (str): relative path to an image file (.png).
        Return:
            image (PIL image): loaded image
        """
        root_dir = root_dir or ""
        image = Image.open(os.path.join(root_dir, file_name)).convert('RGB')
        image = ImageOps.exif_transpose(image)
        if target_size:
            image = image.resize(target_size)
        image = np.array(image)
        return image

    def get_mask(self, file_name, root_dir=None, mode="L", target_size=None):
        """Load mask.

        Args:
            file_name (str): relative path to an image file (.png).
            mode (str): RGB or L
        Return:
            image (PIL image): loaded image
        """
        root_dir = root_dir or ""
        if mode != "L":
            mode = "RGB"
        image = Image.open(os.path.join(root_dir, file_name)).convert(mode)
        if target_size:
            image = image.resize(target_size, resample=Image.NEAREST)
        image = np.array(image)
        return image

    def normalize(self, img, pixel_mean, pixel_std):
        """Normalize input."""
        # 0-255 to 0-1
        img = np.float32(img) / 255.
        img = (img - np.array(pixel_mean)) / np.array(pixel_std)
        img = img.transpose((2, 0, 1))  # [c, h, w]
        return img
