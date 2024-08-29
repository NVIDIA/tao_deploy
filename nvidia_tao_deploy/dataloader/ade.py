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

"""ADE Loader."""

import os
import numpy as np
from PIL import Image, ImageOps
import json
import logging
logger = logging.getLogger(__name__)


class ADELoader:
    """Base Class for ADE dataloader"""

    def __init__(self, gt_list_or_file, shape, dtype,
                 batch_size=1, data_format='channels_last',
                 root_dir=None, eval_samples=None,
                 downsampling_rate=1):
        """Init.

        Args:
            gt_list_or_file (str): validation jsonl file.
            shape (list): shape of the network.
            dtype (str): data type.
            data_format (str): data format (default: channels_last).
            root_dir (str): root directory to where images are located.
            eval_samples (str): total number of samples to evaluate.
        """
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.segm_downsampling_rate = downsampling_rate

        self.parse_input_list(gt_list_or_file)
        self.n_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        assert self.n_batches > 0, "empty image dir or batch size too large!"

        if data_format == "channels_last":
            self.height = shape[1]
            self.width = shape[2]
        else:
            self.height = shape[2]
            self.width = shape[3]
        self.dtype = dtype
        self.data_format = data_format

    def __len__(self):
        """Dataset size."""
        return self.n_batches

    def parse_input_list(self, gt_list_or_file, max_sample=-1, start_idx=-1, end_idx=-1):
        """Parse ADE style annotation file."""
        if isinstance(gt_list_or_file, list):
            self.sample_list = gt_list_or_file
        elif isinstance(gt_list_or_file, str):
            with open(gt_list_or_file, 'r', encoding='utf-8') as f:
                self.sample_list = [json.loads(x.rstrip()) for x in f]
            # self.sample_list = [json.loads(x.rstrip()) for x in open(gt_list_or_file, 'r')]

        if max_sample > 0:
            self.sample_list = self.sample_list[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.sample_list = self.sample_list[start_idx:end_idx]

        self.num_samples = len(self.sample_list)
        assert self.num_samples > 0
        logger.info('Number of samples: {}'.format(self.num_samples))

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

    def _get_item(self, index):
        this_record = self.sample_list[index]
        # load image and label
        image_path = os.path.join(self.root_dir, this_record['img'])
        segm_path = os.path.join(self.root_dir, this_record['segm'])

        img = self.get_image(image_path, target_size=(self.width, self.height))
        segm = self.get_mask(segm_path, target_size=(self.width, self.height))

        cat_ids = np.unique(segm)[1:]  # skip 0
        masks = []
        for i in cat_ids:
            masks.append(segm == i)

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
