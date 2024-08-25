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

"""Mask2former COCO loader."""

import numpy as np
from PIL import Image
import pycocotools.mask as maskUtils

from nvidia_tao_deploy.dataloader.coco import COCOLoader


class Mask2formerCOCOLoader(COCOLoader):

    """Mask2former COCO DataLoader."""

    def __init__(self, val_json_file, shape, dtype, batch_size=1, data_format='channels_last', image_dir=None, eval_samples=None):
        super().__init__(val_json_file, shape, dtype, batch_size, data_format, image_dir, eval_samples)
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]

    def preprocess_image(self, image_path):
        """The image preprocessor loads an image from disk and prepares it as needed for batching.

        This includes padding, resizing, normalization, data type casting, and transposing.
        This Image Batcher implements one algorithm for now:
        * EfficientDet: Resizes and pads the image to fit the input size.

        Args:
            image_path(str): The path to the image on disk to load.

        Returns:
            image (np.array): A numpy array holding the image sample, ready to be concatenated
                              into the rest of the batch
            scale (list): the resize scale used, if any.
        """
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.width, self.height))
        image = np.array(image)

        # Normalize input image.
        # 0-255 to 0-1
        image = np.float32(image) / 255.
        image = (image - np.array(self.pixel_mean)) / np.array(self.pixel_std)
        image = image.transpose((2, 0, 1))  # [c, h, w]
        return image, -1

    def _get_single_processed_item(self, idx):
        """Load and process single image and its label."""
        gt_image_info, _ = self._load_gt_image(idx)
        gt_image, _ = gt_image_info
        gt_label = self._load_gt_label(idx)
        return gt_image, gt_label

    def _load_gt_label(self, image_index):
        """Get GT annotations."""
        image_info = image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        annot_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        target_size = (self.width, self.height)

        h = image_info['height']
        w = image_info['width']
        segm = np.zeros((h, w))
        for i in annot_ids:
            ann = self.coco.loadAnns(ids=i)[0]
            if isinstance(ann['segmentation'], list):
                rles = maskUtils.frPyObjects(ann['segmentation'], h, w)
                rle = maskUtils.merge(rles)
            elif 'counts' in ann['segmentation']:
                # e.g. {'counts': [6, 1, 40, 4, 5, 4, 5, 4, 21], 'size': [9, 10]}
                if isinstance(ann['segmentation']['counts'], list):
                    rle = maskUtils.frPyObjects(ann['segmentation'], h, w)
                else:
                    rle = ann['segmentation']
            else:
                raise ValueError('Please check the segmentation format.')
            mask = np.ascontiguousarray(maskUtils.decode(rle))
            assert len(mask.shape) == 2
            mask = mask.astype(np.uint8)
            segm[mask == 1] = ann["category_id"]

        if target_size:
            image = Image.fromarray(segm)
            image = image.resize(target_size, resample=Image.NEAREST)
            segm = np.array(image)
        return segm

    def __next__(self):
        """Load a full batch."""
        images = []
        segms = []
        if self.n < self.n_batches:
            for idx in range(self.n * self.batch_size,
                             (self.n + 1) * self.batch_size):
                image, segm = self._get_single_processed_item(idx)
                images.append(image)
                segms.append(segm)
            images = np.array(images)
            segms = np.array(segms)
            self.n += 1
            return images, segms
        raise StopIteration
