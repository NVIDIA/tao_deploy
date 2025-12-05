# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Oneformer data loader."""

import numpy as np
from PIL import Image
from panopticapi.utils import rgb2id
from nvidia_tao_deploy.dataloader.coco_panoptic import COCOPanopticLoader


class OneformerDataLoader(COCOPanopticLoader):
    """Oneformer DataLoader for COCO."""

    def __init__(self, ann_path, img_dir, panoptic_dir, shape, dtype, contiguous_id=False, batch_size=1, data_format='channels_last', downsampling_rate=1, eval_samples=None, task="semantic", ignore_index=255):
        super().__init__(ann_path, img_dir, panoptic_dir, shape, dtype, contiguous_id, batch_size, data_format, downsampling_rate, eval_samples)
        self.task = task
        self.ignore_index = ignore_index
        self.n = 0

    def _get_item(self, idx):
        # load image and label
        ann = self.raw_annot['annotations'][idx]
        filename = self.id2img[ann['image_id']]['file_name']
        target_size = (self.width, self.height)
        img = self.get_image(filename, root_dir=self.img_dir,
                             target_size=target_size)
        img = self.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        pan_segm = self.get_mask(ann['file_name'], self.panoptic_dir, mode="RGB",
                                 target_size=target_size)
        pan_segm = rgb2id(pan_segm)

        labels = []
        masks = []
        segm = np.ones_like(pan_segm) * self.ignore_index
        for segment_info in ann["segments_info"]:
            cat_id = segment_info["category_id"]
            if self.contiguous_id:
                cat_id = self.stuff_dataset_id_to_contiguous_id[cat_id]
                cat_id = cat_id - 1
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

        task_text = f"The task is {self.task}"
        return img, segm, task_text

    def __next__(self):
        """Load a full batch."""
        images = []
        segms = []
        tasks = []
        if self.n < self.n_batches:
            for idx in range(self.n * self.batch_size,
                             (self.n + 1) * self.batch_size):
                image, segm, task = self._get_item(idx)
                images.append(image)
                segms.append(segm)
                tasks.append(task)
            images = np.array(images)
            segms = np.array(segms)
            self.n += 1
            return images, segms, tasks
        raise StopIteration
