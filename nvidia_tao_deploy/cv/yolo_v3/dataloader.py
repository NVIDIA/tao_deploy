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

"""YOLOv3 loader."""

import logging
import cv2
import numpy as np
from nvidia_tao_deploy.dataloader.kitti import KITTILoader


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    level="DEBUG")
logger = logging.getLogger(__name__)


def aug_letterbox_resize(img, boxes, num_channels=3, resize_shape=(512, 512)):
    """Apply letter box. resize image to resize_shape, not changing aspect ratio.

    Args:
        img (PIL.Image): RGB image
        boxes (np.array): (N, 4) numpy arrays (xmin, ymin, xmax, ymax) containing bboxes. {x,y}{min,max} is
            in [0, 1] range.
        resize_shape (int, int): (w, h) of new image
    Returns:
        aug_img: img after resize
        aug_boxes: boxes after resize
    """
    img = np.array(img).astype(np.float32)
    if num_channels == 1:
        new_img = np.zeros((resize_shape[1], resize_shape[0]), dtype=float)
    else:
        new_img = np.zeros((resize_shape[1], resize_shape[0], 3), dtype=float)
    new_img += np.mean(img, axis=(0, 1), keepdims=True)
    h, w = img.shape[0], img.shape[1]
    ratio = min(float(resize_shape[1]) / h, float(resize_shape[0]) / w)
    new_h = int(round(ratio * h))
    new_w = int(round(ratio * w))
    l_shift = (resize_shape[0] - new_w) // 2
    t_shift = (resize_shape[1] - new_h) // 2
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)
    new_img[t_shift: t_shift + new_h, l_shift: l_shift + new_w] = img.astype(float)

    xmin = (boxes[:, 0] * new_w + l_shift) / float(resize_shape[0])
    xmax = (boxes[:, 2] * new_w + l_shift) / float(resize_shape[0])
    ymin = (boxes[:, 1] * new_h + t_shift) / float(resize_shape[1])
    ymax = (boxes[:, 3] * new_h + t_shift) / float(resize_shape[1])

    return new_img, np.stack([xmin, ymin, xmax, ymax], axis=-1), \
        [l_shift, t_shift, l_shift + new_w, t_shift + new_h]


class YOLOv3KITTILoader(KITTILoader):
    """YOLOv3 Dataloader."""

    def __init__(self,
                 **kwargs):
        """Init."""
        super().__init__(**kwargs)

        # YOLO series starts label index from 0
        classes = sorted({str(x).lower() for x in self.mapping_dict.values()})
        self.classes = dict(zip(classes, range(len(classes))))
        self.class_mapping = {key.lower(): self.classes[str(val.lower())]
                              for key, val in self.mapping_dict.items()}

    def _filter_invalid_labels(self, labels):
        """filter out invalid labels.

        Arg:
            labels: size (N, 6), where bboxes is normalized to 0~1.
        Returns:
            labels: size (M, 6), filtered bboxes with clipped boxes.
        """
        labels[:, -4:] = np.clip(labels[:, -4:], 0, 1)

        # exclude invalid boxes
        difficult_cond = (labels[:, 1] < 0.5) | (not self.exclude_difficult)
        if np.any(difficult_cond == 0):
            logger.warning(
                "Got label marked as difficult(occlusion > 0), "
                "please set occlusion field in KITTI label to 0 "
                "or set `dataset_config.include_difficult_in_training` to True "
                "in spec file, if you want to include it in training."
            )
        x_cond = labels[:, 4] - labels[:, 2] > 1e-3
        y_cond = labels[:, 5] - labels[:, 3] > 1e-3

        return labels[difficult_cond & x_cond & y_cond]

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
        # change bbox to 0~1
        w, h = image.size
        label[:, 2] /= w
        label[:, 3] /= h
        label[:, 4] /= w
        label[:, 5] /= h

        bboxes = label[:, -4:]

        image, bboxes, _ = aug_letterbox_resize(image,
                                                bboxes,
                                                num_channels=self.num_channels,
                                                resize_shape=(self.width, self.height))
        label[:, -4:] = bboxes

        # Handle Grayscale
        if self.num_channels == 1:
            image = np.expand_dims(image, axis=2)

        # Filter invalid labels
        label = self._filter_invalid_labels(label)

        return image, label
