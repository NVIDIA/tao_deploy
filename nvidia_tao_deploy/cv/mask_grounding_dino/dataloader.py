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

"""Mask Grounding DINO DataLoader."""

import cv2
import json
import logging
import numpy as np
from PIL import Image

from nvidia_tao_deploy.dataloader.coco import COCOLoader
from nvidia_tao_deploy.inferencer.preprocess_input import preprocess_input


def resize(image, target, size, max_size=None):
    """Resize image while maintaining aspect ratio.

    Args:
        image: PIL Image or numpy array
        target: Optional target dictionary with boxes
        size: Target size (scalar or (w, h) tuple)
        max_size: Maximum size constraint

    Returns:
        Resized image and updated target
    """
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        """Calculate size with aspect ratio."""
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        """Get target size."""
        if isinstance(size, (list, tuple)):
            return_size = size[::-1]
        else:
            return_size = get_size_with_aspect_ratio(image_size, size, max_size)
        return return_size

    size = get_size(image.size, size, max_size)

    # Use OpenCV for consistency with PyTorch
    rescaled_image = cv2.resize(np.array(image), size, interpolation=cv2.INTER_LINEAR)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.shape[:2][::-1], image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * [ratio_width, ratio_height, ratio_width, ratio_height]
        target["boxes"] = scaled_boxes

    h, w = size
    target["size"] = np.array([h, w])

    return rescaled_image, target


class MGDINOCOCOLoader(COCOLoader):
    """Mask Grounding DINO DataLoader for Visual Grounding and Object Detection tasks."""

    def __init__(
        self,
        val_json_file=None,
        image_std=None,
        img_mean=None,
        **kwargs
    ):
        """Initialize Mask Grounding DINO DataLoader.

        Args:
            val_json_file (str): JSON/JSONL file path for annotations or referring expressions.
            image_std (list): Image standard deviation for normalization.
            img_mean (list): Image mean for normalization.
            **kwargs: Additional arguments passed to COCOLoader.
        """
        # Store JSON file path before calling parent constructor
        self.json_file = val_json_file

        super().__init__(val_json_file=val_json_file, **kwargs)
        self.image_std = image_std
        self.img_mean = img_mean

        # Create a list of (image_id, annotation) pairs for referring expression tasks
        self.image_caption_pairs = []
        self._build_image_caption_pairs()

        # Update n_samples and n_batches to reflect the new sample count
        self.n_samples = len(self.image_caption_pairs)
        self.n_batches = self.n_samples // self.batch_size

    def _build_image_caption_pairs(self):
        """Build a list of samples from JSONL, extracting sent_id for each referring expression."""
        # Read JSONL file directly to get sent_id
        jsonl_file = self.json_file
        if jsonl_file and jsonl_file.lower().endswith('.jsonl'):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())

                        image_id = data.get('image_id')
                        annotations = data.get('annotations', [])
                        sentence_data = data.get('sentence', {})

                        # Extract caption and sent_id from sentence field
                        caption = sentence_data.get('raw', 'object')
                        sent_id = sentence_data.get('sent_id', -1)

                        # For JSONL format, each line is one sample with one referring expression
                        # Store ALL annotations for this (image_id, sent_id) pair
                        self.image_caption_pairs.append({
                            'image_id': image_id,
                            'caption': caption,
                            'annotations': annotations,  # Store all annotations (list)
                            'sent_id': sent_id  # Store sent_id for exact GT matching
                        })
            except Exception as e:
                logging.error("Error reading JSONL file %s: %s", jsonl_file, e)
                # Fallback to COCO API approach
                self._build_image_caption_pairs_fallback()
        else:
            # Use COCO API approach for regular JSON files
            self._build_image_caption_pairs_fallback()

    def _build_image_caption_pairs_fallback(self):
        """Fallback method using COCO API for regular JSON files."""
        for image_id in self.image_ids:
            # Get all annotations for this image using pycocotools
            annotations_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)

            if len(annotations_ids) > 0:
                coco_annotations = self.coco.loadAnns(annotations_ids)

                for ann in coco_annotations:
                    # Extract caption and sent_id from annotation
                    caption = None
                    if 'referring_expression' in ann and ann['referring_expression']:
                        caption = ann['referring_expression']
                    elif 'caption' in ann and ann['caption']:
                        caption = ann['caption']

                    sent_id = ann.get('sent_id', -1)

                    # Create a pair for each annotation (each represents a unique (image_id, sent_id) sample)
                    if caption:
                        self.image_caption_pairs.append({
                            'image_id': image_id,
                            'caption': caption,
                            'annotation': ann,
                            'sent_id': sent_id
                        })
                    else:
                        # If no caption, still add with default
                        self.image_caption_pairs.append({
                            'image_id': image_id,
                            'caption': 'object',
                            'annotation': ann,
                            'sent_id': sent_id
                        })

    def _get_single_processed_item(self, idx):
        """Load and process single (image, caption) pair.

        Args:
            idx: Index of the sample to load

        Returns:
            Tuple of (image, scale, image_id, label, caption, sent_id)
        """
        # Get the image-caption pair data
        pair_data = self.image_caption_pairs[idx]
        image_id = pair_data['image_id']
        caption = pair_data['caption']
        annotation = pair_data.get('annotation')  # Single annotation per sample (for fallback)

        # Find the image index in the original image list
        image_index = self.image_ids.index(image_id)

        # Load image
        gt_image_info, _ = self._load_gt_image(image_index)
        gt_image, gt_scale = gt_image_info

        # Create label data for this specific annotation
        if annotation:
            gt_label = self._create_single_label(image_index, annotation)
        else:
            # For JSONL with multiple annotations, create label from annotations list
            gt_label = self._create_label_from_annotations(image_index, pair_data.get('annotations', []))

        # Get sent_id for exact GT matching (unique identifier for this referring expression)
        sent_id = pair_data.get('sent_id', -1)

        return gt_image, gt_scale, image_id, gt_label, caption, sent_id

    def _create_single_label(self, image_index, annotation):
        """Create label data for a single annotation.

        Args:
            image_index: Index of the image
            annotation: Single annotation dictionary

        Returns:
            Tuple of (label_data, image_info)
        """
        # Get image info
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        h = image_info['height']
        w = image_info['width']

        # Create bounding box in the expected format
        bbox = annotation['bbox']
        # Convert from [x, y, width, height] to [y_min, x_min, y_max, x_max]
        y_min = bbox[1]
        x_min = bbox[0]
        y_max = bbox[1] + bbox[3]
        x_max = bbox[0] + bbox[2]

        # Create single annotation in the format expected by the evaluation
        label_data = np.array([[y_min, x_min, y_max, x_max, 0, -1, self.coco_label_to_label(annotation['category_id'])]])

        return label_data, [self.height, self.width, h, w]

    def _create_label_from_annotations(self, image_index, annotations):
        """Create label data from a list of annotations (JSONL format).

        Args:
            image_index: Index of the image
            annotations: List of annotation dictionaries

        Returns:
            Tuple of (label_data, image_info)
        """
        # Get image info
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        h = image_info['height']
        w = image_info['width']

        # Create label data for all annotations
        label_list = []
        for ann in annotations:
            bbox = ann['bbox']
            # Convert from [x, y, width, height] to [y_min, x_min, y_max, x_max]
            y_min = bbox[1]
            x_min = bbox[0]
            y_max = bbox[1] + bbox[3]
            x_max = bbox[0] + bbox[2]

            label_list.append([y_min, x_min, y_max, x_max, 0, -1, self.coco_label_to_label(ann['category_id'])])

        label_data = np.array(label_list) if label_list else np.array([[0, 0, 0, 0, 0, -1, 0]])

        return label_data, [self.height, self.width, h, w]

    def __next__(self):
        """Load a full batch.

        Returns:
            Tuple of (images, scales, image_ids, labels, captions, sent_ids)
        """
        images = []
        labels = []
        image_ids = []
        scales = []
        captions = []
        sent_ids = []

        if self.n < self.n_batches:
            for idx in range(self.n * self.batch_size,
                             (self.n + 1) * self.batch_size):
                image, scale, image_id, label, caption, sent_id = self._get_single_processed_item(idx)
                images.append(image)
                labels.append(label)
                image_ids.append(image_id)
                scales.append(scale)
                captions.append(caption)
                sent_ids.append(sent_id)
            self.n += 1
            return images, scales, image_ids, labels, captions, sent_ids
        raise StopIteration

    def preprocess_image(self, image_path):
        """Preprocess image for Mask Grounding DINO inference.

        This includes padding, resizing, normalization, data type casting, and transposing.

        Args:
            image_path (str): The path to the image on disk to load.

        Returns:
            image (np.array): A numpy array holding the image sample, ready to be batched
            scale (list): The resize scale used, if any.
        """
        scale = None
        image = Image.open(image_path)
        image = image.convert(mode='RGB')

        image = np.asarray(image, dtype=self.dtype)
        image, _ = resize(image, None, size=(self.height, self.width))

        if self.data_format == "channels_first":
            image = np.transpose(image, (2, 0, 1))

        image = preprocess_input(
            image,
            data_format=self.data_format,
            img_mean=self.img_mean,
            img_std=self.image_std,
            mode='torch'
        )
        return image, scale
