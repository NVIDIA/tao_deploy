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

"""COCO-style evaluation metrics.

Implements the interface of COCO API and metric_fn in tf.TPUEstimator.

COCO API: github.com/cocodataset/cocoapi/
"""

import logging

from nvidia_tao_deploy.metrics.coco_metric import EvaluationMetric, MaskCOCO


class MaskGDinoCOCO(MaskCOCO):
    """COCO object for mask evaluation."""

    def load_predictions(self,
                         detection_results,
                         include_mask,
                         is_image_mask=False):
        """Create prediction dictionary list from detection and mask results.

        Args:
            detection_results: a dictionary containing numpy arrays which corresponds
                to prediction results.
            include_mask: a boolean, whether to include mask in detection results.
            is_image_mask: a boolean, where the predict mask is a whole image mask.

        Returns:
            a list of dictionary including different prediction results from the model
                in numpy form.
        """
        predictions = []
        num_detections = detection_results['detection_scores'].size
        current_index = 0
        for i, image_id in enumerate(detection_results['source_id']):

            if include_mask:
                encoded_masks = detection_results['detection_masks']

            for box_index in range(int(detection_results['num_detections'][i])):
                if current_index % 1000 == 0:
                    logging.info('%s/%s', current_index, num_detections)

                current_index += 1

                prediction = {'image_id': int(image_id),
                              'bbox': detection_results['detection_boxes'][i][box_index].tolist(),
                              'score': detection_results['detection_scores'][i][box_index],
                              'category_id': int(
                                  detection_results['detection_classes'][i][box_index])}

                if include_mask:
                    prediction['segmentation'] = encoded_masks[box_index]

                predictions.append(prediction)

        return predictions


class MaskGDinoEvaluationMetric(EvaluationMetric):
    """COCO evaluation metric class."""

    def __init__(self, filename, include_mask, eval_class_ids=None):
        """Constructs COCO evaluation class.

        The class provides the interface to metrics_fn in TPUEstimator. The
        _evaluate() loads a JSON file in COCO annotation format as the
        groundtruths and runs COCO evaluation.

        Args:
            filename (str): Ground truth JSON file name. If filename is None, use
                groundtruth data passed from the dataloader for evaluation.
            include_mask (bool): boolean to indicate whether or not to include mask eval.
            eval_class_ids (list): class ids to evaluate on.
        """
        self.filename = filename
        self.coco_gt = MaskGDinoCOCO(self.filename)
        self.metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                             'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
        self._include_mask = include_mask
        if self._include_mask:
            mask_metric_names = ['mask_' + x for x in self.metric_names]
            self.metric_names.extend(mask_metric_names)
        self.eval_class_ids = eval_class_ids

        self._reset()

    def _reset(self):
        """Reset COCO API object."""
        if self.filename is None and not hasattr(self, 'coco_gt'):
            self.coco_gt = MaskGDinoCOCO()
