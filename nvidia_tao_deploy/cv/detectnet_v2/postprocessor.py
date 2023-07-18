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

"""Post processing handler for TLT DetectNet_v2 models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import operator
from time import time

import numpy as np

from six.moves import range
from nvidia_tao_deploy.cv.detectnet_v2.utils import cluster_bboxes


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)

criterion = 'IOU'
scales = [(1.0, 'cc')]

CLUSTERING_ALGORITHM = {
    0: "dbscan",
    1: "nms",
    2: "hybrid",
    "dbscan": "dbscan",
    "nms": "nms",
    "hybrid": "hybrid",
}


class BboxHandler(object):
    """Class to handle bbox output from the inference script."""

    def __init__(self, spec=None, **kwargs):
        """Setting up Bbox handler."""
        self.spec = spec
        self.cluster_params = {}
        self.frame_height = kwargs.get('frame_height', 544)
        self.frame_width = kwargs.get('frame_width', 960)
        self.bbox_normalizer = kwargs.get('bbox_normalizer', 35)
        self.bbox = kwargs.get('bbox', 'ltrb')
        self.cluster_params = kwargs.get('cluster_params', None)
        self.classwise_cluster_params = kwargs.get("classwise_cluster_params", None)
        self.bbox_norm = (self.bbox_normalizer, ) * 2
        self.stride = kwargs.get("stride", 16)
        self.train_img_size = kwargs.get('train_img_size', None)
        self.save_kitti = kwargs.get('save_kitti', True)
        self.image_overlay = kwargs.get('image_overlay', True)
        self.extract_crops = kwargs.get('extract_crops', True)
        self.target_classes = kwargs.get('target_classes', None)
        self.bbox_offset = kwargs.get("bbox_offset", 0.5)
        self.clustering_criterion = kwargs.get("criterion", "IOU")
        self.postproc_classes = kwargs.get('postproc_classes', self.target_classes)
        confidence_threshold = {}
        nms_confidence_threshold = {}

        for key, value in list(self.classwise_cluster_params.items()):
            confidence_threshold[key] = value.clustering_config.dbscan_confidence_threshold
            if value.clustering_config.nms_confidence_threshold:
                nms_confidence_threshold[key] = value.clustering_config.nms_confidence_threshold

        self.state = {
            'scales': scales,
            'display_classes': self.target_classes,
            'min_height': 0,
            'criterion': criterion,
            'confidence_th': {'car': 0.9, 'person': 0.1, 'truck': 0.1},
            'nms_confidence_th': {'car': 0.9, 'person': 0.1, 'truck': 0.1},
            'cluster_weights': (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        }

        self.state['confidence_th'] = confidence_threshold
        self.state['nms_confidence_th'] = nms_confidence_threshold

    def bbox_preprocessing(self, input_cluster):
        """Function to perform inplace manipulation of prediction dicts before clustering.

        Args:
            input_cluster (Dict): prediction dictionary of output cov and bbox per class.

        Returns:
            input_cluster (Dict): shape manipulated prediction dictionary.
        """
        for classes in self.target_classes:
            input_cluster[classes]['bbox'] = self.abs_bbox_converter(input_cluster[classes]
                                                                     ['bbox'])
            # Stack predictions
            for keys in list(input_cluster[classes].keys()):
                if 'bbox' in keys:
                    input_cluster[classes][keys] = \
                        input_cluster[classes][keys][np.newaxis, :, :, :, :]
                    input_cluster[classes][keys] = \
                        np.asarray(input_cluster[classes][keys]).transpose((1, 2, 3, 4, 0))
                elif 'cov' in keys:
                    input_cluster[classes][keys] = input_cluster[classes][keys][np.newaxis,
                                                                                np.newaxis,
                                                                                :, :, :]
                    input_cluster[classes][keys] = \
                        np.asarray(input_cluster[classes][keys]).transpose((2, 1, 3, 4, 0))

        return input_cluster

    def abs_bbox_converter(self, bbox):
        """Convert the raw grid cell corrdinates to image space coordinates.

        Args:
            bbox (np.array): BBox coordinates blob per batch with shape [n, 4, h, w].
        Returns:
            bbox (np.array): BBox coordinates reconstructed from grid cell based coordinates
              with the same dimensions.
        """
        target_shape = bbox.shape[-2:]

        # Define grid cell centers
        gc_centers = [(np.arange(s) * self.stride + self.bbox_offset) for s in target_shape]
        gc_centers = [s / n for s, n in zip(gc_centers, self.bbox_norm)]

        # Mapping cluster output
        if self.bbox == 'arxy':
            assert not self.train_img_size, \
                "ARXY bbox format needs same train and inference image shapes."
            # reverse mapping of abs bbox to arxy
            area = (bbox[:, 0, :, :] / 10.) ** 2.
            width = np.sqrt(area * bbox[:, 1, :, :])
            height = np.sqrt(area / bbox[:, 1, :, :])
            cen_x = width * bbox[:, 2, :, :] + gc_centers[0][:, np.newaxis]
            cen_y = height * bbox[:, 3, :, :] + gc_centers[1]
            bbox[:, 0, :, :] = cen_x - width / 2.
            bbox[:, 1, :, :] = cen_y - height / 2.
            bbox[:, 2, :, :] = cen_x + width / 2.
            bbox[:, 3, :, :] = cen_y + height / 2.
            bbox[:, 0, :, :] *= self.bbox_norm[0]
            bbox[:, 1, :, :] *= self.bbox_norm[1]
            bbox[:, 2, :, :] *= self.bbox_norm[0]
            bbox[:, 3, :, :] *= self.bbox_norm[1]
        elif self.bbox == 'ltrb':
            # Convert relative LTRB bboxes to absolute bboxes inplace.
            # Input bbox in format (image, bbox_value,
            # grid_cell_x, grid_cell_y).
            # Ouput bboxes given in pixel coordinates in the source resolution.
            if not self.train_img_size:
                self.train_img_size = self.bbox_norm
            # Compute scalers that allow using different resolution in
            # inference and training
            scale_w = self.bbox_norm[0] / self.train_img_size[0]
            scale_h = self.bbox_norm[1] / self.train_img_size[1]
            bbox[:, 0, :, :] -= gc_centers[0][:, np.newaxis] * scale_w
            bbox[:, 1, :, :] -= gc_centers[1] * scale_h
            bbox[:, 2, :, :] += gc_centers[0][:, np.newaxis] * scale_w
            bbox[:, 3, :, :] += gc_centers[1] * scale_h
            bbox[:, 0, :, :] *= -self.train_img_size[0]
            bbox[:, 1, :, :] *= -self.train_img_size[1]
            bbox[:, 2, :, :] *= self.train_img_size[0]
            bbox[:, 3, :, :] *= self.train_img_size[1]
        return bbox

    def cluster_detections(self, preds):
        """Cluster detections and filter based on confidence.

        Also determines false positives and missed detections based on the
        clustered detections.

        Args:
            - spec: The experiment spec
            - preds: Raw predictions, a Dict of Dicts
            - state: The DetectNet_v2 viz state

        Returns:
            - classwise_detections (NamedTuple): DBSCan clustered detections.
        """
        # Cluster
        classwise_detections = {}
        clustering_time = 0.
        for object_type in preds:
            start_time = time()
            if object_type not in list(self.classwise_cluster_params.keys()):
                logger.info("Object type %s not defined in cluster file. Falling back to default"
                            "values", object_type)
                buffer_type = "default"
                if buffer_type not in list(self.classwise_cluster_params.keys()):
                    raise ValueError("If the class-wise cluster params for an object isn't "
                                     "there then please mention a default class.")
            else:
                buffer_type = object_type
            logger.debug("Clustering bboxes %s", buffer_type)
            classwise_params = self.classwise_cluster_params[buffer_type]
            clustering_config = classwise_params.clustering_config
            clustering_algorithm = CLUSTERING_ALGORITHM[clustering_config.clustering_algorithm]
            # clustering_algorithm = clustering_config.clustering_algorithm
            nms_iou_threshold = 0.3
            if clustering_config.nms_iou_threshold:
                nms_iou_threshold = clustering_config.nms_iou_threshold
            confidence_threshold = self.state['confidence_th'].get(buffer_type, 0.1)
            nms_confidence_threshold = self.state['nms_confidence_th'].get(buffer_type, 0.1)
            detections = cluster_bboxes(preds[object_type],
                                        criterion=self.clustering_criterion,
                                        eps=classwise_params.clustering_config.dbscan_eps + 1e-12,
                                        min_samples=clustering_config.dbscan_min_samples,
                                        min_weight=clustering_config.coverage_threshold,
                                        min_height=clustering_config.minimum_bounding_box_height,
                                        confidence_model='aggregate_cov',  # TODO: Not hardcode (evaluation doesn't have this option)
                                        cluster_weights=self.state['cluster_weights'],
                                        confidence_threshold=confidence_threshold,
                                        clustering_algorithm=clustering_algorithm,
                                        nms_iou_threshold=nms_iou_threshold,
                                        nms_confidence_threshold=nms_confidence_threshold)

            clustering_time += (time() - start_time) / len(preds)
            classwise_detections[object_type] = detections

        return classwise_detections

    def postprocess(self, class_wise_detections, batch_size, image_size, resized_size, classes):
        """Reformat classwise detections into a single 2d-array for metric calculation.

        Args:
            class_wise_detections (dict): detections per target classes
            batch_size (int): size of batches to process
            image_size (list): list of tuples containing original image size
            resized_size (tuple): a tuple containing the model input size
            classes (list): list containing target classes in the correct order
        """
        results = []

        for image_idx in range(batch_size):
            frame_width, frame_height = resized_size
            scaling_factor = tuple(map(operator.truediv, image_size[image_idx], resized_size))
            per_image_results = []
            for key in class_wise_detections.keys():
                bbox_list, confidence_list = _get_bbox_and_confs(class_wise_detections[key][image_idx],
                                                                 scaling_factor,
                                                                 key,
                                                                 "aggregate_cov",  # TODO: Not hard-code
                                                                 frame_height,
                                                                 frame_width)
                bbox_list = np.array(bbox_list)
                cls_id = [classes[key]] * len(bbox_list)
                if not len(bbox_list):
                    continue
                # each row: cls_id, conf, x1, y1, x2, y2
                result = np.stack((cls_id, confidence_list, bbox_list[:, 0], bbox_list[:, 1], bbox_list[:, 2], bbox_list[:, 3]), axis=-1)
                per_image_results.append(result)

            # Handle cases when there was no detections made at all
            if len(per_image_results) == 0:
                null_prediction = np.zeros((1, 6))
                null_prediction[0][0] = 1  # Dummy class which will be filtered out later.
                results.append(null_prediction)
                logger.debug("Adding null predictions as there were no predictions made.")
                continue

            per_image_results = np.concatenate(per_image_results)
            results.append(per_image_results)
        return results


def _get_bbox_and_confs(classwise_detections, scaling_factor,
                        key, confidence_model, frame_height,
                        frame_width):
    """Simple function to get bbox and confidence formatted list."""
    bbox_list = []
    confidence_list = []
    for i in range(len(classwise_detections)):
        bbox_object = classwise_detections[i]
        coords_scaled = _scale_bbox(bbox_object.bbox, scaling_factor,
                                    frame_height, frame_width)

        if confidence_model == 'mlp':
            confidence = bbox_object.confidence[0]
        else:
            confidence = bbox_object.confidence
        bbox_list.append(coords_scaled)
        confidence_list.append(confidence)
    return bbox_list, confidence_list


def _scale_bbox(bbox, scaling_factor, frame_height, frame_width):
    """Scale bbox coordinates back to original image dimensions.

    Args:
        bbox (list): bbox coordinates ltrb
        scaling factor (float): input_image size/model inference size

    Returns:
        bbox_scaled (list): list of scaled coordinates
    """
    # Clipping and clamping coordinates.
    x1 = min(max(0.0, float(bbox[0])), frame_width)
    y1 = min(max(0.0, float(bbox[1])), frame_height)
    x2 = max(min(float(bbox[2]), frame_width), x1)
    y2 = max(min(float(bbox[3]), frame_height), y1)

    # Rescaling center.
    hx, hy = x2 - x1, y2 - y1
    cx = x1 + hx / 2
    cy = y1 + hy / 2

    # Rescaling height, width
    nx, ny = cx * scaling_factor[0], cy * scaling_factor[1]
    nhx, nhy = hx * scaling_factor[0], hy * scaling_factor[1]

    # Final bbox coordinates.
    nx1, nx2 = nx - nhx / 2, nx + nhx / 2
    ny1, ny2 = ny - nhy / 2, ny + nhy / 2

    # Stacked coordinates.
    bbox_scaled = np.asarray([nx1, ny1, nx2, ny2])
    return bbox_scaled
