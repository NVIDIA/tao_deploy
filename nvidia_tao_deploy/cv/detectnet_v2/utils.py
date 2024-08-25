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

"""Utilities file containing helper functions to post process raw predictions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
from collections import namedtuple
from sklearn.cluster import DBSCAN as dbscan

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


Detection = namedtuple('Detection', [
    # Bounding box of the detection in the LTRB format: [left, top, right, bottom]
    'bbox',
    # Confidence of detection
    'confidence',
    # Weighted variance of the bounding boxes in this cluster, normalized for the size of the box
    'cluster_cv',
    # Number of raw bounding boxes that went into this cluster
    'num_raw_boxes',
    # Sum of of the raw bounding boxes' coverage values in this cluster
    'sum_coverages',
    # Maximum coverage value among bboxes
    'max_cov_value',
    # Minimum converage value among bboxes
    'min_cov_value',
    # Candidate coverages.
    'candidate_covs',
    # Candidate bbox coordinates.
    'candidate_bboxes'
])


def cluster_bboxes(raw_detections,
                   criterion,
                   eps,
                   min_samples,
                   min_weight,
                   min_height,
                   confidence_model,
                   cluster_weights=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                   clustering_algorithm="dbscan",
                   confidence_threshold=0.01,
                   nms_iou_threshold=0.01,
                   nms_confidence_threshold=0.1):
    """Cluster the bboxes from the raw feature map to output boxes.

    It works in two steps.
    1. Obtain grid cell indices where coverage > min_weight.
    2. Make a list of all bboxes from the grid cells short listed from 1.
    3. Cluster this list of bboxes using a density based clustering algorithm..

    Inputs:
        raw_detections : dict with keys:
          bbox: rectangle coordinates, (num_imgs, 4, W, H) array
          cov: weights for the rectangles, (num_imgs, 1, W, H) array
        criterion (str): clustering criterion ('MSE' or 'IOU')
        eps (float): threshold for considering two rectangles as one
        min_samples (int): minimum cumulative weight for output rectangle
        min_weight (float): filter out bboxes with weight smaller than
          min_weight prior to clustering
        min_height (float): filter out bboxes with height smaller than
          min_height after clustering
        cluster_weights (dict): weighting of different distance components
                        (bbox, depth, orientation, bottom_vis, width_vis, orient_markers)
        confidence_model (str): Dict of {kind: 'mlp' or 'aggregate_cov'
                                  model: the expected model format or None}
        clustering_algorithm (str): Algorithm used to cluster the raw predictions.
        confidence_threshold (float): The final overlay threshold post clustering.
        nms_iou_threshold (float): IOU overlap threshold to be used when running NMS.

    Returns:
        detections_per_image (list): A list of lists of Detection objects, one list
          for each input frame.
    """
    db = None
    if clustering_algorithm in ["dbscan", "hybrid"]:
        db = setup_dbscan_object(eps, min_samples, criterion)

    num_images = len(raw_detections['cov'])

    if confidence_model == 'mlp':
        raise NotImplementedError("MLP confidence thresholding not supported.")

    # Initialize output detections to empty lists
    # DO NOT DO a=[[]]*num_images --> that means 'a[0] is a[1] == True' !!!
    detections_per_image = [[] for _ in range(num_images)]

    # Needed when doing keras confidence model.
    # keras.backend.get_session().run(tf.initialize_all_variables())

    # Loop images
    logger.debug("Clustering bboxes")

    for image_idx in range(num_images):
        # Check if the input was empty.
        if raw_detections['cov'][image_idx].size == 0:
            continue
        bboxes, covs = threshold_bboxes(raw_detections, image_idx, min_weight)
        if covs.size == 0:
            continue
        # Cluster using DBSCAN.
        if clustering_algorithm == "dbscan":
            logger.debug("Clustering bboxes using dbscan.")
            clustered_boxes_per_image = cluster_with_dbscan(bboxes,
                                                            covs,
                                                            criterion,
                                                            db,
                                                            confidence_model,
                                                            cluster_weights,
                                                            min_height,
                                                            threshold=confidence_threshold)
        # Clustering detections with NMS.
        elif clustering_algorithm == "nms":
            logger.debug("Clustering using NMS")
            clustered_boxes_per_image = cluster_with_nms(bboxes, covs,
                                                         min_height,
                                                         nms_iou_threshold=nms_iou_threshold,
                                                         threshold=nms_confidence_threshold)
        elif clustering_algorithm == "hybrid":
            logger.debug("Clustering with DBSCAN + NMS")
            clustered_boxes_per_image = cluster_with_hybrid(
                bboxes, covs,
                criterion, db,
                confidence_model,
                cluster_weights,
                min_height,
                nms_iou_threshold=nms_iou_threshold,
                confidence_threshold=confidence_threshold,
                nms_confidence_threshold=nms_confidence_threshold
            )
        else:
            raise NotImplementedError(f"Clustering with {clustering_algorithm} algorithm not supported.")
        detections_per_image[image_idx].extend(clustered_boxes_per_image)

    # Sort in descending order of cumulative weight
    detections_per_image = [sorted(dets, key=lambda det: -det.confidence)
                            for dets in detections_per_image]

    return detections_per_image


def get_keep_indices(dets, covs, min_height,
                     Nt=0.3, sigma=0.4, threshold=0.01,
                     method=4):
    """Perform NMS over raw detections.

    This function implements clustering using multiple variants of NMS, namely,
    Linear, Soft-NMS, D-NMS and NMS. It computes the indexes of the raw detections
    that may be preserved post NMS.

    Args:
        dets (np.ndarray): Array of filtered bboxes.
        scores (np.ndarray): Array of filtered scores (coverages).
        min_height (int): Minimum height of the boxes to be retained.
        Nt (float): Overlap threshold.
        sigma (float): Variance using in the Gaussian soft nms.
        threshold (float): Filtering threshold post bbox clustering.
        method (int): Variant of nms to be used.

    Returns:
        keep (np.ndarray): Array of indices of boxes to be retained after clustering.
    """
    N = dets.shape[0]
    assert len(dets.shape) == 2 and dets.shape[1] == 4, \
        f"BBox dimensions are invalid, {dets.shape}."
    indexes = np.array([np.arange(N)])
    assert len(covs.shape) == 1 and covs.shape[0] == N, \
        f"Coverage dimensions are invalid. {covs.shape}"

    # Convert to t, l, b, r representation for NMS.
    l, t, r, b = dets.T
    dets = np.asarray([t, l, b, r]).T
    dets = np.concatenate((dets, indexes.T), axis=1)
    scores = covs

    # Compute box areas.
    areas = (r - l + 1) * (b - t + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1
        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)
        # min_overlap_box
        x1c = np.minimum(dets[i, 1], dets[pos:, 1])
        y1c = np.minimum(dets[i, 0], dets[pos:, 0])
        x2c = np.maximum(dets[i, 3], dets[pos:, 3])
        y2c = np.maximum(dets[i, 2], dets[pos:, 2])

        c1x, c1y = (dets[i, 1] + dets[i, 3]) / 2.0, (dets[i, 0] + dets[i, 2]) / 2.0
        c2x, c2y = (dets[pos:, 1] + dets[pos:, 3]) / 2.0, (dets[pos:, 0] + dets[pos:, 2]) / 2.0

        centre_dis = ((c1x - c2x) ** 2) + ((c1y - c2y) ** 2)
        diag = ((x1c - x2c) ** 2) + ((y1c - y2c) ** 2)

        ovr_dis = ovr - centre_dis / diag

        # Four methods: 1.linear 2.gaussian soft NMS 3. D-NMS 4.original NMS
        if method == 1:
            # linear NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            # Gaussian Soft NMS
            weight = np.exp(-(ovr * ovr) / sigma)
        elif method == 3:
            # D-NMS
            weight = np.ones(ovr.shape)
            weight[ovr_dis > Nt] = 0
        elif method == 4:
            # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0
        else:
            raise NotImplementedError("NMS variants can only be between [1 - 4] where \n"
                                      "1. linear NMS\n2. Gaussian Soft NMS\n3. D-NMS\n4. "
                                      "Original NMS")
        scores[pos:] = weight * scores[pos:]

    # Filtering based on confidence threshold.
    inds = dets[:, 4][scores > threshold]
    keep = inds.astype(int)
    keep = np.array([[f] for f in keep])
    return keep


def cluster_with_nms(bboxes, covs, min_height,
                     nms_iou_threshold=0.01,
                     threshold=0.01):
    """Cluster raw detections with NMS.

    Args:
        bboxes (np.ndarray): The raw bbox predictions from the network.
        covs (np.ndarray): The raw coverage predictions from the network.
        min_height (float): The minimum height to filter out bboxes post clustering.
        nms_iou_threshold (float): The overlap threshold to be used when running NMS.
        threshold (float): The final confidence threshold to filter out bboxes
            after clustering.

    Returns:
        clustered_boxes_per_images (list): List of clustered and filtered detections.
    """
    keep_indices = get_keep_indices(bboxes, covs, min_height,
                                    threshold=threshold,
                                    Nt=nms_iou_threshold)
    logger.debug("Keep indices: shape: %s, type: %s", keep_indices.shape, type(keep_indices))
    if keep_indices.size == 0:
        return []
    filterred_boxes = np.take_along_axis(bboxes, keep_indices, axis=0)
    filterred_coverages = covs[keep_indices]
    assert (filterred_boxes.shape[0] == filterred_coverages.shape[0])
    clustered_boxes_per_image = []
    for idx in range(len(filterred_boxes)):
        clustered_boxes_per_image.append(Detection(
            bbox=filterred_boxes[idx, :],
            confidence=filterred_coverages[idx][0],
            cluster_cv=None,
            num_raw_boxes=None,
            sum_coverages=None,
            max_cov_value=None,
            min_cov_value=None,
            candidate_covs=filterred_coverages[idx],
            candidate_bboxes=filterred_boxes[idx]))
    return clustered_boxes_per_image


def cluster_with_dbscan(bboxes, covs, criterion, db,
                        confidence_model, cluster_weights,
                        min_height, threshold=0.01):
    """Cluster raw predictions using dbscan.

    Args:
        boxes (np.array): Thresholded raw bbox blob
        covs (np.array): Thresholded raw covs blob
        criterion (str): DBSCAN clustering criterion.
        db: Instantiated dbscan object.
        cluster_weights (dict): weighting of different distance components
            (bbox, depth, orientation, bottom_vis, width_vis, orient_markers)
        min_height (float): filter out bboxes with height smaller than
            min_height after clustering
        threshold (float): Final threshold to filter bboxes post
            clustering.

    Returns:
        detections_per_image.
    """
    detections_per_image = []
    if criterion[:3] in ['MSE', 'IOU']:
        if criterion[:3] == 'MSE':
            data = bboxes
            labeling = db.fit_predict(X=data, sample_weight=covs)
        elif criterion[:3] == 'IOU':
            pairwise_dist = \
                cluster_weights[0] * (1.0 - iou_vectorized(bboxes))
            labeling = db.fit_predict(X=pairwise_dist, sample_weight=covs)
        labels = np.unique(labeling[labeling >= 0])  # pylint: disable=possibly-used-before-assignment
        logger.debug("Number of boxes: %d", len(labels))
        for label in labels:
            w = covs[labeling == label]
            aggregated_w = np.sum(w)
            w_norm = w / aggregated_w
            n = len(w)
            w_max = np.max(w)
            w_min = np.min(w)

            # Mean bounding box
            b = bboxes[labeling == label]
            mean_bbox = np.sum((b.T * w_norm).T, axis=0)

            # Compute coefficient of variation of the box coords
            bbox_area = (mean_bbox[2] - mean_bbox[0]) * (mean_bbox[3] - mean_bbox[1])

            # Calculate weighted bounding box variance normalized by
            # bounding box size
            cluster_cv = np.sum(w_norm.reshape((-1, 1)) * (b - mean_bbox) ** 2, axis=0)

            cluster_cv = np.sqrt(np.mean(cluster_cv) / bbox_area)

            # Update confidence output based on mode of confidence.
            if confidence_model == 'aggregate_cov':
                confidence = aggregated_w
            elif confidence_model == 'mean_cov':
                w_mean = aggregated_w / n
                confidence = (w_mean - w_min) / (w_max - w_min)
            else:
                raise NotImplementedError(f"Unknown confidence kind {confidence_model.kind}!")

            # Filter out too small bboxes
            if mean_bbox[3] - mean_bbox[1] <= min_height:
                continue

            if confidence >= threshold:
                detections_per_image += [Detection(
                    bbox=mean_bbox,
                    confidence=confidence,
                    cluster_cv=cluster_cv,
                    num_raw_boxes=n,
                    sum_coverages=aggregated_w,
                    max_cov_value=w_max,
                    min_cov_value=w_min,
                    candidate_covs=w,
                    candidate_bboxes=b
                )]
        return detections_per_image
    raise NotImplementedError(f"DBSCAN for this criterion is not implemented. {criterion}")


def threshold_bboxes(raw_detections, image_idx, min_weight):
    """Threshold raw predictions based on coverages.

    Args:
        raw_detections (dict): Dictionary containing raw detections, cov
            and bboxes.

    Returns:
        bboxes, covs: The filtered numpy array of bboxes and covs.
    """
    # Get bbox coverage values, flatten (discard spatial and scale info)
    covs = raw_detections['cov'][image_idx].flatten()
    valid_indices = covs > min_weight
    covs = covs[valid_indices]

    # Flatten last three dimensions (discard spatial and scale info)
    # assume bbox is always present
    bboxes = raw_detections['bbox'][image_idx]
    bboxes = bboxes.reshape(bboxes.shape[:1] + (-1,)).T[valid_indices]
    return bboxes, covs


def setup_dbscan_object(eps, min_samples, criterion):
    """Instantiate dbscan object for clustering predictions with dbscan.

    Args:
        eps (float): DBSCAN epsilon value (search distance parameter)
        min_samples (int): minimum cumulative weight for output rectangle
        criterion (str): clustering criterion ('MSE' or 'IOU')

    Returns:
        db (dbscan object): DBSCAN object from scikit learn.
    """
    min_samples = max(int(min_samples), 1)
    if criterion[:3] == 'MSE':
        # MSE between coordinates is used as the distance
        # If depth and orientation are included, add them as
        # additional coordinates
        db = dbscan(eps=eps, min_samples=min_samples)
    elif criterion[:3] == 'IOU':
        # 1.-IOU is used as distance between bboxes
        # For depth and orientation, use a normalized difference
        # measure
        # The final distance metric is a weighted sum of the above
        db = dbscan(eps=eps, min_samples=min_samples, metric='precomputed')
    else:
        raise NotImplementedError("cluster_bboxes: Unknown bbox clustering criterion!")
    return db


def cluster_with_hybrid(bboxes, covs,
                        criterion, db,
                        confidence_model,
                        cluster_weights,
                        min_height,
                        nms_iou_threshold=0.3,
                        confidence_threshold=0.1,
                        nms_confidence_threshold=0.1):
    """Cluster raw predictions with DBSCAN + NMS.

    Args:
        boxes (np.array): Thresholded raw bbox blob
        covs (np.array): Thresholded raw covs blob
        criterion (str): DBSCAN clustering criterion.
        db: Instantiated dbscan object.
        cluster_weights (dict): weighting of different distance components
            (bbox, depth, orientation, bottom_vis, width_vis, orient_markers)
        min_height (float): filter out bboxes with height smaller than
            min_height after clustering
        nms_iou_threshold (float): The overlap threshold to be used when running NMS.
        confiedence_threshold (float): The confidence threshold to filter out bboxes
            after clustering by dbscan.
        nms_confidence_threshold (float): The confidence threshold to filter out bboxes
            after clustering by NMS.

    Returns:
        nms_clustered_detection_per_image (list): List of clustered detections
            after hybrid clustering.
    """
    dbscan_clustered_detections_per_image = cluster_with_dbscan(
        bboxes,
        covs,
        criterion,
        db,
        confidence_model,
        cluster_weights,
        min_height,
        threshold=confidence_threshold
    )

    # Extract raw detections from clustered outputs.
    nms_candidate_boxes = []
    nms_candidate_covs = []
    for detections in dbscan_clustered_detections_per_image:
        nms_candidate_boxes.extend(detections.candidate_bboxes)
        nms_candidate_covs.extend(detections.candidate_covs)
    nms_candidate_boxes = np.asarray(nms_candidate_boxes).astype(np.float32)
    nms_candidate_covs = np.asarray(nms_candidate_covs).astype(np.float32)

    if nms_candidate_covs.size == 0:
        return []
    # Clustered candidates from dbscan to run NMS.
    nms_clustered_detections_per_image = cluster_with_nms(
        nms_candidate_boxes,
        nms_candidate_covs,
        min_height,
        nms_iou_threshold=nms_iou_threshold,
        threshold=nms_confidence_threshold
    )
    return nms_clustered_detections_per_image


def iou_vectorized(rects):
    """Intersection over union among a list of rectangles in LTRB format.

    Args:
        rects (np.array) : numpy array of shape (N, 4), LTRB format, assumes L<R and T<B
    Returns::
        d (np.array) : numpy array of shape (N, N) of the IOU between all pairs of rects
    """
    # coordinates
    l, t, r, b = rects.T

    # form intersection coordinates
    isect_l = np.maximum(l[:, None], l[None, :])
    isect_t = np.maximum(t[:, None], t[None, :])
    isect_r = np.minimum(r[:, None], r[None, :])
    isect_b = np.minimum(b[:, None], b[None, :])

    # form intersection area
    isect_w = np.maximum(0, isect_r - isect_l)
    isect_h = np.maximum(0, isect_b - isect_t)
    area_isect = isect_w * isect_h

    # original rect areas
    areas = (r - l) * (b - t)

    # Union area is area_a + area_b - intersection area
    denom = (areas[:, None] + areas[None, :] - area_isect)

    # Return IOU regularized with .01, to avoid outputing NaN in pathological
    # cases (area_a = area_b = isect = 0)
    return area_isect / (denom + .01)
