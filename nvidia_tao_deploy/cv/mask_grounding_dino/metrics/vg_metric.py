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

"""Custom evaluation metric for Visual Grounding (VG) tasks.

This metric evaluates VG performance using merged mask IoU rather than
traditional COCO instance-based evaluation.
"""

import logging
import numpy as np
import cv2
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO


class VGEvaluationMetric:
    """Visual Grounding evaluation metric using merged mask IoU."""

    def __init__(self, gt_json_file, iou_threshold=0.5):
        """
        Initialize VG evaluation metric.

        Args:
            gt_json_file (str): Path to ground truth JSON file
            iou_threshold (float): IoU threshold for success detection
        """
        self.gt_coco = COCO(gt_json_file)
        self.iou_threshold = iou_threshold
        self.image_results = []

    def _get_annotation_by_image_and_sent_id(self, image_id, sent_id):
        """
        Get ALL annotations matching both image_id and sent_id with pycocotools API.

        Args:
            image_id (int): COCO image ID
            sent_id (int): Sentence ID from referring expression

        Returns:
            list: List of all matching annotations (can be multiple for merged GT)
        """
        matching_annotations = []

        try:
            # Use pycocotools API to get all annotation IDs for this image
            ann_ids = self.gt_coco.getAnnIds(imgIds=[image_id])

            if ann_ids:
                # Load all annotations for this image
                annotations = self.gt_coco.loadAnns(ann_ids)

                # Collect ALL annotations with matching sent_id
                for ann in annotations:
                    if ann.get('sent_id') == sent_id:
                        matching_annotations.append(ann)  # Collect all matches!

        except Exception as e:
            logging.warning("Error in annotation lookup for image_id=%d, sent_id=%d: %s", image_id, sent_id, e)

        return matching_annotations

    def add_predictions(self, image_id, gt_masks, pred_masks, pred_phrases, pred_scores, target_phrase, annotation_id=None):
        """
        Add predictions for a single sample (image + phrase pair).

        Args:
            image_id (int): COCO image ID
            gt_masks (list): List of GT mask RLE encodings or binary masks for this specific phrase
            pred_masks (list): List of predicted mask RLE encodings or binary masks
            pred_phrases (list): List of predicted phrases
            pred_scores (list): List of prediction confidence scores
            target_phrase (str): Target referring expression for this sample
            annotation_id (int, optional): Specific annotation ID for this phrase
        """
        result = {
            'image_id': image_id,
            'target_phrase': target_phrase,
            'annotation_id': annotation_id,
            'gt_masks': gt_masks,
            'pred_masks': pred_masks,
            'pred_phrases': pred_phrases,
            'pred_scores': pred_scores
        }
        self.image_results.append(result)

    def _get_matching_annotation(self, image_id, target_phrase):
        """
        Get the specific annotation that matches both image_id and target_phrase.

        Args:
            image_id (int): COCO image ID
            target_phrase (str): Target referring expression

        Returns:
            dict or None: Matching annotation dict, or None if not found
        """
        # Get all annotations for this image
        ann_ids = self.gt_coco.getAnnIds(imgIds=[image_id])
        anns = self.gt_coco.loadAnns(ann_ids)

        # Find annotation that matches the target phrase
        for ann in anns:
            # Check different possible phrase fields
            ann_phrase = None
            if 'referring_expression' in ann:
                ann_phrase = ann['referring_expression']
            elif 'caption' in ann:
                ann_phrase = ann['caption']
            elif 'phrase' in ann:
                ann_phrase = ann['phrase']

            # Match phrases (exact match or substring match)
            if ann_phrase and self._phrases_match(target_phrase, ann_phrase):
                return ann

        # If no exact match found, return the first annotation as fallback
        # This handles cases where phrase matching is imperfect
        if anns:
            logging.warning("No exact phrase match found for '%s' in image %d. Using first annotation as fallback.", target_phrase, image_id)
            return anns[0]

        return None

    def _phrases_match(self, phrase1, phrase2):
        """Check if two phrases match (exact or substring match).

        Args:
            phrase1 (str): First phrase
            phrase2 (str): Second phrase

        Returns:
            bool: True if phrases match
        """
        if not phrase1 or not phrase2:
            return False
        phrase1_lower = phrase1.lower().strip()
        phrase2_lower = phrase2.lower().strip()
        return phrase1_lower == phrase2_lower or phrase1_lower in phrase2_lower or phrase2_lower in phrase1_lower

    def _decode_masks(self, masks, image_shape):
        """Decode various mask formats to binary arrays."""
        if not masks:
            return np.zeros(image_shape, dtype=bool)

        decoded_masks = []

        # Handle nested lists (flatten first)
        flat_masks = []
        for mask in masks:
            if isinstance(mask, list):
                flat_masks.extend(mask)
            else:
                flat_masks.append(mask)

        for mask in flat_masks:
            decoded = None

            if isinstance(mask, dict) and 'counts' in mask:
                # RLE encoded mask from maskUtils
                try:
                    decoded = maskUtils.decode(mask).astype(bool)
                except Exception:
                    continue
            elif isinstance(mask, np.ndarray):
                # Already binary/float mask
                if mask.dtype == bool:
                    decoded = mask
                else:
                    decoded = (mask > 0.5).astype(bool)
            elif isinstance(mask, list):
                # Polygon format - convert to RLE first
                try:
                    if len(mask) > 0 and isinstance(mask[0], list):
                        # Multiple polygons: [[x1,y1,x2,y2,...], [...]]
                        rle = maskUtils.frPyObjects(mask, image_shape[0], image_shape[1])
                        if isinstance(rle, list):
                            # Multiple polygons
                            merged_rle = maskUtils.merge(rle)
                            decoded = maskUtils.decode(merged_rle).astype(bool)
                        else:
                            # Single RLE
                            decoded = maskUtils.decode(rle).astype(bool)
                    elif len(mask) >= 6 and all(isinstance(x, (int, float)) for x in mask):
                        # Single polygon: [x1,y1,x2,y2,...]
                        rle = maskUtils.frPyObjects([mask], image_shape[0], image_shape[1])
                        decoded = maskUtils.decode(rle).astype(bool)
                except Exception as e:
                    logging.debug("Failed to decode polygon mask: %s", e)
                    continue
            elif isinstance(mask, str):
                # Skip string masks as they're complex to parse (RLE format)
                logging.warning("Skipping mask with string datatype (RLE format not currently handled)")
                continue

            if decoded is not None:
                # Ensure correct shape
                if decoded.shape != image_shape:
                    logging.debug("Resizing mask from %s to %s", decoded.shape, image_shape)
                    decoded = cv2.resize(decoded.astype(np.uint8),
                                         (image_shape[1], image_shape[0]),
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
                decoded_masks.append(decoded)
            else:
                logging.debug("Could not decode mask of type: %s", type(mask))

        if decoded_masks:
            # Merge all masks using logical OR (union)
            merged_mask = np.logical_or.reduce(decoded_masks)
        else:
            merged_mask = np.zeros(image_shape, dtype=bool)

        return merged_mask

    def _calculate_iou(self, mask1, mask2):
        """Calculate IoU between two binary masks."""
        if mask1.shape != mask2.shape:
            logging.warning("Mask shape mismatch: %s vs %s", mask1.shape, mask2.shape)
            return 0.0

        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        iou = intersection / union
        return float(iou)

    def evaluate(self):
        """
        Evaluate all predictions and return comprehensive metrics.

        Returns:
            dict: Dictionary containing comprehensive evaluation metrics
        """
        if not self.image_results:
            logging.warning("No predictions to evaluate!")
            return {
                'mean_iou': 0.0,
                'accuracy': 0.0,
                'overall_iou': 0.0,
                'num_images': 0,
                'iou_std': 0.0,
                'gIoU': 0.0,
                'cIoU': 0.0,
                'mAP50': 0.0,
                'mAP': 0.0,
                'T_acc': 0.0,
                'N_acc': 0.0,
                'Pr@0.7': 0.0,
                'Pr@0.8': 0.0,
                'Pr@0.9': 0.0
            }

        ious = []
        accuracies = []
        total_gt_pixels = 0
        total_pred_pixels = 0
        total_intersection_pixels = 0
        total_union_pixels = 0

        # Statistics for comprehensive metrics
        total_count = len(self.image_results)
        accum_IoU = 0.0
        accum_I = 0  # Accumulated intersection
        accum_U = 0  # Accumulated union
        not_empty_count = 0  # Count of non-empty predictions

        # Precision at different thresholds
        pr_thresholds = [0.7, 0.8, 0.9]
        pr_count = {thresh: 0 for thresh in pr_thresholds}

        # Confusion matrix statistics
        nt_stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

        # For mAP calculation
        ap_scores = []
        iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.5 to 0.95 step 0.05

        for result in self.image_results:
            # Get image info
            image_info = self.gt_coco.loadImgs([result['image_id']])[0]
            image_shape = (image_info['height'], image_info['width'])

            # Decode and merge GT masks
            gt_merged = self._decode_masks(result['gt_masks'], image_shape)

            # Decode and merge predicted masks
            pred_merged = self._decode_masks(result['pred_masks'], image_shape)

            # Calculate IoU
            iou = self._calculate_iou(gt_merged, pred_merged)
            ious.append(iou)

            # Calculate accuracy (IoU > threshold)
            is_correct = iou >= self.iou_threshold
            accuracies.append(is_correct)

            # Accumulate pixel statistics
            gt_pixels = gt_merged.sum()
            pred_pixels = pred_merged.sum()
            intersection_pixels = np.logical_and(gt_merged, pred_merged).sum()
            union_pixels = np.logical_or(gt_merged, pred_merged).sum()

            total_gt_pixels += gt_pixels
            total_pred_pixels += pred_pixels
            total_intersection_pixels += intersection_pixels
            total_union_pixels += union_pixels

            # Accumulate for gIoU and cIoU
            accum_IoU += iou
            accum_I += intersection_pixels
            accum_U += union_pixels

            # Count non-empty predictions
            if pred_pixels > 0:
                not_empty_count += 1

            # Precision at different thresholds
            for thresh in pr_thresholds:
                if iou >= thresh:
                    pr_count[thresh] += 1

            # Confusion matrix statistics (binary classification: correct vs incorrect)
            has_gt = gt_pixels > 0
            has_pred = pred_pixels > 0

            if has_gt and has_pred and iou >= self.iou_threshold:
                nt_stats["TP"] += 1  # True Positive: correct detection
            elif has_gt and (not has_pred or iou < self.iou_threshold):
                nt_stats["FN"] += 1  # False Negative: missed detection
            elif not has_gt and has_pred:
                nt_stats["FP"] += 1  # False Positive: false detection
            else:  # not has_gt and not has_pred
                nt_stats["TN"] += 1  # True Negative: correct non-detection

            # For mAP: calculate precision at different IoU thresholds
            sample_aps = []
            for thresh in iou_thresholds:
                if iou >= thresh:
                    sample_aps.append(1.0)  # True positive
                else:
                    sample_aps.append(0.0)  # False positive
            ap_scores.append(sample_aps)

        # Calculate overall metrics
        mean_iou = np.mean(ious)
        accuracy = np.mean(accuracies)

        # Overall pixel-level IoU across all images
        if total_union_pixels > 0:
            overall_iou = total_intersection_pixels / total_union_pixels
        else:
            overall_iou = 1.0 if total_intersection_pixels == 0 else 0.0

        # Calculate requested comprehensive metrics
        gIoU = 100.0 * accum_IoU / total_count
        cIoU = 100.0 * accum_I / max(accum_U, 1)

        # Calculate mAP and mAP50
        if ap_scores:
            ap_scores = np.array(ap_scores)
            map_val = np.mean(ap_scores)  # mAP averaged over all thresholds
            map50 = np.mean(ap_scores[:, 0])  # mAP at IoU=0.5 (first threshold)
        else:
            map_val = 0.0
            map50 = 0.0

        # Calculate accuracy metrics
        T_acc = 100 * nt_stats["TN"] / max(nt_stats["TN"] + nt_stats["FP"], 1)  # True Negative Rate
        N_acc = 100 * nt_stats["TP"] / max(nt_stats["TP"] + nt_stats["FN"], 1)  # True Positive Rate (Recall)

        # Precision at thresholds (over non-empty predictions)
        pr_at_07 = 100.0 * pr_count[0.7] / max(not_empty_count, 1)
        pr_at_08 = 100.0 * pr_count[0.8] / max(not_empty_count, 1)
        pr_at_09 = 100.0 * pr_count[0.9] / max(not_empty_count, 1)

        # IoU distribution
        iou_bins = [0.5, 0.6, 0.7, 0.8, 0.9]
        iou_dist = {f"IoU@{thresh}": np.mean([iou >= thresh for iou in ious])
                    for thresh in iou_bins}

        # Final comprehensive results (exactly as requested)
        final_results = {
            # Original metrics
            'mean_iou': mean_iou,
            'accuracy': accuracy,
            'overall_iou': overall_iou,
            'num_images': len(self.image_results),
            'iou_std': np.std(ious),

            # Requested comprehensive metrics (matching exact format)
            "gIoU": gIoU,                                                    # e.g., 87.3%
            "cIoU": cIoU,                                                    # e.g., 85.1%
            "mAP50": 100.0 * map50,                                          # e.g., 72.5%
            "mAP": 100.0 * map_val,                                          # e.g., 45.8%
            "T_acc": T_acc,                                                  # e.g., 91.3%
            "N_acc": N_acc,                                                  # e.g., 88.7%
            "Pr@0.7": pr_at_07,                                              # e.g., 68.4%
            "Pr@0.8": pr_at_08,                                              # e.g., 55.2%
            "Pr@0.9": pr_at_09,                                              # e.g., 31.6%

            # Additional distribution metrics
            **iou_dist,

            # Debug information
            'confusion_matrix': nt_stats,
            'not_empty_predictions': not_empty_count
        }

        return final_results

    def predict_metric_fn(self, predictions_dict, target_phrases=None, sent_ids=None):
        """
        Interface compatible with existing evaluation pipeline.

        Args:
            predictions_dict (dict): Dictionary with prediction arrays
            target_phrases (list, optional): List of target phrases for each sample
            sent_ids (list, optional): List of sentence IDs for exact GT matching

        Returns:
            dict: Evaluation metrics
        """
        # Extract predictions - note: these are lists of variable-length lists for VG
        detection_classes = predictions_dict['detection_classes']  # List of phrase lists
        detection_scores = predictions_dict['detection_scores']    # List of score lists
        # detection_boxes not used in VG metric (only masks are evaluated)
        detection_masks = predictions_dict['detection_masks']      # List of RLE dict lists
        source_ids = predictions_dict['source_id']                 # List of image IDs

        # Process each batch/image
        for i in range(len(source_ids)):
            # Extract single image ID (source_id is list of single-element lists for batch_size=1)
            image_id = source_ids[i][0] if isinstance(source_ids[i], list) else source_ids[i]

            # Get predictions for this image (variable length lists)
            image_pred_masks = detection_masks[i] if i < len(detection_masks) else []
            image_pred_phrases = detection_classes[i] if i < len(detection_classes) else []
            image_pred_scores = detection_scores[i] if i < len(detection_scores) else []

            # Get the sent_id and target phrase for this specific sample
            sent_id = None
            target_phrase = "N/A"

            if sent_ids and i < len(sent_ids):
                sent_id = sent_ids[i]

            if target_phrases and i < len(target_phrases):
                target_phrase = target_phrases[i]

            # Use both image_id and sent_id for exact GT lookup with pycocotools API
            if sent_id and sent_id != -1:
                matching_annotations = self._get_annotation_by_image_and_sent_id(image_id, sent_id)

                if matching_annotations:
                    # Extract GT masks from ALL matching annotations for merged evaluation
                    gt_masks = []
                    first_ann = matching_annotations[0]  # Use first annotation for metadata

                    for ann in matching_annotations:
                        if 'segmentation' in ann and ann['segmentation']:
                            gt_masks.append(ann['segmentation'])

                    # Update target phrase from first annotation if not provided
                    if target_phrase == "N/A":
                        if 'referring_expression' in first_ann:
                            target_phrase = first_ann['referring_expression']
                        elif 'caption' in first_ann:
                            target_phrase = first_ann['caption']

                    # Add predictions for this specific sample (with potentially multiple GT masks)
                    self.add_predictions(
                        image_id=image_id,
                        gt_masks=gt_masks,  # List of all GT masks for merged evaluation
                        pred_masks=image_pred_masks,
                        pred_phrases=image_pred_phrases,
                        pred_scores=image_pred_scores,
                        target_phrase=target_phrase,
                        annotation_id=first_ann.get('id')  # Use first annotation ID for reference
                    )
                else:
                    logging.warning("No annotation found for image_id %d with sent_id %s", image_id, sent_id)
            else:
                # Fallback: use phrase matching (less reliable)
                logging.info("No sent_id provided for sample %d, falling back to phrase matching", i)
                matching_ann = self._get_matching_annotation(image_id, target_phrase)

                if matching_ann:
                    # Extract GT masks only for this specific annotation
                    gt_masks = []
                    if 'segmentation' in matching_ann and matching_ann['segmentation']:
                        gt_masks = [matching_ann['segmentation']]

                    # Add predictions for this specific sample
                    self.add_predictions(
                        image_id=image_id,
                        gt_masks=gt_masks,
                        pred_masks=image_pred_masks,
                        pred_phrases=image_pred_phrases,
                        pred_scores=image_pred_scores,
                        target_phrase=target_phrase,
                        annotation_id=matching_ann.get('id')
                    )
                else:
                    logging.warning("No matching annotation found for image %d with phrase '%s'", image_id, target_phrase)

        # Evaluate and return metrics
        return self.evaluate()


def create_vg_evaluator(gt_json_file, iou_threshold=0.5):
    """Factory function to create VG evaluator."""
    return VGEvaluationMetric(gt_json_file, iou_threshold)
