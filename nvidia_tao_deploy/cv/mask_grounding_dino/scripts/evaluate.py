# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Standalone TensorRT evaluation."""

import os
import operator
import copy
import logging
import json
import six
import numpy as np
import tensorrt as trt
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import pycocotools.mask as maskUtils

from nvidia_tao_core.config.mask_grounding_dino.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.deformable_detr.dataloader import DDETRCOCOLoader
from nvidia_tao_deploy.cv.grounding_dino.utils import tokenize_captions

from nvidia_tao_deploy.cv.mask_grounding_dino.inferencer import MaskGDINOInferencer
from nvidia_tao_deploy.cv.mask_grounding_dino.metrics.coco_metric import MaskGDinoEvaluationMetric
from nvidia_tao_deploy.cv.mask_grounding_dino.utils import post_process

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name='grounding_dino', mode='evaluate')
def main(cfg: ExperimentConfig) -> None:
    """Mask Grounding DINO TRT evaluation."""
    if not os.path.exists(cfg.evaluate.trt_engine):
        raise FileNotFoundError(f"Provided evaluate.trt_engine at {cfg.evaluate.trt_engine} does not exist!")

    assert cfg.dataset.batch_size == 1, "Evaluation batch size must be 1."
    max_text_len = cfg.model.max_text_len
    num_select = cfg.model.num_select
    conf_threshold = cfg.evaluate.conf_threshold
    eval_metric = MaskGDinoEvaluationMetric(
        cfg.dataset.test_data_sources.json_file,
        include_mask=True)
    trt_infer = MaskGDINOInferencer(
        cfg.evaluate.trt_engine,
        batch_size=cfg.dataset.batch_size,
        num_classes=max_text_len)

    c, h, w = trt_infer.input_tensors[0].shape

    dl = DDETRCOCOLoader(
        val_json_file=cfg.dataset.test_data_sources.json_file,
        shape=(cfg.dataset.batch_size, c, h, w),
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
        batch_size=cfg.dataset.batch_size,
        data_format="channels_first",
        image_std=cfg.dataset.augmentation.input_std,
        image_dir=cfg.dataset.test_data_sources.image_dir,
        eval_samples=None)

    category_dict = dl.coco.loadCats(dl.coco.getCatIds())
    cat_lists = [item['name'] for item in category_dict]
    captions = [" . ".join(cat_lists) + ' .'] * cfg.dataset.batch_size

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder_type)
    input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map = \
        tokenize_captions(tokenizer, cat_lists, captions, max_text_len)

    predictions = {
        'detection_scores': [],
        'detection_boxes': [],
        'detection_classes': [],
        'detection_masks': [],
        'source_id': [],
        'image_info': [],
        'num_detections': []
    }

    def evaluation_preds(preds):
        # Essential to avoid modifying the source dict
        _preds = copy.deepcopy(preds)
        for k, _ in six.iteritems(_preds):
            _preds[k] = np.concatenate(_preds[k], axis=0)
        eval_results = eval_metric.predict_metric_fn(_preds)
        return eval_results

    for imgs, scale, source_id, labels in tqdm(dl, total=len(dl), desc="Producing predictions"):
        image = np.array(imgs)

        image_info = []
        target_sizes = []
        for i, label in enumerate(labels):
            image_info.append([label[-1][0], label[-1][1], scale[i], label[-1][2], label[-1][3]])
            # target_sizes needs to [W, H, W, H]
            target_sizes.append([label[-1][3], label[-1][2], label[-1][3], label[-1][2]])
        image_info = np.array(image_info)
        inputs = (image, input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks)
        pred_logits, pred_boxes, pred_masks = trt_infer.infer(inputs)

        class_labels, scores, boxes, masks = post_process(pred_logits, pred_boxes, pred_masks, target_sizes, pos_map, num_select)
        segments = masks[0] > conf_threshold
        segments = segments.transpose((2, 0, 1))
        # Convert the mask to uint8 and then to fortranarray for RLE encoder.
        encoded_masks = [
            maskUtils.encode(np.asfortranarray(instance_mask.astype(np.uint8)))
            for instance_mask in segments]

        # Convert to xywh
        boxes[:, :, 2:] -= boxes[:, :, :2]

        predictions['detection_classes'].append(class_labels)
        predictions['detection_scores'].append(scores)
        predictions['detection_boxes'].append(boxes)
        predictions['detection_masks'].append(encoded_masks)
        predictions['num_detections'].append(np.array([num_select] * cfg.dataset.batch_size).astype(np.int32))
        predictions['image_info'].append(image_info)
        predictions['source_id'].append(source_id)

    eval_results = evaluation_preds(preds=predictions)
    for key, value in sorted(eval_results.items(), key=operator.itemgetter(0)):
        eval_results[key] = float(value)
        logging.info("%s: %.9f", key, value)

    with open(os.path.join(cfg.results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f)

    logging.info("Finished evaluation.")


if __name__ == '__main__':
    main()
