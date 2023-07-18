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

"""Standalone TensorRT inference."""

import os
import operator
import copy
import logging
import json
import six
import numpy as np
from tqdm.auto import tqdm

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.deformable_detr.dataloader import DDETRCOCOLoader
from nvidia_tao_deploy.cv.deformable_detr.inferencer import DDETRInferencer
from nvidia_tao_deploy.cv.deformable_detr.utils import post_process
from nvidia_tao_deploy.cv.dino.hydra_config.default_config import ExperimentConfig

from nvidia_tao_deploy.metrics.coco_metric import EvaluationMetric


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name='dino', mode='evaluation')
def main(cfg: ExperimentConfig) -> None:
    """DINO TRT evaluation."""
    if not os.path.exists(cfg.evaluate.trt_engine):
        raise FileNotFoundError(f"Provided evaluate.trt_engine at {cfg.evaluate.trt_engine} does not exist!")

    eval_metric = EvaluationMetric(cfg.dataset.test_data_sources.json_file,
                                   eval_class_ids=cfg.dataset.eval_class_ids,
                                   include_mask=False)
    trt_infer = DDETRInferencer(cfg.evaluate.trt_engine,
                                batch_size=cfg.dataset.batch_size,
                                num_classes=cfg.dataset.num_classes)

    c, h, w = trt_infer._input_shape

    dl = DDETRCOCOLoader(
        val_json_file=cfg.dataset.test_data_sources.json_file,
        shape=(cfg.dataset.batch_size, c, h, w),
        dtype=trt_infer.inputs[0].host.dtype,
        batch_size=cfg.dataset.batch_size,
        data_format="channels_first",
        image_std=cfg.dataset.augmentation.input_std,
        image_dir=cfg.dataset.test_data_sources.image_dir,
        eval_samples=None)

    predictions = {
        'detection_scores': [],
        'detection_boxes': [],
        'detection_classes': [],
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

        pred_logits, pred_boxes = trt_infer.infer(image)

        class_labels, scores, boxes = post_process(pred_logits, pred_boxes, target_sizes, num_select=cfg.model.num_select)

        # Convert to xywh
        boxes[:, :, 2:] -= boxes[:, :, :2]

        predictions['detection_classes'].append(class_labels)
        predictions['detection_scores'].append(scores)
        predictions['detection_boxes'].append(boxes)
        predictions['num_detections'].append(np.array([100] * cfg.dataset.batch_size).astype(np.int32))
        predictions['image_info'].append(image_info)
        predictions['source_id'].append(source_id)

    if cfg.evaluate.results_dir is not None:
        results_dir = cfg.evaluate.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "trt_evaluate")
    os.makedirs(results_dir, exist_ok=True)

    eval_results = evaluation_preds(preds=predictions)
    for key, value in sorted(eval_results.items(), key=operator.itemgetter(0)):
        eval_results[key] = float(value)
        logging.info("%s: %.9f", key, value)

    with open(os.path.join(results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f)

    logging.info("Finished evaluation.")


if __name__ == '__main__':
    main()
