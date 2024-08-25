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

from nvidia_tao_deploy.cv.efficientdet_tf1.dataloader import EfficientDetCOCOLoader
from nvidia_tao_deploy.cv.efficientdet_tf2.inferencer import EfficientDetInferencer
from nvidia_tao_deploy.cv.efficientdet_tf2.hydra_config.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.common.utils import update_results_dir
from nvidia_tao_deploy.metrics.coco_metric import EvaluationMetric

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment_spec", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for TRT engine evaluation."""
    cfg = update_results_dir(cfg, 'evaluate')
    run_evaluation(cfg=cfg)


@monitor_status(name='efficientdet_tf2', mode='evaluation')
def run_evaluation(cfg: ExperimentConfig) -> None:
    """EfficientDet TRT evaluation."""
    eval_samples = cfg.evaluate.num_samples

    eval_metric = EvaluationMetric(cfg.dataset.val_json_file, include_mask=False)
    trt_infer = EfficientDetInferencer(cfg.evaluate.trt_engine)

    dl = EfficientDetCOCOLoader(
        cfg.dataset.val_json_file,
        shape=trt_infer.inputs[0]['shape'],
        dtype=trt_infer.inputs[0]['dtype'],
        batch_size=cfg.evaluate.batch_size,
        image_dir=cfg.dataset.val_dirs[0],
        eval_samples=eval_samples)

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
        for i, label in enumerate(labels):
            image_info.append([label[-1][0], label[-1][1], scale[i], label[-1][2], label[-1][3]])
        image_info = np.array(image_info)
        detections = trt_infer.infer(image, scale)

        predictions['detection_classes'].append(detections['detection_classes'])
        predictions['detection_scores'].append(detections['detection_scores'])
        predictions['detection_boxes'].append(detections['detection_boxes'])
        predictions['num_detections'].append(detections['num_detections'])
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
