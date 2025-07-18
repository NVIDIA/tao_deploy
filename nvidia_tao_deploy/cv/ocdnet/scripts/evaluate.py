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

"""Evaluate a trained ocdnet model."""

import os
import json
import logging
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from nvidia_tao_core.config.ocdnet.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.ocdnet.data_loader.icdar_uber import get_dataloader
from nvidia_tao_deploy.cv.ocdnet.post_processing.seg_detector_representer import get_post_processing
from nvidia_tao_deploy.cv.ocdnet.utils.ocr_metric.icdar2015.quad_metric import get_metric
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.ocdnet.inferencer import OCDNetInferencer


class Evaluate():
    """Eval class."""

    def __init__(self, model_path, config_file, gpu_id=0):
        """Initialize."""
        config = config_file
        config['model']['pretrained'] = False
        self.validate_loader = get_dataloader(config['dataset']['validate_dataset'], False)
        self.post_process = get_post_processing(config['evaluate']['post_processing'])
        self.metric_cls = get_metric(config['evaluate']['metric'])
        self.box_thresh = config['evaluate']['post_processing']["args"]["box_thresh"]
        self.trt_model = OCDNetInferencer(model_path, batch_size=1)

    def eval(self):
        """eval function."""
        raw_metrics = []
        for _, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            if _ >= len(self.validate_loader):
                break
            # @seanf: dataloader always uses a batch size of 1
            img = np.expand_dims(batch["img"], axis=0)
            preds = self.trt_model.infer(img)
            boxes, scores = self.post_process(batch, preds, is_output_polygon=self.metric_cls.is_output_polygon)
            raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores), box_thresh=self.box_thresh)
            raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg


def run_experiment(experiment_config, model_path):
    """Run experiment."""
    experiment_config = OmegaConf.to_container(experiment_config)
    evaluation = Evaluate(model_path, experiment_config)
    result = evaluation.eval()
    print("Precision: ", result[1])
    print("Recall: ", result[0])
    print("F-measure: ", result[2])
    eval_results = {"Precision": result[1], "Recall": result[0], "F-measure": result[2]}
    print("Saving evaluation result in {}/results.json".format(experiment_config['evaluate']['results_dir']))
    with open(os.path.join(experiment_config['evaluate']['results_dir'], "results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f)


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"), config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name="ocdnet", mode='evaluate')
def main(cfg: ExperimentConfig) -> None:
    """Run the evaluation process."""
    run_experiment(experiment_config=cfg,
                   model_path=cfg.evaluate.trt_engine)


if __name__ == "__main__":
    main()
