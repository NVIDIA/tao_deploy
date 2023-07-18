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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from tqdm import tqdm
from omegaconf import OmegaConf
from nvidia_tao_deploy.cv.ocdnet.config.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.ocdnet.data_loader.icdar_uber import get_dataloader
from nvidia_tao_deploy.cv.ocdnet.post_processing.seg_detector_representer import get_post_processing
from nvidia_tao_deploy.cv.ocdnet.utils.ocr_metric.icdar2015.quad_metric import get_metric
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.ocdnet.tensorrt_utils.tensorrt_model import TrtModel


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
        self.trt_model = None
        if model_path.split(".")[-1] in ["trt", "engine"]:
            self.trt_model = TrtModel(model_path, 1)
            self.trt_model.build_or_load_trt_engine()

    def eval(self):
        """eval function."""
        raw_metrics = []
        for _, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            if _ >= len(self.validate_loader):
                break
            img = batch["img"]
            preds = self.trt_model.predict({"input": img})["pred"]
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


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"), config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name="ocdnet", mode="evaluation")
def main(cfg: ExperimentConfig) -> None:
    """Run the evaluation process."""
    if cfg.evaluate.results_dir is not None:
        results_dir = cfg.evaluate.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "evaluate")
    os.makedirs(results_dir, exist_ok=True)

    run_experiment(experiment_config=cfg,
                   model_path=cfg.evaluate.trt_engine)


if __name__ == "__main__":
    main()
