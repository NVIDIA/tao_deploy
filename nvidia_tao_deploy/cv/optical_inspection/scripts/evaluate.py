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

"""Optical Inspection TensorRT evaluate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import numpy as np

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.optical_inspection.inferencer import OpticalInspectionInferencer
from nvidia_tao_deploy.cv.optical_inspection.dataloader import OpticalInspectionDataLoader
from nvidia_tao_deploy.cv.optical_inspection.hydra_config.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.visual_changenet.classification.utils import AOIMetrics
from sklearn import metrics
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="optical_inspection", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Convert encrypted uff or onnx model to TRT engine."""
    logger.info("Running evaluate")
    engine_file = cfg.evaluate.trt_engine
    batch_size = cfg.evaluate.batch_size
    dataset_config = cfg.dataset
    if cfg.evaluate.results_dir:
        results_dir = cfg.evaluate.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "evaluate")
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Instantiate the optical inspection evaluater.")
    optical_inspection_inferencer = OpticalInspectionInferencer(
        engine_path=engine_file,
        batch_size=batch_size
    )

    logger.info("Instantiating the optical inspection dataloader.")
    infer_dataloader = OpticalInspectionDataLoader(
        csv_file=dataset_config.infer_dataset.csv_path,
        input_data_path=dataset_config.infer_dataset.images_dir,
        train=False,
        data_config=dataset_config,
        dtype=optical_inspection_inferencer.inputs[0].host.dtype,
        split='evaluate'
    )
    total_num_samples = len(infer_dataloader)
    logger.info("Number of sample batches: {}".format(total_num_samples))
    logger.info("Running evaluate")
    margin = cfg.model.margin
    evaluate_score = []
    valid_AOIMetrics = AOIMetrics(margin)
    i = 0
    euclid = []
    for unit_batch, golden_batch, label in tqdm(infer_dataloader, total=total_num_samples):
        input_batches = [
            unit_batch,
            golden_batch
        ]
        results = optical_inspection_inferencer.infer(input_batches)
        pairwise_output = metrics.pairwise.paired_distances(results[0], results[1], metric="euclidean")
        evaluate_score.extend(
            [pairwise_output[idx] for idx in range(pairwise_output.shape[0])]
        )
        valid_AOIMetrics.update(pairwise_output, label)
        if i == 0:
            euclid = pairwise_output
        else:
            euclid = np.concatenate((euclid, pairwise_output), axis=0)
        i = i + 1

    total_accuracy = valid_AOIMetrics.compute()['total_accuracy'].item()
    false_alarm = valid_AOIMetrics.compute()['false_alarm'].item()
    defect_accuracy = valid_AOIMetrics.compute()['defect_accuracy'].item()
    false_negative = valid_AOIMetrics.compute()['false_negative'].item()

    logging.info(
        "Tot Comp {} Total Accuracy {} False Negative {} False Alarm {} Defect Correctly Captured {} for Margin {}".format(
            len(euclid),
            round(total_accuracy, 2),
            round(false_negative, 2),
            round(false_alarm, 2),
            round(defect_accuracy, 2),
            margin
        )
    )

    logger.info("Total number of evaluate outputs: {}".format(len(evaluate_score)))
    infer_dataloader.merged["output_score"] = evaluate_score[:len(infer_dataloader.merged)]
    infer_dataloader.merged.to_csv(
        os.path.join(results_dir, "evaluate.csv"),
        header=True,
        index=False
    )

    logging.info("Finished evaluation.")


if __name__ == "__main__":
    main()
