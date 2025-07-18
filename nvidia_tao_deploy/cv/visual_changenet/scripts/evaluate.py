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
import logging
import numpy as np
import tensorrt as trt
from tqdm import tqdm

from nvidia_tao_core.config.visual_changenet.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.visual_changenet.segmentation.inferencer import ChangeNetInferencer as ChangeNetSegmentInferencer
from nvidia_tao_deploy.cv.visual_changenet.classification.inferencer import ChangeNetInferencer as ChangeNetClassifyInferencer
from nvidia_tao_deploy.cv.visual_changenet.segmentation.dataloader import ChangeNetDataLoader as ChangeNetSegmentDataLoader
from nvidia_tao_deploy.cv.optical_inspection.dataloader import OpticalInspectionDataLoader as ChangeNetClassifyDataLoader
from nvidia_tao_deploy.cv.visual_changenet.dataloader import MultiGoldenDataLoader
from nvidia_tao_deploy.cv.visual_changenet.segmentation.utils import get_color_mapping, visualize_infer_output
from nvidia_tao_deploy.cv.visual_changenet.classification.utils import AOIMetrics

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name='visual_changenet', mode='evaluate')
def main(cfg: ExperimentConfig) -> None:
    """Visual ChangeNet TRT evaluation."""
    if not os.path.exists(cfg.evaluate.trt_engine):
        raise FileNotFoundError(f"Provided evaluate.trt_engine at {cfg.evaluate.trt_engine} does not exist!")

    logger.info("Running Evaluation")
    engine_file = cfg.evaluate.trt_engine
    batch_size = cfg.evaluate.batch_size

    task = cfg.task

    if task == 'segment':
        dataset_config = cfg.dataset.segment
        logger.info("Instantiate the Visual ChangeNet Segmentation evaluate.")
        changenet_inferencer = ChangeNetSegmentInferencer(
            engine_path=engine_file,
            batch_size=batch_size,
            n_class=dataset_config.num_classes,
            mode='test'
        )

        logger.info("Instantiating the Visual ChangeNet Segmentation dataloader.")
        infer_dataloader = ChangeNetSegmentDataLoader(
            dataset_config=dataset_config,
            dtype=trt.nptype(changenet_inferencer.input_tensors[0].tensor_dtype),
            mode='test',
            split=dataset_config.test_split
        )

    elif task == 'classify':
        dataset_config = cfg.dataset.classify
        num_golden = dataset_config.num_golden

        logger.info("Instantiate the Visual ChangeNet Classification evaluate.")
        changenet_inferencer = ChangeNetClassifyInferencer(
            engine_path=engine_file,
            batch_size=batch_size,
            diff_module=cfg.model.classify.difference_module
        )

        logger.info("Instantiating the Visual ChangeNet Classification dataloader.")
        if num_golden == 1:
            infer_dataloader = ChangeNetClassifyDataLoader(
                csv_file=dataset_config.test_dataset.csv_path,
                input_data_path=dataset_config.test_dataset.images_dir,
                train=False,
                data_config=dataset_config,
                dtype=trt.nptype(changenet_inferencer.input_tensors[0].tensor_dtype),
                split='evaluate',
                batch_size=batch_size
            )
        else:
            infer_dataloader = MultiGoldenDataLoader(
                csv_file=dataset_config.test_dataset.csv_path,
                input_data_path=dataset_config.test_dataset.images_dir,
                train=False,
                data_config=dataset_config,
                dtype=trt.nptype(changenet_inferencer.input_tensors[0].tensor_dtype),
                split='evaluate',
                batch_size=batch_size,
                num_golden=num_golden
            )

    else:
        raise NotImplementedError('Only tasks supported by Visual ChangeNet are: "segment" and "classify"')

    assert dataset_config.batch_size == batch_size, "Batch size must be the same in both dataset and inference config"
    total_num_samples = len(infer_dataloader)
    logger.info("Number of sample batches: {}".format(total_num_samples))
    logger.info("Running evaluate")

    if task == 'segment':
        # Color map for segmentation output visualisation for multi-class output
        color_map = get_color_mapping(dataset_name=dataset_config.data_name,
                                      color_mapping_custom=dataset_config.color_map,
                                      num_classes=dataset_config.num_classes
                                      )

        # Inference
        for idx, (img_1, img_2, label) in tqdm(enumerate(infer_dataloader), total=total_num_samples):
            input_batches = [
                img_1,
                img_2
            ]
            image_names = infer_dataloader.img_name_list[np.arange(batch_size) + batch_size * idx]
            results = changenet_inferencer.infer(input_batches, target=label)

            # Save output visualisation
            for img1, img2, result, img_name, gt in zip(img_1, img_2, results, image_names, label):
                visualize_infer_output(img_name, result, img1, img2, dataset_config.num_classes,
                                       color_map, cfg.results_dir, gt, mode='test')

        scores, mean_score_dict = changenet_inferencer.running_metric.get_scores()
        logger.info("Evaluation Metric Scores: {}".format(scores))
        logger.info("Evaluation Metric Scores (Mean Scores): {}".format(mean_score_dict))

    elif task == 'classify':
        margin = cfg.model.classify.eval_margin
        evaluate_score = []
        valid_AOIMetrics = AOIMetrics(margin)
        i = 0
        euclid = []
        for unit_batch, golden_batch, label in tqdm(infer_dataloader, total=total_num_samples):
            input_batches = [
                unit_batch,
                golden_batch
            ]
            results = changenet_inferencer.infer(input_batches)
            evaluate_score.extend(
                [results[idx] for idx in range(results.shape[0])]
            )
            valid_AOIMetrics.update(results, label)
            if i == 0:
                euclid = results
            else:
                euclid = np.concatenate((euclid, results), axis=0)
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
            os.path.join(cfg.results_dir, "evaluate.csv"),
            header=True,
            index=False
        )

    else:
        raise NotImplementedError('Only tasks supported by Visual ChangeNet are: "segment" and "classify"')

    logging.info("Finished evaluation.")


if __name__ == '__main__':
    main()
