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

"""DepthNet TensorRT evaluation script.

This script provides comprehensive evaluation capabilities for DepthNet models using
TensorRT-optimized inference. It supports evaluation of both monocular and stereo
depth estimation models with industry-standard metrics.

The evaluation process includes:
- TensorRT engine loading and inference
- Batch processing of test images
- Ground truth comparison and metric computation
- Results storage in JSON format
- Support for both monocular and stereo depth estimation
"""

import os
import logging
import cv2
import numpy as np
import tensorrt as trt
import json
from tqdm.auto import tqdm
import operator

from nvidia_tao_core.config.depth_net.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.depth_net.inferencer import DepthNetInferencer
from nvidia_tao_deploy.cv.depth_net.evaluation import MonoDepthEvaluator, StereoDepthEvaluator
from nvidia_tao_deploy.cv.depth_net.dataloader import DepthNetDataLoader
from nvidia_tao_deploy.cv.depth_net.utils import check_batch_sizes

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name='depth_net', mode='evaluate')
def main(cfg: ExperimentConfig) -> None:
    """
    Evaluate DepthNet model using TensorRT engine with comprehensive metrics.

    This function performs end-to-end evaluation of a DepthNet model using a TensorRT
    engine. It processes test images in batches, computes depth predictions, and
    compares them against ground truth to calculate various evaluation metrics.

    The evaluation supports:
    - Monocular depth estimation (single image input)
    - Stereo depth estimation (left-right image pairs)
    - Configurable preprocessing parameters
    - Multiple evaluation metrics (abs_rel, d1, bp1, bp2, bp3, epe)
    - Results storage in JSON format

    Args:
        cfg (ExperimentConfig): Configuration object containing all evaluation parameters
            including model paths, dataset configuration, and evaluation settings.

    Raises:
        FileNotFoundError: If the TensorRT engine file does not exist.
        ValueError: If configuration parameters are invalid.
        RuntimeError: If evaluation process fails.

    Example:
        The function is typically called through the command line interface:
        ```bash
        python evaluate.py evaluate.trt_engine=/path/to/model.trt
        ```

    Output:
        - Evaluation results are saved to `cfg.results_dir/results.json`
        - Console output includes progress information and final metrics
    """
    if not os.path.exists(cfg.evaluate.trt_engine):
        raise FileNotFoundError(f"Provided evaluate.trt_engine at {cfg.evaluate.trt_engine} does not exist!")

    trt_infer = DepthNetInferencer(cfg.evaluate.trt_engine,
                                   batch_size=cfg.dataset.test_dataset.batch_size)

    c, h, w = trt_infer.input_tensors[0].shape
    dataset_name = cfg.dataset.dataset_name
    if dataset_name.lower() == "stereodataset":
        evaluator = StereoDepthEvaluator(sync_on_compute=False, max_disparity=416)
    else:
        evaluator = MonoDepthEvaluator(align_gt=True, sync_on_compute=False, min_depth=0.0001, max_depth=416)

    loader = DepthNetDataLoader(
        cfg.dataset.test_dataset.data_sources,
        (cfg.dataset.test_dataset.batch_size, c, h, w),
        trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
        preprocessor="DepthNet",
        evaluation=True,
    )

    # Create results directories by going through the batcher
    for batches, img_paths, scales, gt_depths in tqdm(loader.get_batch(), total=loader.num_batches, desc="Producing predictions"):
        left_images, batches = check_batch_sizes(batches, img_paths)

        pred_depths = trt_infer.infer(batches)

        # squeeze depth tensor to 3D for FoundationStereo (B, 1, H, W) -> (B, H, W)
        if pred_depths.ndim == 4:
            pred_depths = pred_depths.squeeze(1)

        pred_dict = []

        # post-processing to resize to original size and update the evaluator
        for batch, scale, pred_depth, gt_depth in zip(left_images, scales, pred_depths, gt_depths):
            valid_mask = np.ones(gt_depth.shape, dtype=bool)
            _, new_h, new_w = batch.shape
            orig_h, orig_w = int(scale[0] * new_h), int(scale[1] * new_w)
            # interpolate pred_depth to original size
            pred_depth = cv2.resize(pred_depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            pred_dict.append({"depth_pred": pred_depth, "disp_gt": gt_depth, "valid_mask": valid_mask})
        evaluator.update(pred_dict)

    # Computing the final evaluation metrics and store evaluation results into JSON
    eval_results = evaluator.compute()
    logging.info("logging evaluation results.")
    for key, value in sorted(eval_results.items(), key=operator.itemgetter(0)):
        eval_results[key] = float(value)
        logging.info("%s: %.9f", key, value)

    with open(os.path.join(cfg.results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    logging.info("Finished Evaluation.")


if __name__ == '__main__':
    main()
