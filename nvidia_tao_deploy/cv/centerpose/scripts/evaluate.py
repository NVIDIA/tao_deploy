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
import tensorrt as trt
from tqdm.auto import tqdm

from nvidia_tao_core.config.centerpose.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.centerpose.dataloader import CPPredictDataset
from nvidia_tao_deploy.cv.centerpose.inferencer import CenterPoseInferencer
from nvidia_tao_deploy.cv.centerpose.utils import transform_outputs, merge_outputs, PnPProcess
from nvidia_tao_deploy.cv.centerpose.centerpose_evaluator import Evaluator


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name='centerpose', mode='evaluate')
def main(cfg: ExperimentConfig) -> None:
    """CenterPose TRT evaluation."""
    if not os.path.exists(cfg.evaluate.trt_engine):
        raise FileNotFoundError(f"Provided evaluate.trt_engine at {cfg.evaluate.trt_engine} does not exist!")

    trt_infer = CenterPoseInferencer(cfg.evaluate.trt_engine,
                                     batch_size=cfg.dataset.batch_size)

    c, h, w = trt_infer.input_tensors[0].shape

    batcher = CPPredictDataset(cfg.dataset, cfg.dataset.test_data, (cfg.dataset.batch_size, c, h, w),
                               trt.nptype(trt_infer.input_tensors[0].tensor_dtype), evaluate=True)

    pnp_solver = PnPProcess(cfg.evaluate, evaluate=True)

    cp_evaluator = Evaluator(cfg)

    for batches, img_paths, (cxcy, max_axis, json_paths, intrinsics) in tqdm(batcher.get_evaluation_batch(), total=batcher.num_batches, desc="Producing predictions"):
        # Handle last batch as we artifically pad images for the last batch idx
        if len(img_paths) != len(batches):
            batches = batches[:len(img_paths)]

        det = trt_infer.infer(batches)

        # Post-processing
        # Transform and merge the decoded outputs
        transformed_det = transform_outputs(det, cxcy, max_axis, cfg.dataset.output_res)

        # Filter and merge the output
        merged_det = merge_outputs(transformed_det)

        # Handle last batch as we artifically pad images for the last batch idx
        if len(img_paths) != len(merged_det):
            merged_det = merged_det[:len(img_paths)]

        # Set up the testing intrinsic matrix and process the pnp process
        pnp_solver.set_intrinsic_matrix(intrinsics)
        final_output = pnp_solver.get_process(merged_det)

        # Launch the evaluation
        cp_evaluator.evaluate(final_output, json_paths)

    cp_evaluator.finalize()
    cp_evaluator.write_report(cfg.results_dir)

    logging.info("Finished evaluation.")


if __name__ == '__main__':
    main()
