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
from tqdm.auto import tqdm

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.centerpose.inferencer import CenterPoseInferencer
from nvidia_tao_deploy.cv.centerpose.hydra_config.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.centerpose.dataloader import CPPredictDataset
from nvidia_tao_deploy.cv.centerpose.utils import transform_outputs, merge_outputs, save_inference_prediction, PnPProcess

from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="infer", schema=ExperimentConfig
)
@monitor_status(name='centerpose', mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """CenterPose TRT Inference."""
    if not os.path.exists(cfg.inference.trt_engine):
        raise FileNotFoundError(f"Provided inference.trt_engine at {cfg.inference.trt_engine} does not exist!")

    trt_infer = CenterPoseInferencer(cfg.inference.trt_engine,
                                     batch_size=cfg.dataset.batch_size)
    c, h, w = trt_infer._input_shape

    batcher = CPPredictDataset(cfg.dataset, cfg.dataset.inference_data, (cfg.dataset.batch_size, c, h, w),
                               trt_infer.inputs[0].host.dtype)

    pnp_solver = PnPProcess(cfg.inference)

    # Create results directories
    if cfg.inference.results_dir:
        results_dir = cfg.inference.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "trt_inference")
    os.makedirs(results_dir, exist_ok=True)

    for batches, img_paths, (cxcy, max_axis) in tqdm(batcher.get_batch(), total=batcher.num_batches, desc="Producing predictions"):
        # Handle last batch as we artifically pad images for the last batch idx
        if len(img_paths) != len(batches):
            batches = batches[:len(img_paths)]

        det = trt_infer.infer(batches)

        # Post-processing
        transformed_det = transform_outputs(det, cxcy, max_axis, cfg.dataset.output_res)
        merged_det = merge_outputs(transformed_det)
        final_output = pnp_solver.get_process(merged_det)

        # Save the final results
        save_inference_prediction(final_output, results_dir, img_paths, cfg.inference)

    logging.info("Finished inference.")


if __name__ == '__main__':
    main()
