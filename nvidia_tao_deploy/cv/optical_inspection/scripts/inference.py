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

"""Optical Inspection TensorRT inference."""

import logging
import os
import tensorrt as trt

from nvidia_tao_core.config.optical_inspection.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.optical_inspection.inferencer import OpticalInspectionInferencer
from nvidia_tao_deploy.cv.optical_inspection.dataloader import OpticalInspectionDataLoader

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
@monitor_status(name="optical_inspection", mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """Convert encrypted uff or onnx model to TRT engine."""
    logger.info("Running inference")
    engine_file = cfg.inference.trt_engine
    batch_size = cfg.inference.batch_size
    dataset_config = cfg.dataset

    logger.info("Instantiate the optical inspection inferencer.")
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
        dtype=trt.nptype(optical_inspection_inferencer.input_tensors[0].tensor_dtype),
        batch_size=batch_size
    )
    inference_score = []
    total_num_samples = len(infer_dataloader)
    logger.info("Number of sample batches: {}".format(total_num_samples))
    logger.info("Running inference")
    for unit_batch, golden_batch in tqdm(infer_dataloader, total=total_num_samples):
        input_batches = [
            unit_batch,
            golden_batch
        ]
        results = optical_inspection_inferencer.infer(input_batches)
        pairwise_output = metrics.pairwise.paired_distances(results[0], results[1], metric="euclidean")
        inference_score.extend(
            [pairwise_output[idx] for idx in range(pairwise_output.shape[0])]
        )
    logger.info("Total number of inference outputs: {}".format(len(inference_score)))
    infer_dataloader.merged["output_score"] = inference_score[:len(infer_dataloader.merged)]
    infer_dataloader.merged.to_csv(
        os.path.join(cfg.results_dir, "inference.csv"),
        header=True,
        index=False
    )


if __name__ == "__main__":
    main()
