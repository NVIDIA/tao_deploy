# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from tqdm.auto import tqdm

from nvidia_tao_core.config.oneformer.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner

from nvidia_tao_deploy.cv.oneformer.dataloader import OneformerDataLoader
from nvidia_tao_deploy.cv.oneformer.inferencer import OneformerInferencer
from nvidia_tao_deploy.cv.oneformer.tokenizer.tokenizer import Tokenize

from nvidia_tao_deploy.cv.mask2former.metrics import total_intersect_over_union

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="infer", schema=ExperimentConfig
)
@monitor_status(name='oneformer', mode='evaluate')
def main(cfg: ExperimentConfig) -> None:
    """Oneformer TRT evaluation."""
    if not os.path.exists(cfg.evaluate.trt_engine):
        raise FileNotFoundError(f"Provided evaluate.trt_engine at {cfg.evaluate.trt_engine} does not exist!")

    # TRT engine
    trt_infer = OneformerInferencer(
        cfg.evaluate.trt_engine,
        batch_size=cfg.dataset.val.batch_size)

    profile_idx = getattr(trt_infer, "profile_idx", 0)
    trt_infer.context.set_optimization_profile_async(profile_idx, trt_infer.stream.handle)

    assert cfg.export.task == "semantic", "[Experimental] Only support engines exported in the `semantic` mode."
    shape = trt_infer.input_tensors[0].tensor_shape

    dl = OneformerDataLoader(
        cfg.dataset.val.annotations,
        cfg.dataset.val.images,
        cfg.dataset.val.panoptic,
        shape,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
        contiguous_id=cfg.dataset.contiguous_id,
        batch_size=trt_infer.max_batch_size,
        data_format='channels_first',
        eval_samples=None,
        task=cfg.export.task,
        ignore_index=255
    )

    task_tokenizer = Tokenize(max_seq_len=cfg.dataset.max_seq_len)

    # Evaluation
    total_area_intersect, total_area_union = 0, 0
    total_area_pred_label, total_area_label = 0, 0

    for imgs, segms, task_texts in tqdm(dl, total=len(dl), desc="Producing predictions"):

        task_tokens = np.stack([task_tokenizer(task) for task in task_texts])
        task_tokens = np.asarray(task_tokens, dtype=trt.nptype(trt_infer.input_tensors[1].tensor_dtype))

        pred_mask_cls, pred_mask_pred = trt_infer.infer(imgs, task_tokens=task_tokens)
        semseg = trt_infer.postprocess_semseg(pred_mask_cls, pred_mask_pred, output_size=imgs.shape[-2:])

        area_intersect, area_union, area_pred_label, area_label = \
            total_intersect_over_union(semseg,
                                       segms,
                                       cfg.model.sem_seg_head.num_classes,
                                       ignore_index=255,
                                       reduce_zero_label=False)

        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label

    iou = total_area_intersect / total_area_union
    miou = np.nanmean(iou)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    logger.info("Evaluation metrics")
    logger.info("==================")
    logger.info("mIOU: %2.2f" % miou)
    logger.info("Acc: %2.2f" % all_acc)
    logger.info("Finished evaluation.")


if __name__ == '__main__':
    main()
