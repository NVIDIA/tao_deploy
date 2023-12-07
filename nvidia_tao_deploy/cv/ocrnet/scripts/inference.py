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

"""OCRNet TensorRT inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.ocrnet.dataloader import OCRNetLoader
from nvidia_tao_deploy.cv.ocrnet.inferencer import OCRNetInferencer
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.ocrnet.config.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.ocrnet.utils import decode_ctc, decode_attn


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="ocrnet", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Convert encrypted uff or onnx model to TRT engine."""
    engine_file = cfg.inference.trt_engine
    batch_size = cfg.inference.batch_size
    img_dirs = cfg.inference.inference_dataset_dir
    character_list_file = cfg.dataset.character_list_file
    img_width = cfg.inference.input_width
    img_height = cfg.inference.input_height
    img_channel = cfg.model.input_channel
    prediction_type = cfg.model.prediction
    shape = [img_channel, img_height, img_width]

    ocrnet_engine = OCRNetInferencer(engine_path=engine_file,
                                     batch_size=batch_size)

    if prediction_type == "CTC":
        character_list = ["CTCBlank"]
    elif prediction_type == "Attn":
        character_list = ["[GO]", "[s]"]
    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")
    with open(character_list_file, "r", encoding="utf-8") as f:
        for ch in f.readlines():
            ch = ch.strip()
            character_list.append(ch)

    inf_dl = OCRNetLoader(shape=shape,
                          image_dirs=[img_dirs],
                          batch_size=batch_size,
                          dtype=ocrnet_engine.inputs[0].host.dtype)

    for idx, (imgs, _) in enumerate(inf_dl):
        y_preds = ocrnet_engine.infer(imgs)
        output_probs, output_ids = y_preds
        img_paths = inf_dl.image_paths[idx * batch_size: (idx + 1) * batch_size]
        assert len(output_ids) == len(output_probs) == len(img_paths)
        for img_path, output_id, output_prob in zip(img_paths, output_ids, output_probs):
            if prediction_type == "CTC":
                text, conf = decode_ctc(output_id, output_prob, character_list=character_list)
            else:
                text, conf = decode_attn(output_id, output_prob, character_list=character_list)
            print(f"{img_path}: {text} {conf}")

    logging.info("TensorRT engine inference finished successfully.")


if __name__ == '__main__':
    main()
