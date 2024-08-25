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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.multitask_classification.inferencer import MClassificationInferencer
from nvidia_tao_deploy.cv.multitask_classification.dataloader import MClassificationLoader
from nvidia_tao_deploy.cv.multitask_classification.proto.utils import load_proto


logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


@monitor_status(name='multitask_classification', mode='inference')
def main(args):
    """Multitask Classification TRT inference."""
    es = load_proto(args.experiment_spec)
    interpolation = es.model_config.resize_interpolation_method if es.model_config.resize_interpolation_method else 0
    interpolation_map = {
        0: "bilinear",
        1: "bicubic"
    }
    interpolation_method = interpolation_map[interpolation]

    # Override spec file if argument is provided
    image_dirs = args.image_dir if args.image_dir else es.dataset_config.image_directory_path
    batch_size = 1  # Inference is set to 1

    trt_infer = MClassificationInferencer(args.model_path, batch_size=batch_size)

    if trt_infer._input_shape[0] in [1, 3]:
        data_format = "channels_first"
    else:
        data_format = "channels_last"

    dl = MClassificationLoader(
        trt_infer._input_shape,
        [image_dirs],
        es.dataset_config.val_csv_path,
        data_format=data_format,
        interpolation_method=interpolation_method,
        batch_size=batch_size,
        dtype=trt_infer.inputs[0].host.dtype)

    if args.results_dir is None:
        results_dir = os.path.dirname(args.model_path)
    else:
        results_dir = args.results_dir
    result_csv_path = os.path.join(results_dir, 'result.csv')
    with open(result_csv_path, 'w', encoding="utf-8") as csv_f:
        for i, (imgs, _) in tqdm(enumerate(dl), total=len(dl), desc="Producing predictions"):
            y_pred = trt_infer.infer(imgs)
            image_paths = dl.image_paths[np.arange(batch_size) + batch_size * i]

            conf = [np.max(pred.reshape(-1)) for pred in y_pred]
            class_labels = []
            for (c, _), pred in zip(dl.class_dict_list_sorted, y_pred):
                class_labels.extend([dl.class_mapping_inv[c][np.argmax(p)] for p in pred])

            r = {"filename": image_paths}
            for idx, (c, _) in enumerate(dl.class_dict_list_sorted):
                r[c] = [f"{class_labels[idx]} ({conf[idx]:.4f})"]

            # Write predictions to file
            df = pd.DataFrame.from_dict(r)
            # Add header only for the first item
            df.to_csv(csv_f, header=bool(i == 0), index=False)

    logging.info("Finished inference.")


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
                parser mechanism.
    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='infer', description='Inference with a Multitask Classification TRT model.')

    parser.add_argument(
        '-i',
        '--image_dir',
        type=str,
        required=False,
        default=None,
        help='Input directory of images')
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        help='Path to the Classification TensorRT engine.'
    )
    parser.add_argument(
        '-e',
        '--experiment_spec',
        type=str,
        required=True,
        help='Path to the experiment spec file.'
    )
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        required=True,
        default=None,
        help='Output directory where the log is saved.'
    )
    return parser


def parse_command_line_arguments(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_command_line_arguments()
    main(args)
