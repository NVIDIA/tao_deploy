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

"""Standalone TensorRT evaluation."""

import argparse
import logging

import os
import json
import numpy as np

from collections import defaultdict
import tensorrt as trt
from tqdm.auto import tqdm

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.multitask_classification.inferencer import MClassificationInferencer
from nvidia_tao_deploy.cv.multitask_classification.dataloader import MClassificationLoader
from nvidia_tao_deploy.cv.multitask_classification.proto.utils import load_proto


logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


@monitor_status(name='multitask_classification', mode='evaluate', hydra=False)
def main(args):
    """Multitask Classification TRT evaluation."""
    es = load_proto(args.experiment_spec)
    interpolation = es.model_config.resize_interpolation_method if es.model_config.resize_interpolation_method else 0
    interpolation_map = {
        0: "bilinear",
        1: "bicubic"
    }
    interpolation_method = interpolation_map[interpolation]

    # Override spec file if argument is provided
    image_dirs = args.image_dir if args.image_dir else es.dataset_config.image_directory_path
    batch_size = args.batch_size if args.batch_size else es.eval_config.batch_size

    trt_infer = MClassificationInferencer(args.model_path, batch_size=batch_size)

    if trt_infer.input_tensors[0].shape[0] in [1, 3]:
        data_format = "channels_first"
    else:
        data_format = "channels_last"

    dl = MClassificationLoader(
        trt_infer.input_tensors[0].shape,
        [image_dirs],
        es.dataset_config.val_csv_path,
        data_format=data_format,
        interpolation_method=interpolation_method,
        batch_size=batch_size,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype))

    tp = defaultdict(list)
    for imgs, labels in tqdm(dl, total=len(dl), desc="Producing predictions"):
        y_pred = trt_infer.infer(imgs)
        assert len(dl.class_dict_list_sorted) == len(labels) == len(y_pred), "Output size mismatch!"
        for (c, _), gt, pred in zip(dl.class_dict_list_sorted, labels, y_pred):
            tp[c].append((np.argmax(gt, axis=1) == np.argmax(pred, axis=1)).sum())

    # Get acc for each task
    eval_results = {}
    for k, v in tp.items():
        acc = sum(v) / len(dl.image_paths)
        logging.info("%s accuracy: %s", k, acc)
        eval_results[k] = float(acc)

    # Store evaluation results into JSON
    if args.results_dir is None:
        results_dir = os.path.dirname(args.model_path)
    else:
        results_dir = args.results_dir
    with open(os.path.join(results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f)
    logging.info("Finished evaluation.")


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
                parser mechanism.
    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='eval', description='Evaluate with a Multitask Classification TRT model.')

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
        '-b',
        '--batch_size',
        type=int,
        required=False,
        default=1,
        help='Batch size.')
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
