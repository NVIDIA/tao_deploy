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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import json
from tqdm.auto import tqdm

import logging

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.unet.dataloader import UNetLoader
from nvidia_tao_deploy.cv.unet.inferencer import UNetInferencer
from nvidia_tao_deploy.cv.unet.proto.utils import load_proto, initialize_params
from nvidia_tao_deploy.metrics.semantic_segmentation_metric import SemSegMetric


logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


@monitor_status(name='unet', mode='evaluation')
def main(args):
    """UNet TRT evaluation."""
    if not os.path.exists(args.experiment_spec):
        raise FileNotFoundError(f"{args.experiment_spec} does not exist!")

    experiment_spec = load_proto(args.experiment_spec)
    params = initialize_params(experiment_spec)

    # Override params if there are corresponding commandline args
    params['batch_size'] = args.batch_size if args.batch_size else params['batch_size']
    params['images_list'] = [args.image_dir] if args.image_dir else params['images_list']
    params['masks_list'] = [args.label_dir] if args.label_dir else params['masks_list']

    trt_infer = UNetInferencer(args.model_path, batch_size=args.batch_size, activation=params['activation'])

    dl = UNetLoader(
        trt_infer._input_shape,
        params['images_list'],
        params['masks_list'],
        params['num_classes'],
        batch_size=args.batch_size,
        resize_method=params['resize_method'],
        preprocess=params['preprocess'],
        resize_padding=params['resize_padding'],
        model_arch=params['arch'],
        input_image_type=params['input_image_type'],
        dtype=trt_infer.inputs[0].host.dtype)

    eval_metric = SemSegMetric(num_classes=params['num_classes'],
                               train_id_name_mapping=params['train_id_name_mapping'],
                               label_id_train_id_mapping=params['label_id_train_id_mapping'])
    gt_labels = []
    pred_labels = []

    for imgs, labels in tqdm(dl, total=len(dl), desc="Producing predictions"):
        gt_labels.extend(labels)
        y_pred = trt_infer.infer(imgs)
        pred_labels.extend(y_pred)

    metrices = eval_metric.get_evaluation_metrics(gt_labels, pred_labels)

    # Store evaluation results into JSON
    if args.results_dir is None:
        results_dir = os.path.dirname(args.model_path)
    else:
        results_dir = args.results_dir
    with open(os.path.join(results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(str(metrices["results_dic"]), f)
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
        parser = argparse.ArgumentParser(prog='eval', description='Evaluate with a UNet TRT model.')

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
        help='Path to the UNet TensorRT engine.'
    )
    parser.add_argument(
        '-e',
        '--experiment_spec',
        type=str,
        required=True,
        help='Path to the experiment spec.'
    )
    parser.add_argument(
        '-l',
        '--label_dir',
        type=str,
        required=False,
        help='Label directory.')
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
