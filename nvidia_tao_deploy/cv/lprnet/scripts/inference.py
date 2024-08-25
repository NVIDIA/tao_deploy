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
import numpy as np

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.lprnet.inferencer import LPRNetInferencer
from nvidia_tao_deploy.cv.lprnet.dataloader import LPRNetLoader
from nvidia_tao_deploy.cv.lprnet.utils import decode_ctc_conf

from nvidia_tao_deploy.cv.lprnet.proto.utils import load_proto

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


@monitor_status(name='lprnet', mode='inference')
def main(args):
    """LPRNet TRT inference."""
    # Load from proto-based spec file
    es = load_proto(args.experiment_spec)

    data_format = "channels_first"
    batch_size = args.batch_size if args.batch_size else es.eval_config.batch_size
    max_label_length = es.lpr_config.max_label_length if es.lpr_config.max_label_length else 8

    # Load image directories from validataion_data_sources
    image_dirs = []
    for data_source in es.dataset_config.validation_data_sources:
        image_dirs.append(data_source.image_directory_path)

    # Override the spec file if argument is passed
    if args.image_dir:
        image_dirs = [args.image_dir]

    trt_infer = LPRNetInferencer(args.model_path, data_format=data_format, batch_size=batch_size)

    characters_list_file = es.dataset_config.characters_list_file
    if not os.path.exists(characters_list_file):
        raise FileNotFoundError(f"{characters_list_file} does not exist!")

    with open(characters_list_file, "r", encoding="utf-8") as f:
        temp_list = f.readlines()
    classes = [i.strip() for i in temp_list]
    blank_id = len(classes)

    dl = LPRNetLoader(
        trt_infer._input_shape,
        image_dirs,
        [None],
        classes=classes,
        is_inference=True,
        batch_size=batch_size,
        max_label_length=max_label_length,
        dtype=trt_infer.inputs[0].host.dtype)

    for i, (imgs, _) in enumerate(dl):
        y_pred = trt_infer.infer(imgs)

        # decode prediction
        decoded_lp, _ = decode_ctc_conf(y_pred,
                                        classes=classes,
                                        blank_id=blank_id)
        image_paths = dl.image_paths[np.arange(args.batch_size) + args.batch_size * i]

        for image_path, lp in zip(image_paths, decoded_lp):
            print(f"{image_path}: {lp} ")

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
        parser = argparse.ArgumentParser(prog='eval', description='Evaluate with a LPRNet TRT model.')

    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        help='Path to the LPRNet TensorRT engine.'
    )
    parser.add_argument(
        '-i',
        '--image_dir',
        type=str,
        required=False,
        default=None,
        help='Input directory of images'
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
        help='Batch size.'
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
