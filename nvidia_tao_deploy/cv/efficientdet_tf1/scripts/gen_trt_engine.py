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

"""EfficientDet convert etlt/onnx model to TRT engine."""

import argparse
import logging
import os
import tempfile

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.efficientdet_tf1.engine_builder import EfficientDetEngineBuilder
from nvidia_tao_deploy.utils.decoding import decode_model


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


DEFAULT_MAX_BATCH_SIZE = 1
DEFAULT_MIN_BATCH_SIZE = 1
DEFAULT_OPT_BATCH_SIZE = 1


@monitor_status(name='efficientdet_tf1', mode='gen_trt_engine', hydra=False)
def main(args):
    """Convert encrypted uff or onnx model to TRT engine."""
    # decrypt etlt
    tmp_onnx_file, file_format = decode_model(args.model_path, args.key)

    if args.engine_file is not None or args.data_type == 'int8':
        if args.engine_file is None:
            engine_handle, temp_engine_path = tempfile.mkstemp()
            os.close(engine_handle)
            output_engine_path = temp_engine_path
        else:
            output_engine_path = args.engine_file

        builder = EfficientDetEngineBuilder(verbose=args.verbose,
                                            workspace=args.max_workspace_size,
                                            min_batch_size=args.min_batch_size,
                                            opt_batch_size=args.opt_batch_size,
                                            max_batch_size=args.max_batch_size,
                                            strict_type_constraints=args.strict_type_constraints)
        builder.create_network(tmp_onnx_file, file_format=file_format)
        builder.create_engine(
            output_engine_path,
            args.data_type,
            calib_input=args.cal_image_dir,
            calib_cache=args.cal_cache_file,
            calib_num_images=args.batch_size * args.batches,
            calib_batch_size=args.batch_size)


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
                parser mechanism.
    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='gen_trt_engine', description='Generate TRT engine of EfficientDet model.')

    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        help='Path to an EfficientDet .etlt or .onnx model file.'
    )
    parser.add_argument(
        '-k',
        '--key',
        type=str,
        required=False,
        help='Key to save or load a .etlt model.'
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="fp32",
        help="Data type for the TensorRT export.",
        choices=["fp32", "fp16", "int8"])
    parser.add_argument(
        "--cal_image_dir",
        default="",
        type=str,
        help="Directory of images to run int8 calibration.")
    parser.add_argument(
        '--cal_cache_file',
        default=None,
        type=str,
        help='Calibration cache file to write to.')
    parser.add_argument(
        "--engine_file",
        type=str,
        default=None,
        help="Path to the exported TRT engine.")
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help="Max batch size for TensorRT engine builder.")
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=DEFAULT_MIN_BATCH_SIZE,
        help="Min batch size for TensorRT engine builder.")
    parser.add_argument(
        "--opt_batch_size",
        type=int,
        default=DEFAULT_OPT_BATCH_SIZE,
        help="Opt batch size for TensorRT engine builder.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images per batch.")
    parser.add_argument(
        "--batches",
        type=int,
        default=10,
        help="Number of batches to calibrate over.")
    parser.add_argument(
        "--max_workspace_size",
        type=int,
        default=2,
        help="Max memory workspace size to allow in Gb for TensorRT engine builder (default: 2).")
    parser.add_argument(
        "-s",
        "--strict_type_constraints",
        action="store_true",
        default=False,
        help="A Boolean flag indicating whether to apply the \
              TensorRT strict type constraints when building the TensorRT engine.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Verbosity of the logger.")
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
    parser = build_command_line_parser()
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_command_line_arguments()
    main(args)
