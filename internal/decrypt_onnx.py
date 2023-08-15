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

"""Internal script to decrypt an .etlt file to onnx."""

import argparse
import os
import shutil
from nvidia_tao_deploy.utils.decoding import decode_etlt

def main(args=None):
    """decrypt an etlt file."""
    args = parse_command_line_arguments(args)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"{args.model_path} does not exist")
    
    dir_name = os.path.dirname(args.output_path)
    os.makedirs(dir_name, exist_ok=True)

    tmp_decrypted_model, backend  = decode_etlt(args.model_path, args.key)
    print(f"Model is decrypted to it's original '{backend}' backend")
    shutil.copy(tmp_decrypted_model, args.output_path)
    print(f"Model decrypted at {args.output_path}")

def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
                parser mechanism.
    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='decrypt_onnx', description='Decrypt an etlt file.')

    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        help='Path to an etlt model file.'
    )
    parser.add_argument(
        '-k',
        '--key',
        type=str,
        required=True,
        help='Key to save a .etlt model.'
    )
    parser.add_argument(
        '-o',
        '--output_path',
        type=str,
        required=True,
        help="Output onnx file path."
    )
    return parser


def parse_command_line_arguments(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
