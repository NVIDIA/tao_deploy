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

import argparse
import logging

import os
import pandas as pd
import json
import numpy as np

import tensorrt as trt
from tqdm.auto import tqdm

from nvidia_tao_deploy.cv.classification_tf1.inferencer import ClassificationInferencer
from nvidia_tao_deploy.cv.classification_tf1.dataloader import ClassificationLoader

from nvidia_tao_deploy.cv.classification_tf1.proto.utils import load_proto
from nvidia_tao_deploy.cv.common.decorators import monitor_status

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


@monitor_status(name='classification_tf1', mode='inference', hydra=False)
def main(args):
    """Classification TRT inference."""
    # Load from proto-based spec file
    es = load_proto(args.experiment_spec)
    interpolation = es.model_config.resize_interpolation_method if es.model_config.resize_interpolation_method else 0
    interpolation_map = {
        0: "bilinear",
        1: "bicubic"
    }
    interpolation_method = interpolation_map[interpolation]
    mode = es.train_config.preprocess_mode if es.train_config.preprocess_mode else "caffe"
    crop = "center" if es.eval_config.enable_center_crop else None
    image_mean = es.train_config.image_mean
    if image_mean:
        assert all(c in image_mean for c in ['r', 'g', 'b']), (
            "'r', 'g', 'b' should all be present in image_mean "
            "for images with 3 channels."
        )
        image_mean = [image_mean['b'], image_mean['g'], image_mean['r']]
    else:
        image_mean = [103.939, 116.779, 123.68]

    data_format = "channel_first"  # TF1 is always channels first

    batch_size = es.eval_config.batch_size if args.batch_size is None else args.batch_size
    image_dirs = args.image_dir

    if args.classmap:
        # if classmap is provided, we explicitly set the mapping from the json file
        if not os.path.exists(args.classmap):
            raise FileNotFoundError(f"{args.classmap} does not exist!")

        with open(args.classmap, "r", encoding="utf-8") as f:
            mapping_dict = json.load(f)
    else:
        # If not, the order of the classes are alphanumeric as defined by Keras
        # Ref: https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/preprocessing/image.py#L507
        mapping_dict = {}
        for idx, subdir in enumerate(sorted(os.listdir(image_dirs))):
            if os.path.isdir(os.path.join(image_dirs, subdir)):
                mapping_dict[subdir] = idx

    trt_infer = ClassificationInferencer(args.model_path, data_format=data_format, batch_size=batch_size)

    if trt_infer.etlt_type == "uff" and batch_size != 1:
        logger.warning("The etlt file was in deprecated UFF format which does not support dynmaic batch size. "
                       "Overriding the batch size to 1")
        batch_size = 1

    dl = ClassificationLoader(
        trt_infer.input_tensors[0].shape,
        [image_dirs],
        mapping_dict,
        is_inference=True,
        data_format="channels_first",
        interpolation_method=interpolation_method,
        mode=mode,
        crop=crop,
        batch_size=batch_size,
        image_mean=image_mean,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype))

    if args.results_dir is None:
        results_dir = os.path.dirname(args.model_path)
    else:
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)

    result_csv_path = os.path.join(results_dir, 'result.csv')
    with open(result_csv_path, 'w', encoding="utf-8") as csv_f:
        for i, (imgs, _) in tqdm(enumerate(dl), total=len(dl), desc="Producing predictions"):
            image_paths = dl.image_paths[np.arange(batch_size) + batch_size * i]

            y_pred = trt_infer.infer(imgs)
            # Class output from softmax layer
            class_indices = np.argmax(y_pred, axis=1)
            # Map label index to label name
            class_labels = map(lambda i: list(mapping_dict.keys())
                               [list(mapping_dict.values()).index(i)],
                               class_indices)
            conf = np.max(y_pred, axis=1)
            # Write predictions to file
            df = pd.DataFrame(zip(image_paths, class_labels, conf))
            df.to_csv(csv_f, header=False, index=False)

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
        parser = argparse.ArgumentParser(prog='infer', description='Inference with a Classification TRT model.')

    parser.add_argument(
        '-i',
        '--image_dir',
        type=str,
        required=True,
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
        default=None,
        help='Batch size.')
    parser.add_argument(
        '-c',
        '--classmap',
        type=str,
        required=False,
        default=None,
        help='File with class mapping.'
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
