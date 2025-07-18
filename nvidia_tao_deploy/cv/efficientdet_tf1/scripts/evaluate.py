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
import os
import operator
import copy
import logging
import json
import six
import numpy as np
import tensorrt as trt
from tqdm.auto import tqdm

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.efficientdet_tf1.dataloader import EfficientDetCOCOLoader
from nvidia_tao_deploy.cv.efficientdet_tf1.inferencer import EfficientDetInferencer
from nvidia_tao_deploy.cv.efficientdet_tf1.proto.utils import load_proto
from nvidia_tao_deploy.metrics.coco_metric import EvaluationMetric


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


@monitor_status(name='efficientdet_tf1', mode='evaluate', hydra=False)
def main(args):
    """EfficientDet TRT evaluation."""
    # Load from proto-based spec file
    es = load_proto(args.experiment_spec)

    eval_samples = es.eval_config.eval_samples if es.eval_config.eval_samples else 0

    eval_metric = EvaluationMetric(es.dataset_config.validation_json_file, include_mask=False)
    trt_infer = EfficientDetInferencer(args.model_path, data_format="channel_last")

    h, w, c = trt_infer.input_tensors[0].shape

    dl = EfficientDetCOCOLoader(
        es.dataset_config.validation_json_file,
        shape=(1, h, w, c),
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
        batch_size=1,  # TF1 EfficentDet only supports bs=1
        image_dir=args.image_dir,
        eval_samples=eval_samples)

    predictions = {
        'detection_scores': [],
        'detection_boxes': [],
        'detection_classes': [],
        'source_id': [],
        'image_info': [],
        'num_detections': []
    }

    def evaluation_preds(preds):

        # Essential to avoid modifying the source dict
        _preds = copy.deepcopy(preds)
        for k, _ in six.iteritems(_preds):
            _preds[k] = np.concatenate(_preds[k], axis=0)
        eval_results = eval_metric.predict_metric_fn(_preds)
        return eval_results

    for imgs, scale, source_id, labels in tqdm(dl, total=len(dl), desc="Producing predictions"):
        image = np.array(imgs)

        image_info = []
        for i, label in enumerate(labels):
            image_info.append([label[-1][0], label[-1][1], scale[i], label[-1][2], label[-1][3]])
        image_info = np.array(image_info)
        detections = trt_infer.infer(image, scale)
        detections = copy.deepcopy(detections)

        predictions['detection_classes'].append(detections['detection_classes'])
        predictions['detection_scores'].append(detections['detection_scores'])
        predictions['detection_boxes'].append(detections['detection_boxes'])
        predictions['num_detections'].append(detections['num_detections'])
        predictions['image_info'].append(image_info)
        predictions['source_id'].append(source_id)

    eval_results = evaluation_preds(preds=predictions)
    for key, value in sorted(eval_results.items(), key=operator.itemgetter(0)):
        eval_results[key] = float(value)
        logging.info("%s: %.9f", key, value)

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
        parser = argparse.ArgumentParser(prog='eval', description='Evaluate with an EfficientDet TRT model.')

    parser.add_argument(
        '-e',
        '--experiment_spec',
        type=str,
        required=True,
        help='Path to the experiment spec file.'
    )
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
        help='Path to the EfficientDet TensorRT engine.'
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
