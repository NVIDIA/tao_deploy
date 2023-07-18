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

"""Base TensorRT inferencer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod

import tensorrt as trt
from PIL import ImageDraw


def load_engine(trt_runtime, engine_path):
    """Helper funtion to load an exported engine."""
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


class TRTInferencer(ABC):
    """Base TRT Inferencer."""

    def __init__(self, engine_path):
        """Init.

        Args:
            engine_path (str): The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(self.logger)
        self.engine = load_engine(self.trt_runtime, engine_path)
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

    @abstractmethod
    def infer(self, imgs, scales=None):
        """Execute inference on a batch of images.

        The images should already be batched and preprocessed.
        Memory copying to and from the GPU device will be performed here.

        Args:
            imgs (np.ndarray): A numpy array holding the image batch.
            scales: The image resize scales for each image in this batch.
                Default: No scale postprocessing applied.

        Returns:
            A nested list for each image in the batch and each detection in the list.
        """
        detections = {}
        return detections

    @abstractmethod
    def __del__(self):
        """Simple function to destroy tensorrt handlers."""

    def draw_bbox(self, img, prediction, class_mapping, threshold=0.3):
        """Draws bbox on image and dump prediction in KITTI format

        Args:
            img (numpy.ndarray): Preprocessed image
            prediction (numpy.ndarray): (N x 6) predictions
            class_mapping (dict): key is the class index and value is the class string.
                If set to None, no class predictions are displayed
            threshold (float): value to filter predictions
        """
        draw = ImageDraw.Draw(img)
        color_list = ['Black', 'Red', 'Blue', 'Gold', 'Purple']

        label_strings = []
        for i in prediction:
            if class_mapping and int(i[0]) not in class_mapping:
                continue
            if float(i[1]) < threshold:
                continue

            if isinstance(class_mapping, dict):
                cls_name = class_mapping[int(i[0])]
            else:
                cls_name = str(int(i[0]))

            # Default format is xyxy
            x1, y1, x2, y2 = float(i[2]), float(i[3]), float(i[4]), float(i[5])

            draw.rectangle(((x1, y1), (x2, y2)),
                           outline=color_list[int(i[0]) % len(color_list)])
            # txt pad
            draw.rectangle(((x1, y1), (x1 + 75, y1 + 10)),
                           fill=color_list[int(i[0]) % len(color_list)])

            if isinstance(class_mapping, dict):
                draw.text((x1, y1), f"{cls_name}: {i[1]:.2f}")
            else:
                # If label_map is not provided, do not show class prediction
                draw.text((x1, y1), f"{i[1]:.2f}")

            # Dump predictions

            label_head = cls_name + " 0.00 0 0.00 "
            bbox_string = f"{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}"
            label_tail = f" 0.00 0.00 0.00 0.00 0.00 0.00 0.00 {float(i[1]):.3f}\n"
            label_string = label_head + bbox_string + label_tail
            label_strings.append(label_string)

        return img, label_strings
