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

"""Utility class for performing TensorRT image inference."""

import numpy as np
from PIL import ImageDraw

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import do_inference


def trt_output_process_fn(y_encoded,
                          target_classes,
                          dims,
                          batch_size=1):
    """Function to process TRT model output.

    This function takes the raw output tensors from the detectnet_v2 model
    and performs the following steps:

    1. Denormalize the output bbox coordinates
    2. Threshold the coverage output to get the valid indices for the bboxes.
    3. Filter out the bboxes from the "output_bbox/BiasAdd" blob.
    4. Cluster the filterred boxes using DBSCAN.
    5. Render the outputs on images and save them to the output_path/images
    6. Serialize the output bboxes to KITTI Format label files in output_path/labels.
    """
    out2cluster = {}

    for idx, out in enumerate(y_encoded):
        # TF1 get_reshaped_outputs()
        out_shape = [batch_size,] + list(dims[idx])[-3:]
        out = np.reshape(out, out_shape)

        # TF1 predictions_to_dict() & keras_output_map()
        if out.shape[-3] == len(target_classes):
            output_meta_cov = out.transpose(0, 1, 3, 2)
        if out.shape[-3] == len(target_classes) * 4:
            output_meta_bbox = out.transpose(0, 1, 3, 2)

    # TF1 keras_output_map()
    for i in range(len(target_classes)):
        key = target_classes[i]
        out2cluster[key] = {'cov': output_meta_cov[:, i, :, :],  # pylint: disable=possibly-used-before-assignment
                            'bbox': output_meta_bbox[:, 4 * i: 4 * i + 4, :, :]}  # pylint: disable=possibly-used-before-assignment

    return out2cluster


class DetectNetInferencer(TRTInferencer):
    """Manages TensorRT objects for model inference."""

    def __init__(self, engine_path, target_classes=None, input_shape=None, batch_size=None, data_format="channel_first"):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            target_classes (list): list of target classes to be used
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last
        """
        # Load TRT engine
        super().__init__(engine_path, input_shape=input_shape, batch_size=batch_size, data_format=data_format)

        self.target_classes = target_classes

    def infer(self, imgs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            imgs (np.ndarray): Numpy array of normalized input tensor.
        """
        # Wrapped in list since arg is list of named tensor inputs
        # For detectnet, there is just 1: [input_1]
        self._copy_input_to_host([imgs])

        # ...fetch model outputs...
        # 2 named results: [output_bbox, output_cov]
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)
        dims = [out.numpy_shape for out in results]
        outputs = [out.host for out in results]

        # Process TRT outputs to proper format
        return trt_output_process_fn(outputs, self.target_classes, dims, self.max_batch_size)

    def draw_bbox(self, img, prediction, class_mapping, threshold=None, color_map=None):
        """Draws bbox on image and dump prediction in KITTI format

        Args:
            img (numpy.ndarray): Preprocessed image
            prediction (numpy.ndarray): (N x 6) predictions
            class_mapping (dict): key is the class index and value is the class string
            threshold (dict): dict containing class based threshold to filter predictions
            color_map (dict): key is the class name and value is the RGB value to be used
        """
        draw = ImageDraw.Draw(img)

        # Create random color map if not provided with the hash of the class name for
        # deterministic colors.
        if color_map is None:
            color_map = {}
            for i in prediction:
                if int(i[0]) in class_mapping and float(i[1]) >= threshold:
                    cls_name = class_mapping[int(i[0])]
                    if cls_name not in color_map:
                        # Use hash of class name for deterministic colors
                        hash_val = hash(cls_name)
                        color = (
                            (hash_val & 0xFF0000) >> 16,      # R
                            (hash_val & 0x00FF00) >> 8,       # G
                            (hash_val & 0x0000FF)             # B
                        )
                        color_map[cls_name] = color

        label_strings = []
        for i in prediction:
            cls_name = class_mapping[int(i[0])]
            if float(i[1]) < threshold[cls_name]:
                continue
            draw.rectangle(((i[2], i[3]), (i[4], i[5])),
                           outline=color_map[cls_name])
            # txt pad
            draw.rectangle(((i[2], i[3]), (i[2] + 75, i[3] + 10)),
                           fill=color_map[cls_name])
            draw.text((i[2], i[3]), f"{cls_name}: {i[1]:.2f}")

            # Dump predictions
            x1, y1, x2, y2 = float(i[2]), float(i[3]), float(i[4]), float(i[5])

            label_head = cls_name + " 0.00 0 0.00 "
            bbox_string = f"{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}"
            label_tail = f" 0.00 0.00 0.00 0.00 0.00 0.00 0.00 {float(i[1]):.3f}\n"
            label_string = label_head + bbox_string + label_tail
            label_strings.append(label_string)

        return img, label_strings
