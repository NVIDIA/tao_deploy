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
import pycocotools.mask as maskUtils

import tensorrt as trt

from nvidia_tao_deploy.cv.mask_rcnn.utils import generate_segmentation_from_masks, draw_mask_on_image_array
from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import allocate_buffers, do_inference


def trt_output_process_fn(y_pred, nms_size, mask_size, n_classes):
    """Proccess raw output from TRT engine."""
    y_detection = y_pred[0].reshape((-1, nms_size, 6))
    y_mask = y_pred[1].reshape((-1, nms_size, n_classes, mask_size, mask_size))
    y_mask[y_mask < 0] = 0
    return [y_detection, y_mask]


def process_prediction_for_eval(scales, box_coordinates):
    """Process the model prediction for COCO eval."""
    processed_box_coordinates = np.zeros_like(box_coordinates)

    # Handle the last batch where the # of images is smaller than the batch size.
    # Need to pad the scales to be in the correct batch shape
    if len(scales) != box_coordinates.shape[0]:
        new_scales = [1.0] * box_coordinates.shape[0]
        new_scales[:len(scales)] = scales
        scales = new_scales

    for image_id in range(box_coordinates.shape[0]):
        scale = scales[image_id]
        for box_id in range(box_coordinates.shape[1]):
            # Map [y1, x1, y2, x2] -> [x1, y1, w, h] and multiply detections
            # by image scale.
            y1, x1, y2, x2 = box_coordinates[image_id, box_id, :]
            new_box = scale * np.array([x1, y1, x2 - x1, y2 - y1])
            processed_box_coordinates[image_id, box_id, :] = new_box

    return processed_box_coordinates


class MRCNNInferencer(TRTInferencer):
    """Manages TensorRT objects for model inference."""

    def __init__(self, engine_path, nms_size=100, n_classes=2, mask_size=28, input_shape=None, batch_size=None, data_format="channel_first"):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last
        """
        # Load TRT engine
        super().__init__(engine_path)
        self.nms_size = nms_size
        self.n_classes = n_classes
        self.mask_size = mask_size
        self.max_batch_size = self.engine.max_batch_size
        self.execute_v2 = False

        # Execution context is needed for inference
        self.context = None

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        self._input_shape = []
        for binding in range(self.engine.num_bindings):
            if self.engine.binding_is_input(binding):
                self._input_shape = self.engine.get_binding_shape(binding)[-3:]
        assert len(self._input_shape) == 3, "Engine doesn't have valid input dimensions"

        if data_format == "channel_first":
            self.height = self._input_shape[1]
            self.width = self._input_shape[2]
        else:
            self.height = self._input_shape[0]
            self.width = self._input_shape[1]

        # set binding_shape for dynamic input
        if (input_shape is not None) or (batch_size is not None):
            self.context = self.engine.create_execution_context()
            if input_shape is not None:
                self.context.set_binding_shape(0, input_shape)
                self.max_batch_size = input_shape[0]
            else:
                self.context.set_binding_shape(0, [batch_size] + list(self._input_shape))
                self.max_batch_size = batch_size
            self.execute_v2 = True

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine,
                                                                                 self.context)
        if self.context is None:
            self.context = self.engine.create_execution_context()

        input_volume = trt.volume(self._input_shape)
        self.numpy_array = np.zeros((self.max_batch_size, input_volume))

    def infer(self, imgs, scales=None):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        # Verify if the supplied batch size is not too big
        max_batch_size = self.max_batch_size
        actual_batch_size = len(imgs)
        if actual_batch_size > max_batch_size:
            raise ValueError(f"image_paths list bigger ({actual_batch_size}) than \
                               engine max batch size ({max_batch_size})")

        self.numpy_array[:actual_batch_size] = imgs.reshape(actual_batch_size, -1)
        # ...copy them into appropriate place into memory...
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, self.numpy_array.ravel())

        # ...fetch model outputs...
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=max_batch_size,
            execute_v2=self.execute_v2)

        # ...and return results up to the actual batch size.
        y_pred = [i.reshape(max_batch_size, -1)[:actual_batch_size] for i in results]

        # Process TRT outputs to proper format
        processed_outputs = trt_output_process_fn(y_pred,
                                                  n_classes=self.n_classes,
                                                  mask_size=self.mask_size,
                                                  nms_size=self.nms_size)

        detections = {}

        bs, nd, _, _, _ = processed_outputs[1].shape
        masks = np.zeros((bs, nd)).tolist()
        for b in range(bs):
            for n in range(nd):
                class_idx = processed_outputs[0][..., -2][b, n]
                masks[b][n] = processed_outputs[1][b, n, int(class_idx), ...]  # if class_idx = -1
        masks = np.array(masks)
        bboxes = process_prediction_for_eval(scales, processed_outputs[0][..., 0:4])
        classes = np.copy(processed_outputs[0][..., -2])
        scores = np.copy(processed_outputs[0][..., -1])

        detections['detection_classes'] = classes
        detections['detection_scores'] = scores
        detections['detection_boxes'] = bboxes
        detections['detection_masks'] = masks
        detections['num_detections'] = np.array([self.nms_size] * self.max_batch_size).astype(np.int32)
        return detections

    def __del__(self):
        """Clear things up on object deletion."""
        # Clear session and buffer
        if self.trt_runtime:
            del self.trt_runtime

        if self.context:
            del self.context

        if self.engine:
            del self.engine

        if self.stream:
            del self.stream

        # Loop through inputs and free inputs.
        for inp in self.inputs:
            inp.device.free()

        # Loop through outputs and free them.
        for out in self.outputs:
            out.device.free()

    def draw_bbox_and_segm(self, img, classes, scores, bboxes, masks, class_mapping, threshold=0.3):
        """Draws bounding box and segmentation on image and dump prediction in KITTI format

        Args:
            img (numpy.ndarray): Preprocessed image
            classes (numpy.ndarray): (N x 100) predictions
            scores (numpy.ndarray): (N x 100) predictions
            bboxes (numpy.ndarray): (N x 100 x 4) predictions
            masks (numpy.ndarray): (N x 100 x mask_height x mask_width) predictions
            class_mapping (dict): key is the class index and value is the class string
            threshold (float): value to filter predictions
        """
        draw = ImageDraw.Draw(img)
        color_list = ['Black', 'Red', 'Blue', 'Gold', 'Purple']

        label_strings = []
        for idx, (cls, score, bbox, mask) in enumerate(zip(classes, scores, bboxes, masks)):
            cls_name = class_mapping[int(cls)]
            if float(score) < threshold:
                continue
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h

            draw.rectangle(((x1, y1), (x2, y2)),
                           outline=color_list[int(cls) % len(color_list)])
            # txt pad
            draw.rectangle(((x1, y1), (x1 + 75, y1 + 10)),
                           fill=color_list[int(cls) % len(color_list)])
            draw.text((x1, y1), f"{cls_name}: {score:.2f}")

            # Overlay segmentations
            mask = np.expand_dims(mask, axis=0)
            detected_bbox = np.expand_dims(bbox, axis=0)
            segms = generate_segmentation_from_masks(
                mask, detected_bbox,
                image_width=self.width,
                image_height=self.height,
                is_image_mask=False)
            segms = segms[0, :, :]
            img = draw_mask_on_image_array(img, segms, color=color_list[int(cls) % len(color_list)], alpha=0.4)
            draw = ImageDraw.Draw(img)

            # Dump labels
            json_obj = {}
            hhh, www = bbox[3] - bbox[1], bbox[2] - bbox[0]
            json_obj['area'] = int(www * hhh)
            json_obj['is_crowd'] = 0
            json_obj['bbox'] = [int(bbox[1]), int(bbox[0]), int(hhh), int(www)]
            json_obj['id'] = idx
            json_obj['category_id'] = int(cls)
            json_obj['score'] = float(score)
            # use RLE
            encoded_mask = maskUtils.encode(
                np.asfortranarray(segms.astype(np.uint8)))
            encoded_mask['counts'] = encoded_mask['counts'].decode('ascii')
            json_obj["segmentation"] = encoded_mask
            label_strings.append(json_obj)
        return img, label_strings
