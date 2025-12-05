# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""TensorRT Inferencer class for Mask GDINO."""

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import do_inference, BINDING_TO_NPTYPE, HostDeviceMem
from nvidia_tao_deploy.cv.mask_grounding_dino.utils import default_colors, get_rgb_from_color_name, random_color

import ast
import numpy as np
from PIL import ImageDraw, Image
import random
import tensorrt as trt
import pycuda.driver as cuda
import cv2


class MaskGDINOInferencer(TRTInferencer):
    """Implements inference for the Mask G-DINO TensorRT engine."""

    def __init__(self, engine_path, num_classes,
                 input_shape=None, batch_size=None, data_format="channel_first", task="OD"):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            num_classes (int): number of classes that the model was trained on
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last
            task (str): task type - "OD" for object detection, "VG" for visual grounding
        """
        # Load TRT engine
        super().__init__(engine_path,
                         input_shape=input_shape,
                         batch_size=batch_size,
                         data_format=data_format)
        self.task = task

    def infer(self, inputs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            inputs (tuple): tuple of input tensors for the model
                - For OD: (image, input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks)
                - For VG: same as OD

        Returns:
            For OD task: (pred_logits, pred_boxes, pred_masks)
            For VG task: (pred_logits, pred_boxes, pred_masks, no_targets, union_mask_logits)
        """
        # 6 inputs: [inputs, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask]
        # inputs is already a list, so we can just pass it in since arg is list of named tensor inputs
        self._copy_input_to_host(inputs)

        # ...fetch model outputs...
        # OD: 3 outputs [pred_logits, pred_boxes, pred_masks]
        # VG: 5 outputs [pred_logits, pred_boxes, pred_masks, no_targets, union_mask_logits]
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)

        # Process TRT outputs to proper format
        processed_results = self.trt_output_process_fn(results)

        if self.task == "OD":
            # OD task returns 3 outputs
            pred_logits, pred_boxes, pred_masks = processed_results[:3]
            # pred_masks gets reshaped to (B, NQ, 1, H / 4, W / 4), so we get rid of the extra middle dimension
            pred_masks = np.squeeze(pred_masks, axis=2)
            return pred_logits, pred_boxes, pred_masks

        # for "VG" task
        # VG task returns 5 outputs
        pred_logits, pred_boxes, pred_masks, no_targets, union_mask_logits = processed_results
        # pred_masks gets reshaped to (B, NQ, 1, H / 4, W / 4), so we get rid of the extra middle dimension
        pred_masks = np.squeeze(pred_masks, axis=2)
        return pred_logits, pred_boxes, pred_masks, no_targets, union_mask_logits

    def trt_output_process_fn(self, y_encoded):
        """Function to process TRT model output.

        Args:
            y_encoded (list): list of TRT outputs in numpy

        Returns:
            pred_logits (np.ndarray): (B x NQ x N) logits of the prediction
            pred_boxes (np.ndarray): (B x NQ x 4) bounding boxes of the prediction
            pred_masks (np.ndarray): (B x NQ x h x w) masks of the prediction
        """
        return [np.reshape(out.host, out.numpy_shape) for out in y_encoded]

    def draw_bbox(self, img, prediction, masks_filt, class_mapping, threshold=0.3, color_map=None):  # noqa pylint: disable=W0237
        """Draws bbox on image and dump prediction in KITTI format

        Args:
            img (PIL image): Preprocessed image
            prediction (numpy.ndarray): (N x 6) predictions
            masks_filt: h x w x N (300)
            class_mapping (dict): key is the class index and value is the class name
            threshold (float): value to filter predictions
            color_map (dict): key is the class name and value is the color to be used
        """
        draw = ImageDraw.Draw(img)
        # Create random color map if not provided
        if not color_map:
            # Get unique class names from valid predictions
            unique_classes = set()
            for i in prediction:
                if int(i[0]) in class_mapping and float(i[1]) >= threshold:
                    unique_classes.add(class_mapping[int(i[0])])

            # Generate random colors for each unique class
            color_map = {}
            for cls_name in unique_classes:
                color_map[cls_name] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )

        W, H = img.size
        label_strings = []
        for j, i in enumerate(prediction):
            if int(i[0]) not in class_mapping:
                print(i[0], class_mapping)
                continue
            cls_name = class_mapping[int(i[0])]
            if float(i[1]) < threshold:
                continue

            if cls_name in color_map:
                draw.rectangle(((i[2], i[3]), (i[4], i[5])),
                               outline=color_map[cls_name])
                # txt pad
                draw.rectangle(((i[2], i[3] - 10), (i[2] + (i[4] - i[2]), i[3])),
                               fill=color_map[cls_name])
                draw.text((i[2], i[3] - 10), f"{cls_name}: {i[1]:.2f}")

                masks_filt = masks_filt > 0.5
                color = tuple(np.random.randint(0, 255, size=3).tolist())
                pred_color = masks_filt[..., j][..., None].astype(np.uint8) * np.array([color])
                pred_color = pred_color.astype(np.uint8)
                pred_color = Image.fromarray(pred_color).convert('RGBA')
                pred_color = pred_color.resize((W, H), resample=Image.NEAREST)
                img.paste(pred_color, (0, 0), Image.fromarray(masks_filt[..., j]).convert("L").resize((W, H), resample=Image.NEAREST))

            x1, y1, x2, y2 = float(i[2]), float(i[3]), float(i[4]), float(i[5])
            label_head = cls_name + " 0.00 0 0.00 "
            bbox_string = f"{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}"
            label_tail = f" 0.00 0.00 0.00 0.00 0.00 0.00 0.00 {float(i[1]):.3f}\n"
            label_string = label_head + bbox_string + label_tail
            label_strings.append(label_string)
        return img, label_strings

    def define_io_tensors(self, reshape: bool = False, input_shape: dict = None, batch_size: int = None):
        """
        Define input and output tensors for TensorRT engine with dynamic shape support.

        This method handles 0-sized dimensions that may occur with dynamic shapes
        and sets up the necessary buffers for inference.

        Args:
            reshape (bool): Whether to reshape tensors. Defaults to False.
            input_shape (dict): Optional dict mapping tensor names to shapes.
            batch_size (int): Optional batch size override.
        """
        # Use the enhanced allocate_buffers function
        self.input_tensors = []
        self.output_tensors = []

        for idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(idx)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            tensor_shape = self.context.get_tensor_shape(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT

            if is_input:
                self.define_input_tensor(tensor_name, tensor_dtype, tensor_shape, input_shape, batch_size)
            else:
                self.define_output_tensor(tensor_name, tensor_dtype, tensor_shape)

        # Use enhanced buffer allocation
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers_enhanced(
            self.engine, self.context, profile_idx=self.profile_idx, reshape=reshape
        )

    def _allocate_buffers_enhanced(self, engine, context=None, reshape=False, profile_idx=None):
        """Enhanced allocate_buffers that handles 0-sized dimensions properly.

        This method is needed for engines with dynamic shapes that may have 0-sized
        output tensors before actual inference is run.

        Args:
            engine: TensorRT engine
            context: TensorRT execution context
            reshape: Whether to reshape output buffers
            profile_idx: Profile index for dynamic shapes

        Returns:
            inputs, outputs, bindings, stream: Allocated buffers and CUDA stream
        """
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for idx in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(idx)
            is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            tensor_shape = engine.get_tensor_shape(tensor_name)

            if is_input:
                if tensor_shape[0] < 0:
                    profile_shape = engine.get_tensor_profile_shape(tensor_name, profile_idx)
                    assert len(profile_shape) == 3, (
                        "There should be min, opt and max shapes."
                        f"There are only {len(profile_shape)} with size {profile_shape}"
                    )
                    tensor_shape = profile_shape[-1]
            else:
                tensor_shape = context.get_tensor_shape(tensor_name)

            # Enhanced validation: Allow 0-sized dimensions, just not negative
            if not all(dim >= 0 for dim in tensor_shape):
                raise ValueError(f"Tensor {tensor_name} has invalid shape: {tensor_shape}")

            size = trt.volume(tensor_shape)
            trt_data_type = engine.get_tensor_dtype(tensor_name)

            # Determine dtype
            if tensor_name in BINDING_TO_NPTYPE.keys():
                dtype = BINDING_TO_NPTYPE[tensor_name]
            else:
                if trt.nptype(trt_data_type):
                    dtype = np.dtype(trt.nptype(trt_data_type))
                else:
                    size = int(size * trt_data_type.itemsize)
                    dtype = np.uint8

            # Enhanced memory allocation: Handle 0-sized tensors
            if size > 0:
                host_mem = cuda.pagelocked_empty(size, dtype)
                if reshape and not is_input:
                    if engine.has_implicit_batch_dimension:
                        target_shape = (engine.max_batch_size, tensor_shape[1], tensor_shape[2], tensor_shape[3])
                        host_mem = host_mem.reshape(*target_shape)
                    host_mem = host_mem.reshape(*tensor_shape)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
            else:
                # Handle 0-sized tensors
                host_mem = np.array([], dtype=dtype)
                device_mem = cuda.mem_alloc(1)  # Allocate at least 1 byte

            bindings.append(int(device_mem))
            tensor_obj = HostDeviceMem(host_mem, device_mem, tensor_shape, name=tensor_name)

            if is_input:
                inputs.append(tensor_obj)
            else:
                outputs.append(tensor_obj)

        return inputs, outputs, bindings, stream

    def draw_bbox_vg_simple(self, img, class_labels, scores, boxes, masks, threshold=0.3, color_map=None):
        """
        Simplified version for single image predictions (flattened format).

        Args:
            img (PIL Image): Input image
            class_labels (list): Flat list of phrases, e.g., ['yellow bananas', 'red car']
            scores (list): Flat list of scores, e.g., [0.81, 0.95]
            boxes (list): Flat list of boxes, e.g., [[x1,y1,x2,y2], [x1,y1,x2,y2]]
            masks (list): Flat list of masks
            threshold (float): Confidence threshold
            color_map (dict): Color mapping

        Returns:
            tuple: (annotated_image, label_strings)
        """
        img_annotated = img.copy()
        draw = ImageDraw.Draw(img_annotated)
        W, H = img_annotated.size
        label_strings = []

        if color_map is None:
            color_map = {}
        else:
            color_map = dict(color_map)

        # Process each detection
        for i in range(len(class_labels)):
            phrase = class_labels[i]
            score = float(scores[i])
            box = boxes[i]

            if score < threshold:
                continue

            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

            # Generate color
            if phrase not in color_map:
                color_map[phrase] = random_color(list(default_colors.keys()))

            phrase_color = ast.literal_eval(color_map[phrase])
            # Draw bbox
            draw.rectangle(((x1, y1), (x2, y2)), outline=phrase_color, width=3)

            # Draw text
            text_content = f"{phrase}: {score:.2f}"
            text_bbox = draw.textbbox((x1, y1 - 15), text_content)
            draw.rectangle(text_bbox, fill=phrase_color)
            draw.text((x1, y1 - 15), text_content, fill=(255, 255, 255))

            # Draw mask if available
            if i < len(masks) and masks[i] is not None:
                mask = masks[i]

                if mask.shape != (H, W):
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)

                binary_mask = mask > 0.5
                mask_overlay = np.zeros((H, W, 4), dtype=np.uint8)
                rgb = get_rgb_from_color_name(phrase_color)
                mask_overlay[binary_mask] = [*rgb, 100]
                mask_pil = Image.fromarray(mask_overlay, 'RGBA')
                img_annotated.paste(mask_pil, (0, 0), mask_pil)

            # KITTI format
            label_head = f"{phrase} 0.00 0 0.00 "
            bbox_string = f"{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}"
            label_tail = f" 0.00 0.00 0.00 0.00 0.00 0.00 0.00 {score:.3f}\n"
            label_string = label_head + bbox_string + label_tail
            label_strings.append(label_string)

        return img_annotated, label_strings
