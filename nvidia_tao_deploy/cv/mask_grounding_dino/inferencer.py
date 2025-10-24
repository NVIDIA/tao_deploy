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
from nvidia_tao_deploy.inferencer.utils import do_inference
import numpy as np
from PIL import ImageDraw, Image
import random


class MaskGDINOInferencer(TRTInferencer):
    """Implements inference for the Mask G-DINO TensorRT engine."""

    def __init__(self, engine_path, num_classes,
                 input_shape=None, batch_size=None, data_format="channel_first"):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            num_classes (int): number of classes that the model was trained on
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last
        """
        # Load TRT engine
        super().__init__(engine_path,
                         input_shape=input_shape,
                         batch_size=batch_size,
                         data_format=data_format)

    def infer(self, inputs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        # 6 inputs: [inputs, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask]
        # inputs is already a list, so we can just pass it in since arg is list of named tensor inputs
        self._copy_input_to_host(inputs)

        # ...fetch model outputs...
        # 3 named results: [pred_boxes, pred_logits_, pred_masks]
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)

        # Process TRT outputs to proper format
        pred_logits, pred_boxes, pred_masks = self.trt_output_process_fn(results)
        # pred_masks gets reshaped to (B, NQ, 1, H / 4, W / 4), so we get rid of the extra middle dimension
        pred_masks = np.squeeze(pred_masks, axis=2)
        return pred_logits, pred_boxes, pred_masks

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
