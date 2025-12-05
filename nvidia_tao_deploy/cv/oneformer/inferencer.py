# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""TensorRT Engine class for Oneformer."""

import numpy as np
import pycuda.driver as cuda
from scipy.ndimage import zoom

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer


class OneformerInferencer(TRTInferencer):
    """Implements inference for the Oneformer TensorRT engine."""

    def __init__(self, engine_path,
                 input_shape=None,
                 batch_size=None,
                 data_format="channel_first",
                 is_inference=False):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last
        """
        # Load TRT engine
        super().__init__(engine_path,
                         input_shape=input_shape,
                         batch_size=batch_size,
                         data_format=data_format)
        self.is_inference = is_inference

    def infer(self, imgs, scales=None, task_tokens=None):
        """Run TensorRT inference (v3 async) and return raw model outputs.

        Args:
            imgs (np.ndarray): input images of shape [B, C, H, W]
            scales (None): The image resize scales for each image in this batch.
                Default: No scale postprocessing applied.
            task_tokens (np.ndarray): task tokens of shape [B, D]

        Returns:
            tuple: (pred_logits, pred_masks) as numpy arrays with shapes
                   matching engine outputs.
        """
        tokens = task_tokens if task_tokens is not None else scales

        # Set dynamic shapes
        name_img = self.input_tensors[0].tensor_name
        name_task = self.input_tensors[1].tensor_name
        batch_size, num_channels, height, width = imgs.shape
        token_dim = tokens.shape[1]
        self.context.set_input_shape(name_img, [batch_size, num_channels, height, width])
        self.context.set_input_shape(name_task, [batch_size, token_dim])

        # Host to device
        self._copy_input_to_host([imgs, tokens])
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Device to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        self.stream.synchronize()

        # Reshape like original postprocess
        results = self.outputs
        pred_logits = np.reshape(results[0].host, results[0].numpy_shape)
        pred_masks = np.reshape(results[1].host, results[1].numpy_shape)
        return pred_logits, pred_masks

    def postprocess_semseg(self, mask_cls: np.ndarray, mask_pred: np.ndarray, output_size=None) -> np.ndarray:
        """Convert model outputs to semantic segmentation maps.

        Args:
            mask_cls (np.ndarray): classification logits, shape [B, Q, C] or [Q, C]
            mask_pred (np.ndarray): mask logits, shape [B, Q, H, W] or [Q, H, W]
            output_size (tuple[int, int] or None): desired (H, W) of output semseg.
                If None, uses mask_pred spatial size.

        Returns:
            np.ndarray: semantic segmentation maps of shape [B, H, W].
        """
        if mask_cls.ndim == 2:
            mask_cls = mask_cls[None, ...]
        if mask_pred.ndim == 3:
            mask_pred = mask_pred[None, ...]

        # Softmax: exp(x) / sum(exp(x))
        mask_cls_exp = np.exp(mask_cls - np.max(mask_cls, axis=-1, keepdims=True))
        mask_cls_softmax = mask_cls_exp / np.sum(mask_cls_exp, axis=-1, keepdims=True)
        mask_cls_softmax = mask_cls_softmax[..., :-1]  # Remove last class

        # Sigmoid: 1 / (1 + exp(-x))
        mask_pred_sigmoid = 1.0 / (1.0 + np.exp(-mask_pred))

        # Einstein summation: B x Q x C  and  B x Q x H x W  ->  B x C x H x W
        pred_masks = np.einsum("bqc,bqhw->bchw", mask_cls_softmax, mask_pred_sigmoid)

        if output_size is not None:
            # Bilinear interpolation using scipy
            _, _, H, W = pred_masks.shape
            target_h, target_w = output_size
            zoom_factors = (1, 1, target_h / H, target_w / W)
            pred_masks = zoom(pred_masks, zoom_factors, order=1)  # order=1 for bilinear

        semseg = np.argmax(pred_masks, axis=1)
        return semseg
