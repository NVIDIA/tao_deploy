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

"""DepthNet utility functions.

This module provides utility functions for DepthNet, including image preprocessing,
visualization, and ground truth depth reading. It supports both monocular and stereo
depth estimation models with efficient batch processing capabilities.
"""

import numpy as np
import matplotlib
import cv2
import re


def check_batch_sizes(batches, img_paths):
    """
    When the last batch is not full, we need to check the batch sizes of the batches and img_paths and discard padded data in the last batch.

    Args:
        batches (dict or np.ndarray): The batches to check.
        img_paths (list): The image paths to check.

    Returns:
        tuple: A tuple containing the left images and the batches.
    """
    if isinstance(batches, dict):
        if len(img_paths) != len(batches["left_image"]):
            left_images = batches["left_image"]
            batches["left_image"] = batches["left_image"][:len(img_paths)]
            batches["right_image"] = batches["right_image"][:len(img_paths)]
        else:
            left_images = batches["left_image"]
    else:
        left_images = batches
        if len(img_paths) != len(batches):
            batches = batches[:len(img_paths)]
    return left_images, batches


def apply_3d_mask(tensor, mask, value=0):
    """
    Apply a 3D boolean mask to a 3D tensor or array, preserving the original shape.

    This function is useful for masking depth maps or other 3D data while preserving
    the original dimensions. It replaces masked-out values with a specified fill value.
    The operation is performed element-wise using numpy.where for efficient computation.

    Args:
        tensor (np.ndarray): A 3D numpy array of any data type and shape (H, W, C).
        mask (np.ndarray): A 3D boolean mask array of the same shape as the input tensor.
            True values indicate positions to keep, False values will be replaced.
        value (float, optional): The value to fill in where the mask is False.
            Defaults to 0.

    Returns:
        np.ndarray: A new 3D array with the mask applied, preserving the original shape.
            Values where mask is False are replaced with the specified value.

    Raises:
        ValueError: If tensor and mask shapes do not match.

    Example:
        >>> depth_map = np.random.rand(100, 200, 1)
        >>> valid_mask = depth_map > 0.1
        >>> masked_depth = apply_3d_mask(depth_map, valid_mask, value=-1)
        >>> print(f"Original shape: {depth_map.shape}, Masked shape: {masked_depth.shape}")
    """
    return np.where(mask, tensor, np.full_like(tensor, value))


def vis_disparity(depth, normalize_depth=False, valid_mask=None):
    """
    Visualize disparity/depth map with color mapping and optional normalization.

    This function converts a depth/disparity map into a colorized visualization
    using the 'turbo' colormap. It supports both normalized and raw depth values,
    and can apply a valid mask to exclude invalid regions. The output is in BGR
    format suitable for OpenCV operations and file saving.

    Args:
        depth (np.ndarray): Input depth/disparity map of shape (H, W) or (H, W, 1).
            Values should be in the appropriate depth/disparity units.
        normalize_depth (bool, optional): Whether to normalize the depth values to [0, 255]
            before applying colormap. If False, uses min-max normalization. Defaults to False.
        valid_mask (np.ndarray, optional): Boolean mask indicating valid depth regions.
            Invalid regions will be masked out before visualization. Defaults to None.

    Returns:
        np.ndarray: Colorized depth visualization of shape (H, W, 3) in BGR format
            suitable for OpenCV operations and file saving.

    Raises:
        ValueError: If depth array is empty or contains only invalid values.

    Example:
        >>> depth_map = np.random.rand(480, 640)
        >>> colored_depth = vis_disparity(depth_map, normalize_depth=True)
        >>> cv2.imwrite('depth_visualization.png', colored_depth)

        >>> # With valid mask
        >>> valid_mask = depth_map > 0.1
        >>> colored_depth = vis_disparity(depth_map, valid_mask=valid_mask)
    """
    depth = depth.copy()

    if valid_mask is not None:
        depth = apply_3d_mask(depth, valid_mask)

    if normalize_depth:
        depth = depth * 255.0
    else:
        if depth.max() == depth.min():
            depth = depth = (depth / depth.max()) * 255.0
        else:
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    cmap = matplotlib.colormaps.get_cmap('turbo')
    vis = depth.astype(np.uint8)
    vis = (cmap(vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    return vis


def read_pfm(file_name, flip_up_down=False):
    """
    Read a PFM (Portable Float Map) file and return the data as a numpy array.

    PFM is a simple image format designed for high dynamic range images, commonly
    used for storing depth maps and disparity maps. This function supports both
    grayscale (Pf) and color (PF) PFM files with automatic endianness detection.

    Args:
        file_name (str): Path to the PFM file to read. The file must exist and
            be a valid PFM format file.
        flip_up_down (bool, optional): Whether to flip the image vertically.
            Useful when the PFM file has a different coordinate system convention.
            Defaults to False.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The image data as a numpy array of shape (H, W) for grayscale
              or (H, W, 3) for color images. Data type is float32.
            - float: The scale factor from the PFM file header, used for proper
              interpretation of the data values.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        TypeError: If the file is not a valid PFM file (invalid header).
        RuntimeError: If the PFM header is malformed or cannot be parsed.

    Example:
        >>> data, scale = read_pfm('depth_map.pfm', flip_up_down=True)
        >>> print(f"Image shape: {data.shape}, Scale: {scale}")
        >>> # Apply scale factor if needed
        >>> scaled_data = data * scale
    """
    with open(file_name, 'rb') as pfm_file:
        header = pfm_file.readline().rstrip()
        if header.decode('ascii') == 'PF':
            color = True
        elif header.decode('ascii') == 'Pf':
            color = False
        else:
            raise TypeError(f'Not a PFM file. {header} file: {file_name}')

        dim_match = re.search(r'(\d+)\s(\d+)', pfm_file.readline().decode('ascii'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise RuntimeError(f'Malformed PFM header. {file_name}')

        scale = float(pfm_file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(pfm_file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        if flip_up_down:
            data = np.flip(data, axis=0)
        return data, scale


def read_gt_depth(gt_path, scale=1000):
    """
    Read a ground truth depth map from disk, supporting multiple file formats.

    This function can read depth maps stored in PFM or PNG formats. For PNG files,
    it assumes the depth is encoded in the RGB channels using a specific encoding
    scheme where depth = R*255*255 + G*255 + B, scaled by the provided scale factor.
    This encoding allows storing high-precision depth values in standard PNG format.

    Args:
        gt_path (str): Path to the ground truth depth file. Must be a valid file
            with .pfm or .png extension.
        scale (float, optional): Scale factor for PNG depth encoding.
            Used to convert the encoded depth values to actual depth units.
            For example, if depth is stored in millimeters, scale=1000 converts
            to meters. Defaults to 1000.

    Returns:
        np.ndarray: The depth map as a numpy array of shape (H, W). Invalid depth
            values (0) are replaced with infinity to indicate missing data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        NotImplementedError: If the file extension is not supported (.pfm or .png).
        ValueError: If the PNG file cannot be read or decoded properly.

    Example:
        >>> depth_map = read_gt_depth('gt_depth.pfm')
        >>> depth_map_png = read_gt_depth('gt_depth.png', scale=1000)
        >>> print(f"Depth range: {depth_map.min():.3f} to {depth_map.max():.3f}")
        >>> # Handle infinite values
        >>> valid_mask = np.isfinite(depth_map)
        >>> valid_depth = depth_map[valid_mask]
    """
    if '.pfm' in gt_path:
        disp, _ = read_pfm(gt_path, flip_up_down=True)
    elif '.png' in gt_path:
        disp = cv2.imread(gt_path)[..., ::-1]
        disp = disp.astype(float)
        disp = disp[..., 0] * 255 * 255 + disp[..., 1] * 255 + disp[..., 2]
        disp = disp / float(scale)
        disp[disp == 0] = np.inf
        return disp
    else:
        raise NotImplementedError(f"Unsupported file extension: {gt_path}")
    return disp
