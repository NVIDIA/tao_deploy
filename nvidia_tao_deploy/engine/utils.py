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

"""TRT engine building base utilities."""

import logging
import numpy as np
from PIL import Image
from six.moves import xrange
from tqdm import tqdm

from nvidia_tao_deploy.engine.tensorfile import TensorFile

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


def generate_random_tensorfile(data_file_name, input_dims, n_batches=1, batch_size=1):
    """Generate a random tensorfile.

    This function generates a random tensorfile containing n_batches of random np.arrays
    of dimensions (batch_size,) + (input_dims).

    Args:
        data_file_name (str): Path to where the data tensorfile will be stored.
        input_dims (tuple): Input blob dimensions in CHW order.
        n_batches (int): Number of batches to save.
        batch_size (int): Number of images per batch.

    Return:
        No explicit returns.
    """
    sample_shape = (batch_size, ) + tuple(input_dims)
    with TensorFile(data_file_name, 'w') as f:
        for i in tqdm(xrange(n_batches)):
            logger.debug("Writing batch: %d", i)
            dump_sample = np.random.sample(sample_shape)
            f.write(dump_sample)


def prepare_chunk(image_ids, image_list,
                  image_width=480,
                  image_height=272,
                  channels=3,
                  scale=1.0,
                  means=None,
                  flip_channel=False,
                  batch_size=1):
    """Prepare a single batch of data to dump into a Tensorfile."""
    dump_placeholder = np.zeros(
        (batch_size, channels, image_height, image_width))
    for i in xrange(len(image_ids)):
        idx = image_ids[i]
        im = Image.open(image_list[idx]).resize((image_width, image_height),
                                                Image.LANCZOS)
        if channels == 1:
            logger.debug("Converting image from RGB to Grayscale")
            if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                bg_colour = (255, 255, 255)
                # Need to convert to RGBA if LA format due to a bug in PIL
                alpha = im.convert('RGBA').split()[-1]
                # Create a new background image of our matt color.
                # Must be RGBA because paste requires both images have the same format
                bg = Image.new("RGBA", im.size, bg_colour + (255,))
                bg.paste(im, mask=alpha)

            im = im.convert('L')
            dump_input = np.asarray(im).astype(np.float32)
            dump_input = dump_input[:, :, np.newaxis]
        elif channels == 3:
            dump_input = np.asarray(im.convert('RGB')).astype(np.float32)
        else:
            raise NotImplementedError("Unsupported channel dimensions.")
        # flip channel: RGB --> BGR
        if flip_channel:
            dump_input = dump_input[:, :, ::-1]
        # means is a list of per-channel means, (H, W, C) - (C)
        if means is not None:
            dump_input -= np.array(means)
        # (H, W, C) --> (C, H, W)
        dump_input = dump_input.transpose(2, 0, 1) * scale
        dump_placeholder[i, :, :, :] = dump_input
    return dump_placeholder
