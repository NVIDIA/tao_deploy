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

"""RetinaNet TensorRT engine builder."""

import logging
import os
import random
from six.moves import xrange

from tqdm import tqdm

import tensorrt as trt

from nvidia_tao_deploy.engine.builder import EngineBuilder
from nvidia_tao_deploy.engine.tensorfile import TensorFile
from nvidia_tao_deploy.engine.tensorfile_calibrator import TensorfileCalibrator
from nvidia_tao_deploy.engine.utils import generate_random_tensorfile, prepare_chunk


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


class RetinaNetEngineBuilder(EngineBuilder):
    """Parses an UFF graph and builds a TensorRT engine from it."""

    def __init__(
        self,
        data_format="channels_first",
        **kwargs
    ):
        """Init.

        Args:
            data_format (str): data_format.
        """
        super().__init__(**kwargs)
        self._data_format = data_format

    def set_input_output_node_names(self):
        """Set input output node names."""
        self._output_node_names = ["NMS"]
        self._input_node_names = ["Input"]

    def create_network(self, model_path, file_format="uff"):
        """Parse the ONNX graph and create the corresponding TensorRT network definition.

        Args:
            model_path: The path to the UFF/ONNX graph to load.
            file_format: The file format of the decrypted etlt file (default: uff).
        """
        if file_format == "uff":
            self.parse_uff_model(model_path)
        else:
            super().create_network(model_path, file_format)

    def set_calibrator(self,
                       inputs=None,
                       calib_cache=None,
                       calib_input=None,
                       calib_num_images=5000,
                       calib_batch_size=8,
                       calib_data_file=None,
                       image_mean=None):
        """Simple function to set an Tensorfile based int8 calibrator.

        Args:
            calib_data_file: Path to the TensorFile. If the tensorfile doesn't exist
                             at this path, then one is created with either n_batches
                             of random tensors, images from the file in calib_input of dimensions
                             (batch_size,) + (input_dims).
            calib_input: The path to a directory holding the calibration images.
            calib_cache: The path where to write the calibration cache to,
                         or if it already exists, load it from.
            calib_num_images: The maximum number of images to use for calibration.
            calib_batch_size: The batch size to use for the calibration process.
            image_mean: Image mean per channel.

        Returns:
            No explicit returns.
        """
        logger.info("Calibrating using TensorfileCalibrator")

        n_batches = calib_num_images // calib_batch_size
        if not os.path.exists(calib_data_file):
            self.generate_tensor_file(calib_data_file,
                                      calib_input,
                                      list(self._input_dims.values())[0],
                                      n_batches=n_batches,
                                      batch_size=calib_batch_size,
                                      image_mean=image_mean)
        self.config.int8_calibrator = TensorfileCalibrator(calib_data_file,
                                                           calib_cache,
                                                           n_batches,
                                                           calib_batch_size)

    def generate_tensor_file(self, data_file_name,
                             calibration_images_dir,
                             input_dims, n_batches=10,
                             batch_size=1, image_mean=None):
        """Generate calibration Tensorfile for int8 calibrator.

        This function generates a calibration tensorfile from a directory of images, or dumps
        n_batches of random numpy arrays of shape (batch_size,) + (input_dims).

        Args:
            data_file_name (str): Path to the output tensorfile to be saved.
            calibration_images_dir (str): Path to the images to generate a tensorfile from.
            input_dims (list): Input shape in CHW order.
            n_batches (int): Number of batches to be saved.
            batch_size (int): Number of images per batch.
            image_mean (list): Image mean per channel.

        Returns:
            No explicit returns.
        """
        if not os.path.exists(calibration_images_dir):
            logger.info("Generating a tensorfile with random tensor images. This may work well as "
                        "a profiling tool, however, it may result in inaccurate results at "
                        "inference. Please generate a tensorfile using the tlt-int8-tensorfile, "
                        "or provide a custom directory of images for best performance.")
            generate_random_tensorfile(data_file_name,
                                       input_dims,
                                       n_batches=n_batches,
                                       batch_size=batch_size)
        else:
            # Preparing the list of images to be saved.
            num_images = n_batches * batch_size
            valid_image_ext = ['jpg', 'jpeg', 'png']
            image_list = [os.path.join(calibration_images_dir, image)
                          for image in os.listdir(calibration_images_dir)
                          if image.split('.')[-1] in valid_image_ext]
            if len(image_list) < num_images:
                raise ValueError('Not enough number of images provided:'
                                 f' {len(image_list)} < {num_images}')
            image_idx = random.sample(xrange(len(image_list)), num_images)
            self.set_data_preprocessing_parameters(input_dims, image_mean)
            # Writing out processed dump.
            with TensorFile(data_file_name, 'w') as f:
                for chunk in tqdm(image_idx[x:x + batch_size] for x in xrange(0, len(image_idx),
                                                                              batch_size)):
                    dump_data = prepare_chunk(chunk, image_list,
                                              image_width=input_dims[2],
                                              image_height=input_dims[1],
                                              channels=input_dims[0],
                                              batch_size=batch_size,
                                              **self.preprocessing_arguments)
                    f.write(dump_data)
            f.closed

    def set_data_preprocessing_parameters(self, input_dims, image_mean=None):
        """Set data pre-processing parameters for the int8 calibration."""
        num_channels = input_dims[0]
        if num_channels == 3:
            if not image_mean:
                means = [103.939, 116.779, 123.68]
            else:
                assert len(image_mean) == 3, "Image mean should have 3 values for RGB inputs."
                means = image_mean
        elif num_channels == 1:
            if not image_mean:
                means = [117.3786]
            else:
                assert len(image_mean) == 1, "Image mean should have 1 value for grayscale inputs."
                means = image_mean
        else:
            raise NotImplementedError(
                f"Invalid number of dimensions {num_channels}.")
        self.preprocessing_arguments = {"scale": 1.0,
                                        "means": means,
                                        "flip_channel": True}

    def _set_tensor_dynamic_ranges(self, network, tensor_scale_dict):
        """Set the scaling factors obtained from quantization-aware training.

        Args:
            network: TensorRT network object.
            tensor_scale_dict (dict): Dictionary mapping names to tensor scaling factors.
        """
        tensors_in_dict = tensor_scale_dict.keys()
        tensors_found = []
        for idx in range(network.num_inputs):
            input_tensor = network.get_input(idx)
            for k in tensor_scale_dict.keys():
                if k.startswith(str(input_tensor.name).split('/', maxsplit=1)[0]):
                    cal_scale = tensor_scale_dict.pop(k)
                    tensors_found.append(k)
                    input_tensor.dynamic_range = (-cal_scale, cal_scale)
                    break

        for layer in network:
            found_all_outputs = True
            for idx in range(layer.num_outputs):
                output_tensor = layer.get_output(idx)
                for k in tensor_scale_dict.keys():
                    if k.startswith(str(output_tensor.name).split('/', maxsplit=1)[0]):
                        cal_scale = tensor_scale_dict.pop(k)
                        tensors_found.append(k)
                        output_tensor.dynamic_range = (-cal_scale, cal_scale)
                        break
                    found_all_outputs = False
            if found_all_outputs:
                layer.precision = trt.int8
        if set(tensors_in_dict) != set(tensors_found):
            logger.info("Tensors in scale dictionary but not in network: %s",
                        set(tensors_in_dict) - set(tensors_found))
