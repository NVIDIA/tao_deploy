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

"""UNet TensorRT engine builder."""

import logging
import os
import random
from six.moves import xrange
import sys
import onnx

from tqdm import tqdm

import tensorrt as trt

from nvidia_tao_deploy.cv.common.constants import VALID_IMAGE_EXTENSIONS
from nvidia_tao_deploy.engine.builder import EngineBuilder
from nvidia_tao_deploy.engine.tensorfile import TensorFile
from nvidia_tao_deploy.engine.tensorfile_calibrator import TensorfileCalibrator
from nvidia_tao_deploy.engine.utils import prepare_chunk


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


class UNetEngineBuilder(EngineBuilder):
    """Parses an UFF/ONNX graph and builds a TensorRT engine from it."""

    def __init__(
        self,
        image_list=None,
        data_format="channels_first",
        **kwargs
    ):
        """Init.

        Args:
            data_format (str): data_format.
            image_list (list): list of training image list from the experiment spec.
        """
        super().__init__(**kwargs)
        self._data_format = data_format
        self.training_image_list = image_list

    def get_onnx_input_dims(self, model_path):
        """Get input dimension of ONNX model."""
        onnx_model = onnx.load(model_path)
        onnx_inputs = onnx_model.graph.input
        logger.info('List inputs:')
        for i, inputs in enumerate(onnx_inputs):
            logger.info('Input %s -> %s.', i, inputs.name)
            logger.info('%s.', [i.dim_value for i in inputs.type.tensor_type.shape.dim][1:])
            logger.info('%s.', [i.dim_value for i in inputs.type.tensor_type.shape.dim][0])
            return [i.dim_value for i in inputs.type.tensor_type.shape.dim][:]

    def create_network(self, model_path, file_format="onnx"):
        """Parse the UFF/ONNX graph and create the corresponding TensorRT network definition.

        Args:
            model_path: The path to the UFF/ONNX graph to load.
            file_format: The file format of the decrypted etlt file (default: onnx).
        """
        if file_format == "onnx":
            logger.info("Parsing ONNX model")
            self._input_dims = self.get_onnx_input_dims(model_path)
            self.batch_size = self._input_dims[0]

            network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

            self.network = self.builder.create_network(network_flags)
            self.parser = trt.OnnxParser(self.network, self.trt_logger)

            model_path = os.path.realpath(model_path)
            with open(model_path, "rb") as f:
                if not self.parser.parse(f.read()):
                    logger.error("Failed to load ONNX file: %s", model_path)
                    for error in range(self.parser.num_errors):
                        logger.error(self.parser.get_error(error))
                    sys.exit(1)

            inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
            outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

            logger.info("Network Description")
            for input in inputs: # noqa pylint: disable=W0622
                logger.info("Input '%s' with shape %s and dtype %s", input.name, input.shape, input.dtype)
            for output in outputs:
                logger.info("Output '%s' with shape %s and dtype %s", output.name, output.shape, output.dtype)

            if self.batch_size <= 0:  # dynamic batch size
                logger.info("dynamic batch size handling")
                opt_profile = self.builder.create_optimization_profile()
                model_input = self.network.get_input(0)
                input_shape = model_input.shape
                input_name = model_input.name
                real_shape_min = (self.min_batch_size, input_shape[1],
                                  input_shape[2], input_shape[3])
                real_shape_opt = (self.opt_batch_size, input_shape[1],
                                  input_shape[2], input_shape[3])
                real_shape_max = (self.max_batch_size, input_shape[1],
                                  input_shape[2], input_shape[3])
                opt_profile.set_shape(input=input_name,
                                      min=real_shape_min,
                                      opt=real_shape_opt,
                                      max=real_shape_max)
                self.config.add_optimization_profile(opt_profile)
        else:
            logger.info("Parsing UFF model")
            raise NotImplementedError("UFF for YOLO_v3 is not supported")

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
                                      self._input_dims[1:],
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
        # Preparing the list of images to be saved.
        num_images = n_batches * batch_size
        if os.path.exists(calibration_images_dir):
            image_list = []
            for image in os.listdir(calibration_images_dir):
                if image.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    image_list.append(os.path.join(calibration_images_dir, image))
        else:
            logger.info("Calibration image directory is not specified. Using training images from experiment spec!")
            if self.training_image_list[0].endswith(".txt"):
                image_list = []
                for imgs in self.training_image_list:

                    # Read image files
                    with open(imgs, encoding="utf-8") as f:
                        x_set = f.readlines()

                    for f_im in x_set:
                        # Ensuring all image files are present
                        f_im = f_im.strip()
                        if f_im.lower().endswith(VALID_IMAGE_EXTENSIONS):
                            image_list.append(f_im)
            else:
                image_list = [os.path.join(self.training_image_list[0], f) for f in os.listdir(self.training_image_list[0])
                              if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]

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
                means = [127.5, 127.5, 127.5]
            else:
                assert len(image_mean) == 3, "Image mean should have 3 values for RGB inputs."
                means = image_mean
        elif num_channels == 1:
            if not image_mean:
                means = [127]
            else:
                assert len(image_mean) == 1, "Image mean should have 1 value for grayscale inputs."
                means = image_mean
        else:
            raise NotImplementedError(
                f"Invalid number of dimensions {num_channels}.")
        self.preprocessing_arguments = {"scale": 1.0 / 127.5,
                                        "means": means,
                                        "flip_channel": True}
