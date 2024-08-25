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

"""MRCNN TensorRT engine builder."""

import logging
import os
import random
from six.moves import xrange
import sys
import traceback

from tqdm import tqdm

try:
    from uff.model.uff_pb2 import MetaGraph
except ImportError:
    print("Loading uff directly from the package source code")
    # @scha: To disable tensorflow import issue
    import importlib
    import types
    import pkgutil

    package = pkgutil.get_loader("uff")
    # Returns __init__.py path
    src_code = package.get_filename().replace('__init__.py', 'model/uff_pb2.py')

    loader = importlib.machinery.SourceFileLoader('helper', src_code)
    helper = types.ModuleType(loader.name)
    loader.exec_module(helper)
    MetaGraph = helper.MetaGraph

import numpy as np
import tensorrt as trt

from nvidia_tao_deploy.engine.builder import EngineBuilder
from nvidia_tao_deploy.engine.tensorfile import TensorFile
from nvidia_tao_deploy.engine.tensorfile_calibrator import TensorfileCalibrator
from nvidia_tao_deploy.engine.utils import generate_random_tensorfile, prepare_chunk


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


class MRCNNEngineBuilder(EngineBuilder):
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
        self._output_node_names = ["generate_detections", "mask_fcn_logits/BiasAdd"]
        self._input_node_names = ["Input"]

    def get_input_dims(self, model_path):
        """Get input dimension of UFF model."""
        metagraph = MetaGraph()
        with open(model_path, "rb") as f:
            metagraph.ParseFromString(f.read())
        for node in metagraph.graphs[0].nodes:
            if node.operation == "Input":
                return np.array(node.fields['shape'].i_list.val)[1:]
        raise ValueError("Input dimension is not found in the UFF metagraph.")

    def create_network(self, model_path, file_format="uff"):
        """Parse the ONNX graph and create the corresponding TensorRT network definition.

        Args:
            model_path: The path to the UFF/ONNX graph to load.
        """
        if file_format == "uff":
            logger.info("Parsing UFF model")
            self.network = self.builder.create_network()
            self.parser = trt.UffParser()

            self.set_input_output_node_names()

            in_tensor_name = self._input_node_names[0]

            self._input_dims = self.get_input_dims(model_path)
            input_dict = {in_tensor_name: self._input_dims}
            for key, value in input_dict.items():
                if self._data_format == "channels_first":
                    self.parser.register_input(key, value, trt.UffInputOrder(0))
                else:
                    self.parser.register_input(key, value, trt.UffInputOrder(1))
            for name in self._output_node_names:
                self.parser.register_output(name)
            self.builder.max_batch_size = self.max_batch_size
            try:
                assert self.parser.parse(model_path, self.network, trt.DataType.FLOAT)
            except AssertionError as e:
                logger.error("Failed to parse UFF File")
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)  # Fixed format
                tb_info = traceback.extract_tb(tb)
                _, line, _, text = tb_info[-1]
                raise AssertionError(
                    f"UFF parsing failed on line {line} in statement {text}"
                ) from e
        else:
            logger.info("Parsing UFF model")
            raise NotImplementedError("UFF for Faster RCNN is not supported")

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
                                      self._input_dims,
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
            means = [123.675, 116.280, 103.53]
        else:
            raise NotImplementedError(f"Invalid number of dimensions {num_channels}.")
        # ([R, G, B]/ 255 - [0.485, 0.456, 0.406]) / 0.224
        # (R/G/B - mean) * ratio
        self.preprocessing_arguments = {"scale": 0.017507,
                                        "means": means,
                                        "flip_channel": False}
