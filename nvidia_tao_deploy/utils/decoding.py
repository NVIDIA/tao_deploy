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

"""EfficientDet convert etlt model to TRT engine."""

import logging
import os
import struct
import tempfile
from zipfile import ZipFile
from eff.core import Archive

from eff_tao_encryption.tao_codec import decrypt_stream, encrypt_stream

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


def decode_model(model_path, key=None):
    """Decrypt the model."""
    if model_path.endswith('etlt'):
        return decode_etlt(model_path, key)
    if model_path.endswith('onnx'):
        return model_path, 'onnx'
    if model_path.endswith('uff'):
        return model_path, 'uff'
    raise NotImplementedError(
        "TAO Deploy gen_trt_engine action accepts either .onnx, .uff "
        "or .etlt file extension"
    )


def decode_etlt(etlt_model_path, key):
    """Decrypt ETLT model."""
    _handle, decrypted_model = tempfile.mkstemp()
    os.close(_handle)
    with open(decrypted_model, 'wb') as temp_file, open(etlt_model_path, 'rb') as encoded_file:
        size = encoded_file.read(4)
        size = struct.unpack("<i", size)[0]
        input_node_name = encoded_file.read(size)
        if size:
            # ETLT is in UFF format
            logging.info("The provided .etlt file is in UFF format.")
            logging.info("Input name: %s", input_node_name)
            file_format = "uff"
        else:
            # ETLT is in ONNX format
            logging.info("The provided .etlt file is in ONNX format.")
            file_format = "onnx"
        decrypt_stream(encoded_file, temp_file, key.encode(), encryption=True, rewind=False)
    return decrypted_model, file_format


def decode_eff(eff_model_path, key):
    """Decrypt EFF."""
    eff_filename = os.path.basename(eff_model_path)
    eff_art = Archive.restore_artifact(
        restore_path=eff_model_path,
        artifact_name=eff_filename,
        passphrase=key)
    zip_path = eff_art.get_handle()
    # Unzip
    ckpt_path = os.path.dirname(zip_path)
    # TODO(@yuw): try catch?
    with ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(ckpt_path)
    extracted_files = os.listdir(ckpt_path)
    # TODO(@yuw): get onnx path
    ckpt_name = None
    for f in extracted_files:
        if 'ckpt' in f:
            ckpt_name = f.split('.')[0]
    return ckpt_path, ckpt_name


def encode_etlt(tmp_file_name, output_file_name, input_tensor_name, key):
    """Encrypt ETLT model."""
    # Encode temporary uff to output file
    with open(tmp_file_name, "rb") as open_temp_file, \
         open(output_file_name, "wb") as open_encoded_file:
        # TODO: @vpraveen: Remove this hack to support multiple input nodes.
        # This will require an update to tlt_converter and DS. Postponing this for now.
        if isinstance(input_tensor_name, list):
            input_tensor_name = input_tensor_name[0]
        open_encoded_file.write(struct.pack("<i", len(input_tensor_name)))
        open_encoded_file.write(input_tensor_name.encode())
        encrypt_stream(open_temp_file,
                       open_encoded_file,
                       key, encryption=True, rewind=False)
