# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/mask_rcnn/proto/data_config.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n6nvidia_tao_deploy/cv/mask_rcnn/proto/data_config.proto\"\xcb\x02\n\nDataConfig\x12\x12\n\nimage_size\x18\x01 \x01(\t\x12\x1a\n\x12\x61ugment_input_data\x18\x02 \x01(\x08\x12\x13\n\x0bnum_classes\x18\x03 \x01(\r\x12\"\n\x1askip_crowd_during_training\x18\x04 \x01(\x08\x12\x1d\n\x15training_file_pattern\x18\x06 \x01(\t\x12\x1f\n\x17validation_file_pattern\x18\x07 \x01(\t\x12\x15\n\rval_json_file\x18\x08 \x01(\t\x12\x14\n\x0c\x65val_samples\x18\t \x01(\r\x12\x1c\n\x14prefetch_buffer_size\x18\n \x01(\r\x12\x1b\n\x13shuffle_buffer_size\x18\x0b \x01(\r\x12\x11\n\tn_workers\x18\x0c \x01(\r\x12\x19\n\x11max_num_instances\x18\r \x01(\rb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'nvidia_tao_deploy.cv.mask_rcnn.proto.data_config_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _DATACONFIG._serialized_start=59
  _DATACONFIG._serialized_end=390
# @@protoc_insertion_point(module_scope)
