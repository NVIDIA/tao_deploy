# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/faster_rcnn/proto/learning_rate.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n:nvidia_tao_deploy/cv/faster_rcnn/proto/learning_rate.proto\"\x86\x01\n\x18SoftStartAnnealingConfig\x12\x0f\n\x07\x62\x61se_lr\x18\x01 \x01(\x02\x12\x10\n\x08start_lr\x18\x02 \x01(\x02\x12\x12\n\nsoft_start\x18\x03 \x01(\x02\x12\x18\n\x10\x61nnealing_points\x18\x04 \x03(\x02\x12\x19\n\x11\x61nnealing_divider\x18\x05 \x01(\x02\"A\n\x0cStepLrConfig\x12\x0f\n\x07\x62\x61se_lr\x18\x01 \x01(\x02\x12\r\n\x05gamma\x18\x02 \x01(\x02\x12\x11\n\tstep_size\x18\x03 \x01(\x02\"g\n\x08LRConfig\x12/\n\nsoft_start\x18\x01 \x01(\x0b\x32\x19.SoftStartAnnealingConfigH\x00\x12\x1d\n\x04step\x18\x02 \x01(\x0b\x32\r.StepLrConfigH\x00\x42\x0b\n\tlr_configb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'nvidia_tao_deploy.cv.faster_rcnn.proto.learning_rate_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _SOFTSTARTANNEALINGCONFIG._serialized_start=63
  _SOFTSTARTANNEALINGCONFIG._serialized_end=197
  _STEPLRCONFIG._serialized_start=199
  _STEPLRCONFIG._serialized_end=264
  _LRCONFIG._serialized_start=266
  _LRCONFIG._serialized_end=369
# @@protoc_insertion_point(module_scope)
