# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/detectnet_v2/proto/learning_rate_config.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from nvidia_tao_deploy.cv.detectnet_v2.proto import soft_start_annealing_schedule_config_pb2 as nvidia__tao__deploy_dot_cv_dot_detectnet__v2_dot_proto_dot_soft__start__annealing__schedule__config__pb2
from nvidia_tao_deploy.cv.detectnet_v2.proto import early_stopping_annealing_schedule_config_pb2 as nvidia__tao__deploy_dot_cv_dot_detectnet__v2_dot_proto_dot_early__stopping__annealing__schedule__config__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\nBnvidia_tao_deploy/cv/detectnet_v2/proto/learning_rate_config.proto\x1aRnvidia_tao_deploy/cv/detectnet_v2/proto/soft_start_annealing_schedule_config.proto\x1aVnvidia_tao_deploy/cv/detectnet_v2/proto/early_stopping_annealing_schedule_config.proto\"\xc5\x01\n\x12LearningRateConfig\x12J\n\x1dsoft_start_annealing_schedule\x18\x01 \x01(\x0b\x32!.SoftStartAnnealingScheduleConfigH\x00\x12R\n!early_stopping_annealing_schedule\x18\x02 \x01(\x0b\x32%.EarlyStoppingAnnealingScheduleConfigH\x00\x42\x0f\n\rlearning_rateb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'nvidia_tao_deploy.cv.detectnet_v2.proto.learning_rate_config_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _LEARNINGRATECONFIG._serialized_start=243
  _LEARNINGRATECONFIG._serialized_end=440
# @@protoc_insertion_point(module_scope)
