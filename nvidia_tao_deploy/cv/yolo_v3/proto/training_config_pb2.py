# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/yolo_v3/proto/training_config.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from nvidia_tao_deploy.cv.common.proto import cost_scaling_config_pb2 as nvidia__tao__deploy_dot_cv_dot_common_dot_proto_dot_cost__scaling__config__pb2
from nvidia_tao_deploy.cv.common.proto import learning_rate_config_pb2 as nvidia__tao__deploy_dot_cv_dot_common_dot_proto_dot_learning__rate__config__pb2
from nvidia_tao_deploy.cv.common.proto import optimizer_config_pb2 as nvidia__tao__deploy_dot_cv_dot_common_dot_proto_dot_optimizer__config__pb2
from nvidia_tao_deploy.cv.common.proto import regularizer_config_pb2 as nvidia__tao__deploy_dot_cv_dot_common_dot_proto_dot_regularizer__config__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n8nvidia_tao_deploy/cv/yolo_v3/proto/training_config.proto\x1a;nvidia_tao_deploy/cv/common/proto/cost_scaling_config.proto\x1a<nvidia_tao_deploy/cv/common/proto/learning_rate_config.proto\x1a\x38nvidia_tao_deploy/cv/common/proto/optimizer_config.proto\x1a:nvidia_tao_deploy/cv/common/proto/regularizer_config.proto\"\xc4\x03\n\x0eTrainingConfig\x12\x1a\n\x12\x62\x61tch_size_per_gpu\x18\x01 \x01(\r\x12\x12\n\nnum_epochs\x18\x02 \x01(\r\x12*\n\rlearning_rate\x18\x03 \x01(\x0b\x32\x13.LearningRateConfig\x12\'\n\x0bregularizer\x18\x04 \x01(\x0b\x32\x12.RegularizerConfig\x12#\n\toptimizer\x18\x05 \x01(\x0b\x32\x10.OptimizerConfig\x12(\n\x0c\x63ost_scaling\x18\x06 \x01(\x0b\x32\x12.CostScalingConfig\x12\x1b\n\x13\x63heckpoint_interval\x18\x07 \x01(\r\x12\x12\n\nenable_qat\x18\x08 \x01(\x08\x12\x1b\n\x11resume_model_path\x18\t \x01(\tH\x00\x12\x1d\n\x13pretrain_model_path\x18\n \x01(\tH\x00\x12\x1b\n\x11pruned_model_path\x18\x0b \x01(\tH\x00\x12\x16\n\x0emax_queue_size\x18\x0c \x01(\r\x12\x11\n\tn_workers\x18\r \x01(\r\x12\x1b\n\x13use_multiprocessing\x18\x0e \x01(\x08\x42\x0c\n\nload_modelb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'nvidia_tao_deploy.cv.yolo_v3.proto.training_config_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _TRAININGCONFIG._serialized_start=302
  _TRAININGCONFIG._serialized_end=754
# @@protoc_insertion_point(module_scope)
