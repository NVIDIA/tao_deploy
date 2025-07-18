# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/ssd/proto/experiment.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from nvidia_tao_deploy.cv.ssd.proto import augmentation_config_pb2 as nvidia__tao__deploy_dot_cv_dot_ssd_dot_proto_dot_augmentation__config__pb2
from nvidia_tao_deploy.cv.common.proto import detection_sequence_dataset_config_pb2 as nvidia__tao__deploy_dot_cv_dot_common_dot_proto_dot_detection__sequence__dataset__config__pb2
from nvidia_tao_deploy.cv.common.proto import training_config_pb2 as nvidia__tao__deploy_dot_cv_dot_common_dot_proto_dot_training__config__pb2
from nvidia_tao_deploy.cv.common.proto import nms_config_pb2 as nvidia__tao__deploy_dot_cv_dot_common_dot_proto_dot_nms__config__pb2
from nvidia_tao_deploy.cv.ssd.proto import eval_config_pb2 as nvidia__tao__deploy_dot_cv_dot_ssd_dot_proto_dot_eval__config__pb2
from nvidia_tao_deploy.cv.ssd.proto import ssd_config_pb2 as nvidia__tao__deploy_dot_cv_dot_ssd_dot_proto_dot_ssd__config__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n/nvidia_tao_deploy/cv/ssd/proto/experiment.proto\x1a\x38nvidia_tao_deploy/cv/ssd/proto/augmentation_config.proto\x1aInvidia_tao_deploy/cv/common/proto/detection_sequence_dataset_config.proto\x1a\x37nvidia_tao_deploy/cv/common/proto/training_config.proto\x1a\x32nvidia_tao_deploy/cv/common/proto/nms_config.proto\x1a\x30nvidia_tao_deploy/cv/ssd/proto/eval_config.proto\x1a/nvidia_tao_deploy/cv/ssd/proto/ssd_config.proto\"\xb7\x02\n\nExperiment\x12\x13\n\x0brandom_seed\x18\x01 \x01(\r\x12&\n\x0e\x64\x61taset_config\x18\x02 \x01(\x0b\x32\x0e.DatasetConfig\x12\x30\n\x13\x61ugmentation_config\x18\x03 \x01(\x0b\x32\x13.AugmentationConfig\x12(\n\x0ftraining_config\x18\x04 \x01(\x0b\x32\x0f.TrainingConfig\x12 \n\x0b\x65val_config\x18\x05 \x01(\x0b\x32\x0b.EvalConfig\x12\x1e\n\nnms_config\x18\x06 \x01(\x0b\x32\n.NMSConfig\x12 \n\nssd_config\x18\x07 \x01(\x0b\x32\n.SSDConfigH\x00\x12!\n\x0b\x64ssd_config\x18\x08 \x01(\x0b\x32\n.SSDConfigH\x00\x42\t\n\x07networkb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'nvidia_tao_deploy.cv.ssd.proto.experiment_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _EXPERIMENT._serialized_start=393
  _EXPERIMENT._serialized_end=704
# @@protoc_insertion_point(module_scope)
