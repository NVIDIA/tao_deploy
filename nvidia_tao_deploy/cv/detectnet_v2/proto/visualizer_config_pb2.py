# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/detectnet_v2/proto/visualizer_config.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from nvidia_tao_deploy.cv.common.proto import clearml_config_pb2 as nvidia__tao__deploy_dot_cv_dot_common_dot_proto_dot_clearml__config__pb2
from nvidia_tao_deploy.cv.common.proto import wandb_config_pb2 as nvidia__tao__deploy_dot_cv_dot_common_dot_proto_dot_wandb__config__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n?nvidia_tao_deploy/cv/detectnet_v2/proto/visualizer_config.proto\x1a\x36nvidia_tao_deploy/cv/common/proto/clearml_config.proto\x1a\x34nvidia_tao_deploy/cv/common/proto/wandb_config.proto\"\xa2\x03\n\x10VisualizerConfig\x12\x0f\n\x07\x65nabled\x18\x01 \x01(\x08\x12\x12\n\nnum_images\x18\x02 \x01(\r\x12 \n\x18scalar_logging_frequency\x18\x03 \x01(\r\x12$\n\x1cinfrequent_logging_frequency\x18\x04 \x01(\r\x12\x45\n\x13target_class_config\x18\x05 \x03(\x0b\x32(.VisualizerConfig.TargetClassConfigEntry\x12\"\n\x0cwandb_config\x18\x06 \x01(\x0b\x32\x0c.WandBConfig\x12&\n\x0e\x63learml_config\x18\x07 \x01(\x0b\x32\x0e.ClearMLConfig\x1a/\n\x11TargetClassConfig\x12\x1a\n\x12\x63overage_threshold\x18\x01 \x01(\x02\x1a]\n\x16TargetClassConfigEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x32\n\x05value\x18\x02 \x01(\x0b\x32#.VisualizerConfig.TargetClassConfig:\x02\x38\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'nvidia_tao_deploy.cv.detectnet_v2.proto.visualizer_config_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _VISUALIZERCONFIG_TARGETCLASSCONFIGENTRY._options = None
  _VISUALIZERCONFIG_TARGETCLASSCONFIGENTRY._serialized_options = b'8\001'
  _VISUALIZERCONFIG._serialized_start=178
  _VISUALIZERCONFIG._serialized_end=596
  _VISUALIZERCONFIG_TARGETCLASSCONFIG._serialized_start=454
  _VISUALIZERCONFIG_TARGETCLASSCONFIG._serialized_end=501
  _VISUALIZERCONFIG_TARGETCLASSCONFIGENTRY._serialized_start=503
  _VISUALIZERCONFIG_TARGETCLASSCONFIGENTRY._serialized_end=596
# @@protoc_insertion_point(module_scope)
