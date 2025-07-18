# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/detectnet_v2/proto/inference.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from nvidia_tao_deploy.cv.detectnet_v2.proto import inferencer_config_pb2 as nvidia__tao__deploy_dot_cv_dot_detectnet__v2_dot_proto_dot_inferencer__config__pb2
from nvidia_tao_deploy.cv.detectnet_v2.proto import postprocessing_config_pb2 as nvidia__tao__deploy_dot_cv_dot_detectnet__v2_dot_proto_dot_postprocessing__config__pb2
from nvidia_tao_deploy.cv.common.proto import wandb_config_pb2 as nvidia__tao__deploy_dot_cv_dot_common_dot_proto_dot_wandb__config__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n7nvidia_tao_deploy/cv/detectnet_v2/proto/inference.proto\x1a?nvidia_tao_deploy/cv/detectnet_v2/proto/inferencer_config.proto\x1a\x43nvidia_tao_deploy/cv/detectnet_v2/proto/postprocessing_config.proto\x1a\x34nvidia_tao_deploy/cv/common/proto/wandb_config.proto\"\xe1\x01\n\x1a\x43lasswiseBboxHandlerConfig\x12,\n\x11\x63lustering_config\x18\x01 \x01(\x0b\x32\x11.ClusteringConfig\x12\x18\n\x10\x63onfidence_model\x18\x02 \x01(\t\x12\x12\n\noutput_map\x18\x03 \x01(\t\x12\x39\n\nbbox_color\x18\x07 \x01(\x0b\x32%.ClasswiseBboxHandlerConfig.BboxColor\x1a,\n\tBboxColor\x12\t\n\x01R\x18\x01 \x01(\x05\x12\t\n\x01G\x18\x02 \x01(\x05\x12\t\n\x01\x42\x18\x03 \x01(\x05\"\xd4\x02\n\x11\x42\x62oxHandlerConfig\x12\x12\n\nkitti_dump\x18\x01 \x01(\x08\x12\x17\n\x0f\x64isable_overlay\x18\x02 \x01(\x08\x12\x19\n\x11overlay_linewidth\x18\x03 \x01(\x05\x12Y\n\x1d\x63lasswise_bbox_handler_config\x18\x04 \x03(\x0b\x32\x32.BboxHandlerConfig.ClasswiseBboxHandlerConfigEntry\x12\x18\n\x10postproc_classes\x18\x05 \x03(\t\x12\"\n\x0cwandb_config\x18\x06 \x01(\x0b\x32\x0c.WandBConfig\x1a^\n\x1f\x43lasswiseBboxHandlerConfigEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12*\n\x05value\x18\x02 \x01(\x0b\x32\x1b.ClasswiseBboxHandlerConfig:\x02\x38\x01\"j\n\tInference\x12,\n\x11inferencer_config\x18\x01 \x01(\x0b\x32\x11.InferencerConfig\x12/\n\x13\x62\x62ox_handler_config\x18\x02 \x01(\x0b\x32\x12.BboxHandlerConfigb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'nvidia_tao_deploy.cv.detectnet_v2.proto.inference_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _BBOXHANDLERCONFIG_CLASSWISEBBOXHANDLERCONFIGENTRY._options = None
  _BBOXHANDLERCONFIG_CLASSWISEBBOXHANDLERCONFIGENTRY._serialized_options = b'8\001'
  _CLASSWISEBBOXHANDLERCONFIG._serialized_start=248
  _CLASSWISEBBOXHANDLERCONFIG._serialized_end=473
  _CLASSWISEBBOXHANDLERCONFIG_BBOXCOLOR._serialized_start=429
  _CLASSWISEBBOXHANDLERCONFIG_BBOXCOLOR._serialized_end=473
  _BBOXHANDLERCONFIG._serialized_start=476
  _BBOXHANDLERCONFIG._serialized_end=816
  _BBOXHANDLERCONFIG_CLASSWISEBBOXHANDLERCONFIGENTRY._serialized_start=722
  _BBOXHANDLERCONFIG_CLASSWISEBBOXHANDLERCONFIGENTRY._serialized_end=816
  _INFERENCE._serialized_start=818
  _INFERENCE._serialized_end=924
# @@protoc_insertion_point(module_scope)
