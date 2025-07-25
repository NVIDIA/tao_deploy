# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/detectnet_v2/proto/dataset_export_config.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from nvidia_tao_deploy.cv.detectnet_v2.proto import kitti_config_pb2 as nvidia__tao__deploy_dot_cv_dot_detectnet__v2_dot_proto_dot_kitti__config__pb2
from nvidia_tao_deploy.cv.detectnet_v2.proto import coco_config_pb2 as nvidia__tao__deploy_dot_cv_dot_detectnet__v2_dot_proto_dot_coco__config__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\nCnvidia_tao_deploy/cv/detectnet_v2/proto/dataset_export_config.proto\x1a:nvidia_tao_deploy/cv/detectnet_v2/proto/kitti_config.proto\x1a\x39nvidia_tao_deploy/cv/detectnet_v2/proto/coco_config.proto\"\xec\x06\n\x13\x44\x61tasetExportConfig\x12\"\n\x0b\x63oco_config\x18\x01 \x01(\x0b\x32\x0b.COCOConfigH\x00\x12$\n\x0ckitti_config\x18\x02 \x01(\x0b\x32\x0c.KITTIConfigH\x00\x12I\n\x16sample_modifier_config\x18\x05 \x01(\x0b\x32).DatasetExportConfig.SampleModifierConfig\x12\x1c\n\x14image_directory_path\x18\x06 \x01(\t\x12J\n\x14target_class_mapping\x18\x07 \x03(\x0b\x32,.DatasetExportConfig.TargetClassMappingEntry\x1a\x83\x04\n\x14SampleModifierConfig\x12&\n\x1e\x66ilter_samples_containing_only\x18\x01 \x03(\t\x12\x1f\n\x17\x64ominant_target_classes\x18\x02 \x03(\t\x12r\n\x1eminimum_target_class_imbalance\x18\x03 \x03(\x0b\x32J.DatasetExportConfig.SampleModifierConfig.MinimumTargetClassImbalanceEntry\x12\x16\n\x0enum_duplicates\x18\x04 \x01(\r\x12\x1c\n\x14max_training_samples\x18\x05 \x01(\r\x12q\n\x1esource_to_target_class_mapping\x18\x06 \x03(\x0b\x32I.DatasetExportConfig.SampleModifierConfig.SourceToTargetClassMappingEntry\x1a\x42\n MinimumTargetClassImbalanceEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\x1a\x41\n\x1fSourceToTargetClassMappingEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x39\n\x17TargetClassMappingEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x42\x15\n\x13\x63onvert_config_typeb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'nvidia_tao_deploy.cv.detectnet_v2.proto.dataset_export_config_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _DATASETEXPORTCONFIG_SAMPLEMODIFIERCONFIG_MINIMUMTARGETCLASSIMBALANCEENTRY._options = None
  _DATASETEXPORTCONFIG_SAMPLEMODIFIERCONFIG_MINIMUMTARGETCLASSIMBALANCEENTRY._serialized_options = b'8\001'
  _DATASETEXPORTCONFIG_SAMPLEMODIFIERCONFIG_SOURCETOTARGETCLASSMAPPINGENTRY._options = None
  _DATASETEXPORTCONFIG_SAMPLEMODIFIERCONFIG_SOURCETOTARGETCLASSMAPPINGENTRY._serialized_options = b'8\001'
  _DATASETEXPORTCONFIG_TARGETCLASSMAPPINGENTRY._options = None
  _DATASETEXPORTCONFIG_TARGETCLASSMAPPINGENTRY._serialized_options = b'8\001'
  _DATASETEXPORTCONFIG._serialized_start=191
  _DATASETEXPORTCONFIG._serialized_end=1067
  _DATASETEXPORTCONFIG_SAMPLEMODIFIERCONFIG._serialized_start=470
  _DATASETEXPORTCONFIG_SAMPLEMODIFIERCONFIG._serialized_end=985
  _DATASETEXPORTCONFIG_SAMPLEMODIFIERCONFIG_MINIMUMTARGETCLASSIMBALANCEENTRY._serialized_start=852
  _DATASETEXPORTCONFIG_SAMPLEMODIFIERCONFIG_MINIMUMTARGETCLASSIMBALANCEENTRY._serialized_end=918
  _DATASETEXPORTCONFIG_SAMPLEMODIFIERCONFIG_SOURCETOTARGETCLASSMAPPINGENTRY._serialized_start=920
  _DATASETEXPORTCONFIG_SAMPLEMODIFIERCONFIG_SOURCETOTARGETCLASSMAPPINGENTRY._serialized_end=985
  _DATASETEXPORTCONFIG_TARGETCLASSMAPPINGENTRY._serialized_start=987
  _DATASETEXPORTCONFIG_TARGETCLASSMAPPINGENTRY._serialized_end=1044
# @@protoc_insertion_point(module_scope)
