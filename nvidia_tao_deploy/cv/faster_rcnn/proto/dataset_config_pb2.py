# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/faster_rcnn/proto/dataset_config.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n;nvidia_tao_deploy/cv/faster_rcnn/proto/dataset_config.proto\"Y\n\nDataSource\x12\x16\n\x0etfrecords_path\x18\x01 \x01(\t\x12\x1c\n\x14image_directory_path\x18\x02 \x01(\t\x12\x15\n\rsource_weight\x18\x03 \x01(\x02\"\x99\x04\n\rDatasetConfig\x12!\n\x0c\x64\x61ta_sources\x18\x01 \x03(\x0b\x32\x0b.DataSource\x12\x17\n\x0fimage_extension\x18\x02 \x01(\t\x12\x44\n\x14target_class_mapping\x18\x03 \x03(\x0b\x32&.DatasetConfig.TargetClassMappingEntry\x12\x19\n\x0fvalidation_fold\x18\x04 \x01(\rH\x00\x12-\n\x16validation_data_source\x18\x05 \x01(\x0b\x32\x0b.DataSourceH\x00\x12\x37\n\x0f\x64\x61taloader_mode\x18\x06 \x01(\x0e\x32\x1e.DatasetConfig.DATALOADER_MODE\x12\x33\n\rsampling_mode\x18\x07 \x01(\x0e\x32\x1c.DatasetConfig.SAMPLING_MODE\x1a\x39\n\x17TargetClassMappingEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\";\n\x0f\x44\x41TALOADER_MODE\x12\x0f\n\x0bMULTISOURCE\x10\x00\x12\n\n\x06LEGACY\x10\x01\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x02\"@\n\rSAMPLING_MODE\x12\x10\n\x0cUSER_DEFINED\x10\x00\x12\x10\n\x0cPROPORTIONAL\x10\x01\x12\x0b\n\x07UNIFORM\x10\x02\x42\x14\n\x12\x64\x61taset_split_typeb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'nvidia_tao_deploy.cv.faster_rcnn.proto.dataset_config_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _DATASETCONFIG_TARGETCLASSMAPPINGENTRY._options = None
  _DATASETCONFIG_TARGETCLASSMAPPINGENTRY._serialized_options = b'8\001'
  _DATASOURCE._serialized_start=63
  _DATASOURCE._serialized_end=152
  _DATASETCONFIG._serialized_start=155
  _DATASETCONFIG._serialized_end=692
  _DATASETCONFIG_TARGETCLASSMAPPINGENTRY._serialized_start=486
  _DATASETCONFIG_TARGETCLASSMAPPINGENTRY._serialized_end=543
  _DATASETCONFIG_DATALOADER_MODE._serialized_start=545
  _DATASETCONFIG_DATALOADER_MODE._serialized_end=604
  _DATASETCONFIG_SAMPLING_MODE._serialized_start=606
  _DATASETCONFIG_SAMPLING_MODE._serialized_end=670
# @@protoc_insertion_point(module_scope)
