# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/detectnet_v2/proto/coco_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nvidia_tao_deploy/cv/detectnet_v2/proto/coco_config.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n9nvidia_tao_deploy/cv/detectnet_v2/proto/coco_config.proto\"\x86\x01\n\nCOCOConfig\x12\x1b\n\x13root_directory_path\x18\x01 \x01(\t\x12\x15\n\rimg_dir_names\x18\x02 \x03(\t\x12\x18\n\x10\x61nnotation_files\x18\x03 \x03(\t\x12\x16\n\x0enum_partitions\x18\x04 \x01(\r\x12\x12\n\nnum_shards\x18\x05 \x03(\rb\x06proto3')
)




_COCOCONFIG = _descriptor.Descriptor(
  name='COCOConfig',
  full_name='COCOConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='root_directory_path', full_name='COCOConfig.root_directory_path', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='img_dir_names', full_name='COCOConfig.img_dir_names', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='annotation_files', full_name='COCOConfig.annotation_files', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_partitions', full_name='COCOConfig.num_partitions', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_shards', full_name='COCOConfig.num_shards', index=4,
      number=5, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=62,
  serialized_end=196,
)

DESCRIPTOR.message_types_by_name['COCOConfig'] = _COCOCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

COCOConfig = _reflection.GeneratedProtocolMessageType('COCOConfig', (_message.Message,), dict(
  DESCRIPTOR = _COCOCONFIG,
  __module__ = 'nvidia_tao_deploy.cv.detectnet_v2.proto.coco_config_pb2'
  # @@protoc_insertion_point(class_scope:COCOConfig)
  ))
_sym_db.RegisterMessage(COCOConfig)


# @@protoc_insertion_point(module_scope)