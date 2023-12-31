# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/common/proto/regularizer_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nvidia_tao_deploy/cv/common/proto/regularizer_config.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n:nvidia_tao_deploy/cv/common/proto/regularizer_config.proto\"\x8a\x01\n\x11RegularizerConfig\x12\x33\n\x04type\x18\x01 \x01(\x0e\x32%.RegularizerConfig.RegularizationType\x12\x0e\n\x06weight\x18\x02 \x01(\x02\"0\n\x12RegularizationType\x12\n\n\x06NO_REG\x10\x00\x12\x06\n\x02L1\x10\x01\x12\x06\n\x02L2\x10\x02\x62\x06proto3')
)



_REGULARIZERCONFIG_REGULARIZATIONTYPE = _descriptor.EnumDescriptor(
  name='RegularizationType',
  full_name='RegularizerConfig.RegularizationType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NO_REG', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='L1', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='L2', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=153,
  serialized_end=201,
)
_sym_db.RegisterEnumDescriptor(_REGULARIZERCONFIG_REGULARIZATIONTYPE)


_REGULARIZERCONFIG = _descriptor.Descriptor(
  name='RegularizerConfig',
  full_name='RegularizerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='RegularizerConfig.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight', full_name='RegularizerConfig.weight', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _REGULARIZERCONFIG_REGULARIZATIONTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=63,
  serialized_end=201,
)

_REGULARIZERCONFIG.fields_by_name['type'].enum_type = _REGULARIZERCONFIG_REGULARIZATIONTYPE
_REGULARIZERCONFIG_REGULARIZATIONTYPE.containing_type = _REGULARIZERCONFIG
DESCRIPTOR.message_types_by_name['RegularizerConfig'] = _REGULARIZERCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RegularizerConfig = _reflection.GeneratedProtocolMessageType('RegularizerConfig', (_message.Message,), dict(
  DESCRIPTOR = _REGULARIZERCONFIG,
  __module__ = 'nvidia_tao_deploy.cv.common.proto.regularizer_config_pb2'
  # @@protoc_insertion_point(class_scope:RegularizerConfig)
  ))
_sym_db.RegisterMessage(RegularizerConfig)


# @@protoc_insertion_point(module_scope)
