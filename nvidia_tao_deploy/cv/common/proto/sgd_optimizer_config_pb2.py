# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/common/proto/sgd_optimizer_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nvidia_tao_deploy/cv/common/proto/sgd_optimizer_config.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n<nvidia_tao_deploy/cv/common/proto/sgd_optimizer_config.proto\"8\n\x12SGDOptimizerConfig\x12\x10\n\x08momentum\x18\x01 \x01(\x02\x12\x10\n\x08nesterov\x18\x02 \x01(\x08\x62\x06proto3')
)




_SGDOPTIMIZERCONFIG = _descriptor.Descriptor(
  name='SGDOptimizerConfig',
  full_name='SGDOptimizerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='momentum', full_name='SGDOptimizerConfig.momentum', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nesterov', full_name='SGDOptimizerConfig.nesterov', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=64,
  serialized_end=120,
)

DESCRIPTOR.message_types_by_name['SGDOptimizerConfig'] = _SGDOPTIMIZERCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SGDOptimizerConfig = _reflection.GeneratedProtocolMessageType('SGDOptimizerConfig', (_message.Message,), dict(
  DESCRIPTOR = _SGDOPTIMIZERCONFIG,
  __module__ = 'nvidia_tao_deploy.cv.common.proto.sgd_optimizer_config_pb2'
  # @@protoc_insertion_point(class_scope:SGDOptimizerConfig)
  ))
_sym_db.RegisterMessage(SGDOptimizerConfig)


# @@protoc_insertion_point(module_scope)