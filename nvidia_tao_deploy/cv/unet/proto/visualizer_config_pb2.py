# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/unet/proto/visualizer_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nvidia_tao_deploy/cv/unet/proto/visualizer_config.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n7nvidia_tao_deploy/cv/unet/proto/visualizer_config.proto\"f\n\x10VisualizerConfig\x12\x0f\n\x07\x65nabled\x18\x01 \x01(\x08\x12\x1a\n\x12save_summary_steps\x18\x02 \x01(\r\x12%\n\x1dinfrequent_save_summary_steps\x18\x03 \x01(\rb\x06proto3')
)




_VISUALIZERCONFIG = _descriptor.Descriptor(
  name='VisualizerConfig',
  full_name='VisualizerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='enabled', full_name='VisualizerConfig.enabled', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='save_summary_steps', full_name='VisualizerConfig.save_summary_steps', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='infrequent_save_summary_steps', full_name='VisualizerConfig.infrequent_save_summary_steps', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=59,
  serialized_end=161,
)

DESCRIPTOR.message_types_by_name['VisualizerConfig'] = _VISUALIZERCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

VisualizerConfig = _reflection.GeneratedProtocolMessageType('VisualizerConfig', (_message.Message,), dict(
  DESCRIPTOR = _VISUALIZERCONFIG,
  __module__ = 'nvidia_tao_deploy.cv.unet.proto.visualizer_config_pb2'
  # @@protoc_insertion_point(class_scope:VisualizerConfig)
  ))
_sym_db.RegisterMessage(VisualizerConfig)


# @@protoc_insertion_point(module_scope)