#!/bin/sh
output="$(locate tensorflow | grep include/google/protobuf/compiler/objectivec/objectivec_message.h)"
varname='--tf_xla_auto_jit=2 '
# eval ${varname} ${output%${output:(-64)}}

TF_XLA_FLAGS=${varname}${output%${output:(-64)}}


