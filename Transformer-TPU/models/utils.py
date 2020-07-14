# coding=utf-8
# Copyright 2020 EM-NAT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Transformer Network
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import inplace_ops


def get_shape_list(tensor, name=None):
  """Adapted from BERT repo: https://github.com/google-research/bert"""
  if name is None:
    name = tensor.name

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def dropout(input_tensor, dropout_prob):
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor
  else:
    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def linear(input_data,
           output_size,
           bias=True,
           name=None):
  with tf.variable_scope(name, default_name="linear"):
    input_shape = get_shape_list(input_data)
    input_size = input_shape[-1]

    weight_initializer = tf.variance_scaling_initializer(
        1.0, mode="fan_avg", distribution="uniform")
    W = tf.get_variable("W",
                        shape=[input_size, output_size],
                        initializer=weight_initializer)
    output = tf.matmul(tf.reshape(input_data, [-1, input_size]), W)

    if bias:
      bias = tf.get_variable("b",
                             shape=[output_size],
                             initializer=tf.zeros_initializer)
      output = tf.nn.bias_add(output, bias)

    output_shape = input_shape[:-1] + [output_size]
    output = tf.reshape(output, output_shape)

    return output


def layer_norm(input_tensor, name=None):
  return tf.contrib.layers.layer_norm(inputs=input_tensor,
                                      begin_norm_axis=-1,
                                      begin_params_axis=-1,
                                      scope=name)


def post_process(previous_data,
                 input_data,
                 dropout_rate=None):
  input_data = dropout(input_data, dropout_rate)
  return layer_norm(previous_data + input_data)


def ffn_layer(inputs, hidden_size, output_size, dropout_rate=None, name=None):
  hidden = linear(inputs, hidden_size, name="input_layer")
  hidden = tf.nn.relu(hidden)
  hidden = dropout(hidden, dropout_rate)
  output = linear(hidden, output_size, name="output_layer")
  return output


def dot_product_attention(q,
                          k,
                          v,
                          bias=None,
                          attention_dropout_rate=None,
                          num_heads=None,
                          name=None):
  # split heads
  q_shape = get_shape_list(q)
  k_shape = get_shape_list(k)
  v_shape = get_shape_list(v)

  head_size = q_shape[-1] // num_heads
  value_size = v_shape[-1] // num_heads

  new_q_shape = q_shape[:-1] + [num_heads, head_size]
  new_k_shape = k_shape[:-1] + [num_heads, head_size]
  new_v_shape = v_shape[:-1] + [num_heads, value_size]

  q = tf.transpose(tf.reshape(q, new_q_shape), [0, 2, 1, 3])
  k = tf.transpose(tf.reshape(k, new_k_shape), [0, 2, 1, 3])
  v = tf.transpose(tf.reshape(v, new_v_shape), [0, 2, 1, 3])

  # [batch, num_heads, query_length, memory_length]
  logits = tf.matmul(q, k, transpose_b=True) * (head_size ** -0.5)
  if bias is not None:
    logits += bias
  weights = tf.nn.softmax(logits)
  weights = dropout(weights, attention_dropout_rate)
  x = tf.matmul(weights, v)

  # combine heads
  new_x_shape = q_shape[:-1] + [v_shape[-1]]
  x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), new_x_shape)
  return x


def attention_mask(mask, neg_inf=-1e9, name=None):
  with tf.name_scope(name, default_name="attention_mask"):
    ret = (1.0 - mask) * neg_inf
    return tf.expand_dims(tf.expand_dims(ret, 1), 1)


def causal_mask(length, neg_inf=-1e9, name=None):
  with tf.name_scope(name, default_name="causal_mask"):
    lower_triangle = tf.matrix_band_part(
        tf.ones([length, length]), -1, 0
    )
    ret = neg_inf * (1.0 - lower_triangle)
    return tf.reshape(ret, [1, 1, length, length])


def multihead_self_attention(queries,
                             bias,
                             num_heads,
                             key_size,
                             value_size,
                             output_size,
                             dropout_rate=None,
                             state=None,
                             decode_step=None):
  q = linear(queries, key_size, name="q_transform")
  k = linear(queries, key_size, name="k_transform")
  v = linear(queries, value_size, name="v_transform")

  if state is not None:
    # incrementally append current KV to previous KV
    tmp_k = tf.transpose(state["key"], perm=[1, 0, 2])
    tmp_k = inplace_ops.alias_inplace_update(
        tmp_k, decode_step, tf.squeeze(k, axis=1))
    k = tf.transpose(tmp_k, perm=[1, 0, 2])
    tmp_v = tf.transpose(state["value"], perm=[1, 0, 2])
    tmp_v = inplace_ops.alias_inplace_update(
        tmp_v, decode_step, tf.squeeze(v, axis=1))
    v = tf.transpose(tmp_v, perm=[1, 0, 2])

    next_state = {}
    next_state["key"] = k
    next_state["value"] = v

  results = dot_product_attention(q, k, v, bias, dropout_rate, num_heads)

  outputs = linear(results, output_size, name="output_transform")

  outputs = {"outputs": outputs}
  if state is not None:
    outputs["state"] = next_state

  return outputs


def multihead_encdec_attention(queries,
                               memories,
                               bias,
                               num_heads,
                               key_size,
                               value_size,
                               output_size,
                               dropout_rate=None):
  q = linear(queries, key_size, name="q_transform")
  k = linear(memories, key_size, name="k_transform")
  v = linear(memories, value_size, name="v_transform")

  results = dot_product_attention(q, k, v, bias, dropout_rate, num_heads)

  outputs = linear(results, output_size, name="output_transform")

  return outputs