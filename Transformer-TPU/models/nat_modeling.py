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
from .utils import *


class NatModel(object):
  """Nat Model: Trasnformer_NAT"""

  def __init__(self, config, scope="Trasnformer_NAT"):
    self._scope = scope
    self._config = config

  def encoder_subgraph(self, features, mode):
    config = self._config.deep_copy()

    if mode != "train":
      config.hidden_dropout = 0.0
      config.attention_dropout = 0.0
      config.relu_dropout = 0.0
      config.label_smoothing = 0.0

    hidden_size = config.hidden_size
    initializer = tf.random_normal_initializer(0.0, hidden_size ** -0.5)

    source_embedding = tf.get_variable("shared_embedding",
                                       [config.vocab_size, hidden_size],
                                       initializer=initializer)

    if "one_hot_source" in features:
      batch_size, max_source_length, _ = get_shape_list(features["one_hot_source"])
    else:
      batch_size, max_source_length = get_shape_list(features["source"])

    source_length = features["source_length"] + 1
    source_mask = tf.sequence_mask(source_length,
                                   maxlen=max_source_length,
                                   dtype=tf.float32)

    if config.use_one_hot_embeddings:
      if "one_hot_source" in features:
        one_hot_source = features["one_hot_source"]
      else:
        flat_source = tf.reshape(features["source"], [-1])
        one_hot_source = tf.one_hot(flat_source, depth=config.vocab_size)
      inputs = tf.matmul(one_hot_source, source_embedding) * (hidden_size ** 0.5)
      inputs = tf.reshape(inputs, [batch_size, max_source_length, hidden_size])
    else:
      inputs = tf.gather(source_embedding, features["source"]) * (hidden_size ** 0.5)

    full_position_embeddings = tf.get_variable(
        "position_embeddings",
        [config.max_position_embeddings, hidden_size],
        initializer=initializer)
    position_embeddings = full_position_embeddings[:max_source_length] * (hidden_size ** 0.5)
    position_embeddings = tf.expand_dims(position_embeddings, axis=0)

    inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    inputs += position_embeddings

    bias = tf.get_variable("source_language_bias",
                           [hidden_size],
                           initializer=tf.zeros_initializer)
    encoder_input = tf.nn.bias_add(inputs, bias)

    attention_bias = attention_mask(source_mask)

    with tf.variable_scope("encoder"):
      x = encoder_input
      x = dropout(x, config.hidden_dropout)
      x = layer_norm(x)
      for layer in range(config.num_encoder_layers):
        with tf.variable_scope("layer_%d" % layer):
          with tf.variable_scope("self_attention"):
            y = multihead_self_attention(
                x,
                attention_bias,
                config.num_heads,
                config.attention_key_channels or config.hidden_size,
                config.attention_value_channels or config.hidden_size,
                config.hidden_size,
                config.attention_dropout
            )
            y = y["outputs"]
            x = post_process(x, y, config.hidden_dropout)

          with tf.variable_scope("feed_forward"):
            y = ffn_layer(
                x,
                config.filter_size,
                config.hidden_size,
                config.relu_dropout
            )
            x = post_process(x, y, config.hidden_dropout)

      encoder_output = x

    weights = tf.get_variable(
        "position_prediction_weights",
        [config.max_position_embeddings, hidden_size],
        initializer=initializer)
    length_logits = tf.matmul(encoder_output[:, 0], weights, False, True)
    length_logits = tf.reshape(length_logits, [batch_size, -1])

    return encoder_output, length_logits

  def decoder_subgraph(self, features, state, mode):
    config = self._config.deep_copy()

    if mode != "train":
      config.hidden_dropout = 0.0
      config.attention_dropout = 0.0
      config.relu_dropout = 0.0
      config.label_smoothing = 0.0

    source_length = features["source_length"]
    max_source_length = get_shape_list(features["source"])[1]

    source_mask = tf.sequence_mask(source_length,
                                   maxlen=max_source_length,
                                   dtype=tf.float32)

    hidden_size = config.hidden_size
    initializer = tf.random_normal_initializer(0.0, hidden_size ** -0.5)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      target_embedding = tf.get_variable(
          "shared_embedding",
          [config.vocab_size, hidden_size],
          initializer=initializer)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      weights = tf.get_variable("shared_embedding",
                                [config.vocab_size, hidden_size],
                                initializer=initializer)

    bias = tf.get_variable("output_bias",
                           [config.vocab_size],
                           initializer=tf.zeros_initializer)

    if "one_hot_source" in features:
      batch_size, max_target_length, _ = get_shape_list(features["one_hot_source"])
    else:
      batch_size, max_target_length = get_shape_list(features["source"])

    target_length = features["target_length"]
    target_mask = tf.sequence_mask(target_length,
                                   maxlen=max_target_length,
                                   dtype=tf.float32)

    if config.use_one_hot_embeddings:
      if "one_hot_target" in features:
        one_hot_target = features["one_hot_target"]
      else:
        flat_target = tf.reshape(features["target"], [-1])
        one_hot_target = tf.one_hot(flat_target, depth=config.vocab_size)
      targets = tf.matmul(one_hot_target, target_embedding) * (hidden_size ** 0.5)
      targets = tf.reshape(targets, [batch_size, max_target_length, hidden_size])
    else:
      targets = tf.gather(target_embedding, features["target"]) * (hidden_size ** 0.5)

    enc_attn_bias = attention_mask(source_mask)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      full_position_embeddings = tf.get_variable(
          "position_embeddings",
          [config.max_position_embeddings, hidden_size],
          initializer=initializer)
    position_embeddings = full_position_embeddings * (hidden_size ** 0.5)
    position_embeddings = tf.expand_dims(position_embeddings, axis=0)

    position_embeddings = position_embeddings[:, :max_target_length]
    decoder_input = targets
    decoder_input += position_embeddings
    decoder_input = dropout(decoder_input, config.hidden_dropout)
    decoder_state = None
    decode_step = None
    dec_attn_bias = attention_mask(target_mask)

    encoder_output = state["encoder"]

    with tf.variable_scope("decoder"):
      x = layer_norm(decoder_input)
      next_state = {}
      for layer in range(config.num_decoder_layers + 1):
        layer_name = "layer_%d" % layer
        with tf.variable_scope(layer_name):
          layer_state = decoder_state[layer_name] if decoder_state is not None else None

          with tf.variable_scope("self_attention"):
            y = multihead_self_attention(
                x,
                dec_attn_bias,
                config.num_heads,
                config.attention_key_channels or config.hidden_size,
                config.attention_value_channels or config.hidden_size,
                config.hidden_size,
                config.attention_dropout,
                state=layer_state,
                decode_step=decode_step
            )

            if layer_state is not None:
              next_state[layer_name] = y["state"]

            y = y["outputs"]
            x = post_process(x, y, config.hidden_dropout)

          with tf.variable_scope("encdec_attention"):
            y = multihead_encdec_attention(
                x,
                encoder_output,
                enc_attn_bias,
                config.num_heads,
                config.attention_key_channels or config.hidden_size,
                config.attention_value_channels or config.hidden_size,
                config.hidden_size,
                config.attention_dropout
            )
            x = post_process(x, y, config.hidden_dropout)

          with tf.variable_scope("feed_forward"):
            y = ffn_layer(
                x,
                config.filter_size,
                config.hidden_size,
                config.relu_dropout,
            )
            x = post_process(x, y, config.hidden_dropout)

      decoder_output = x

    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, weights, False, True)
    logits = tf.nn.bias_add(logits, bias)

    logits = tf.reshape(logits, [batch_size, max_target_length, -1])

    return logits

  def get_training_encoder_func(self):
    def training_encoder_fn(features):
      with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
        mode = "train"
        encoder_output, length_logits = self.encoder_subgraph(features, mode)
        state = {
            "encoder": encoder_output
        }
        return state, length_logits

    return training_encoder_fn

  def get_training_decoder_func(self):
    def training_decoder_fn(features, state):
      with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
        mode = "train"
        logits = self.decoder_subgraph(features, state, mode)
        return logits

    return training_decoder_fn

  def get_evaluation_encoder_func(self):
    def evaluation_encoder_fn(features):
      with tf.variable_scope(self._scope):
        mode = "eval"
        encoder_output, length_logits = self.encoder_subgraph(features, mode)
        state = {
            "encoder": encoder_output
        }
        return state, length_logits

    return evaluation_encoder_fn

  def get_evaluation_decoder_func(self):
    def evaluation_decoder_fn(features, state):
      with tf.variable_scope(self._scope):
        mode = "eval"
        logits = self.decoder_subgraph(features, state, mode)
        return logits

    return evaluation_decoder_fn
