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

import copy
import json

import six
import tensorflow as tf
from .utils import *


class NmtConfig(object):
  """Adapted from BERT repo: https://github.com/google-research/bert"""

  def __init__(self,
               vocab_size,
               pad="<PAD>",
               eos="<EOS>",
               unk="<UNK>",
               bos="<BOS>",
               padId=0,
               eosId=1,
               unkId=2,
               bosId=3,
               hidden_size=512,
               filter_size=2048,
               num_heads=8,
               num_encoder_layers=6,
               num_decoder_layers=6,
               attention_dropout=0.1,
               hidden_dropout=0.1,
               relu_dropout=0.1,
               label_smoothing=0.1,
               attention_key_channels=0,
               attention_value_channels=0,
               shared_embedding_and_softmax_weights=True,
               shared_source_target_embedding=True,
               use_one_hot_embeddings=True,
               max_position_embeddings=256):
    self.vocab_size = vocab_size
    self.pad = pad
    self.bos = bos
    self.unk = unk
    self.eos = eos
    self.padId = padId
    self.eosId = eosId
    self.unkId = unkId
    self.bosId = bosId
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.num_heads = num_heads
    self.num_encoder_layers = num_encoder_layers
    self.num_decoder_layers = num_decoder_layers
    self.attention_dropout = attention_dropout
    self.hidden_dropout = hidden_dropout
    self.relu_dropout = relu_dropout
    self.label_smoothing = label_smoothing
    self.attention_key_channels = attention_key_channels
    self.attention_value_channels = attention_value_channels
    self.shared_embedding_and_softmax_weights = shared_embedding_and_softmax_weights
    self.shared_source_target_embedding = shared_source_target_embedding
    self.max_position_embeddings = max_position_embeddings
    self.use_one_hot_embeddings = use_one_hot_embeddings

  @classmethod
  def from_dict(cls, json_object, vocab_size=None):
    """Constructs a `NmtConfig` from a Python dictionary of parameters."""
    config = NmtConfig(vocab_size=vocab_size)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file, vocab_size=None):
    """Constructs a `NmtConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text), vocab_size=vocab_size)

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

  def deep_copy(self):
    return self.from_dict(self.to_dict())


class NmtModel(object):
  """Nmt Model: Transformer"""

  def __init__(self, config, scope="Trasnformer"):
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

    source_length = features["source_length"]
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

    return encoder_output

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

    if "one_hot_target" in features:
      batch_size, max_target_length, _ = get_shape_list(features["one_hot_target"])
    else:
      batch_size, max_target_length = get_shape_list(features["target"])

    target_length = features["target_length"]
    target_mask = tf.sequence_mask(target_length,
                                   maxlen=max_target_length,
                                   dtype=tf.float32)

    if config.use_one_hot_embeddings:
      if "one_hot_target" in features:
        one_hot_target = tf.reshape(features["one_hot_target"], [-1, config.vocab_size])
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

    if mode != "infer":
      position_embeddings = position_embeddings[:, :max_target_length]
      decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
      decoder_input += position_embeddings
      decoder_input = dropout(decoder_input, config.hidden_dropout)
      decoder_state = None
      decode_step = None
      dec_attn_bias = causal_mask(max_target_length)
    else:
      decoder_state = state["decoder"]
      decode_step = tf.reduce_max(decoder_state["step"])
      decoder_input = tf.slice(position_embeddings,
                               (0, decode_step, 0),
                               (1, 1, hidden_size))
      decoder_input += targets
      dec_attn_bias = causal_mask(self._config.max_position_embeddings)
      dec_attn_bias = tf.slice(dec_attn_bias,
                               (0, 0, decode_step, 0),
                               (1, 1, 1, self._config.max_position_embeddings))

    encoder_output = state["encoder"]

    with tf.variable_scope("decoder"):
      x = layer_norm(decoder_input)
      next_state = {}
      for layer in range(config.num_decoder_layers):
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

    if mode != "infer":
      decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
      logits = tf.matmul(decoder_output, weights, False, True)
      logits = tf.nn.bias_add(logits, bias)

      if config.use_one_hot_embeddings:
        onehot_labels = one_hot_target
      else:
        labels = tf.reshape(features["target"], [-1])
        onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), depth=config.vocab_size)

      if mode == "reinforce":
        xentropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                   logits=logits,
                                                   reduction=tf.losses.Reduction.NONE)
        xentropy = tf.reshape(xentropy, get_shape_list(target_mask))
        xentropy = tf.reduce_sum(xentropy * target_mask, axis=1)
        xentropy = xentropy / tf.reduce_sum(target_mask, axis=1)

        return xentropy

      if mode == "eval":
        xentropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                   logits=logits,
                                                   reduction=tf.losses.Reduction.NONE)
        xentropy = tf.reshape(xentropy, get_shape_list(target_mask))
        return -tf.reduce_sum(xentropy * target_mask, axis=1)

      weights = tf.reshape(target_mask, [-1])
      xentropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                 logits=logits,
                                                 weights=weights,
                                                 label_smoothing=config.label_smoothing)

      n = tf.to_float(config.vocab_size - 1)
      p = 1.0 - config.label_smoothing
      q = config.label_smoothing / n
      normalizing = -(p * tf.log(p) + n * q * tf.log(q + 1e-20))

      loss = xentropy - normalizing

      return loss

    else:
      next_state["step"] = decoder_state["step"] + 1
      decoder_state = next_state
      decoder_output = decoder_output[:, -1, :]
      logits = tf.matmul(decoder_output, weights, False, True)
      logits = tf.nn.bias_add(logits, bias)
      log_prob = tf.nn.log_softmax(logits)

      return log_prob, {"encoder": encoder_output, "decoder": decoder_state}

  def get_training_func(self):
    def training_fn(features):
      with tf.variable_scope(self._scope):
        mode = "train"
        encoder_output = self.encoder_subgraph(features, mode)
        state = {
            "encoder": encoder_output
        }
        loss = self.decoder_subgraph(features, state, mode)
        return loss

    return training_fn

  def get_evaluation_func(self):
    def evaluation_fn(features):
      with tf.variable_scope(self._scope):
        mode = "eval"
        encoder_output = self.encoder_subgraph(features, mode)
        state = {
            "encoder": encoder_output
        }
        score = self.decoder_subgraph(features, state, mode)
        return score

    return evaluation_fn

  def get_inference_func(self):
    def encoding_fn(features, config=None):
      with tf.variable_scope(self._scope):
        encoder_output = self.encoder_subgraph(features, "infer")
        batch = tf.shape(encoder_output)[0]

        step_dict = {
            "step": tf.zeros([batch, self._config.max_position_embeddings, 1], dtype=tf.int32)
        }
        state = {
            "encoder": encoder_output,
            "decoder": {
                "layer_%d" % i: {
                    "key": tf.zeros([batch, self._config.max_position_embeddings,
                                     self._config.attention_key_channels or self._config.hidden_size]),
                    "value": tf.zeros([batch, self._config.max_position_embeddings,
                                       self._config.attention_value_channels or self._config.hidden_size])
                }
                for i in range(self._config.num_decoder_layers)
            }
        }
        state["decoder"].update(step_dict)

      return state

    def decoding_fn(features, state, config=None):
      with tf.variable_scope(self._scope):
        log_prob, new_state = self.decoder_subgraph(features, state, "infer")

      return log_prob, new_state

    return encoding_fn, decoding_fn