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
Scoring with NAT
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import argparse
import tensorflow as tf

from models import utils, modeling, nat_modeling


def parse_args():
  parser = argparse.ArgumentParser(description="Scoring with NAT")

  ## Required parameters
  parser.add_argument("--nmt_config_file", type=str, required=True)
  parser.add_argument("--source_input_file", type=str, required=True)
  parser.add_argument("--target_input_file", type=str, required=True)
  parser.add_argument("--score_output_file", type=str, required=True)
  parser.add_argument("--vocab_file", type=str, required=True)
  parser.add_argument("--init_checkpoint", type=str, required=True)

  ## Other parameters
  parser.add_argument("--max_seq_length", default=256, type=int)
  parser.add_argument("--decode_batch_size", default=32, type=int)

  ## TPU parameters
  parser.add_argument("--use_tpu", default=False, type=bool)
  parser.add_argument("--tpu_name", default=None, type=str)

  return parser.parse_args()


def get_inference_input(source_input_file,
                        target_input_file,
                        vocabulary,
                        max_seq_length,
                        decode_batch_size,
                        num_cpu_threads=4,
                        eos="<EOS>",
                        unkId=1):
  src_dataset = tf.data.TextLineDataset(source_input_file)
  tgt_dataset = tf.data.TextLineDataset(target_input_file)

  dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

  dataset = dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values[:max_seq_length - 1],
          tf.string_split([tgt]).values[:max_seq_length - 1],
      ),
      num_parallel_calls=num_cpu_threads
  )

  dataset = dataset.map(
      lambda src, tgt: (
          tf.concat([src, [tf.constant(eos)]], axis=0),
          tf.concat([tgt, [tf.constant(eos)]], axis=0),
      ),
      num_parallel_calls=num_cpu_threads
  )

  dataset = dataset.map(
      lambda src, tgt: (
          src,
          tgt,
          tf.shape(src),
          tf.shape(tgt),
      ),
      num_parallel_calls=num_cpu_threads
  )

  dataset = dataset.padded_batch(
      batch_size=decode_batch_size,
      padded_shapes=(max_seq_length, max_seq_length, 1, 1),
      drop_remainder=False)

  dataset = dataset.map(
      lambda src, tgt, src_len, tgt_len: {
          "source": src,
          "target": tgt,
          "source_length": src_len,
          "target_length": tgt_len
      },
      num_parallel_calls=num_cpu_threads
  )

  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()

  shared_table = tf.contrib.lookup.index_table_from_tensor(
      tf.constant(vocabulary),
      default_value=unkId
  )

  features["source"] = shared_table.lookup(features["source"])
  features["target"] = shared_table.lookup(features["target"])

  features["source"] = tf.to_int32(features["source"])
  features["target"] = tf.to_int32(features["target"])
  features["source_length"] = tf.to_int32(features["source_length"])
  features["target_length"] = tf.to_int32(features["target_length"])
  features["source_length"] = tf.squeeze(features["source_length"], 1)
  features["target_length"] = tf.squeeze(features["target_length"], 1)

  batch_size = utils.get_shape_list(features["source"])[0]

  features["source"] = tf.pad(features["source"], ((0, decode_batch_size - batch_size), (0, 0)))
  features["target"] = tf.pad(features["target"], ((0, decode_batch_size - batch_size), (0, 0)))
  features["source_length"] = tf.pad(features["source_length"], ((0, decode_batch_size - batch_size),))
  features["target_length"] = tf.pad(features["target_length"], ((0, decode_batch_size - batch_size),))

  return features


def make_vocab(vocab_file):
  vocab = []

  vocab.append("<PAD>")
  vocab.append("<EOS>")
  vocab.append("<UNK>")
  vocab.append("<BOS>")

  with tf.gfile.Open(vocab_file, "r") as fin:
    lines = fin.readlines()

  for line in lines:
    word, freq = line.strip().split()
    if int(freq) >= 5:
      vocab.append(word)

  tf.logging.info("Vocabulary size: %d" % (len(vocab)))

  return vocab


def nat_output_mask(logits, neg_inf=-1e9, filter_num=4, name=None):
  with tf.name_scope(name, default_name="nat_output_mask"):
    batch_size, max_length, vocab_size = utils.get_shape_list(logits)

    mask_upper = tf.ones([1, max_length, filter_num]) * neg_inf
    mask_lower = tf.zeros([1, max_length, vocab_size - filter_num])

    mask = tf.concat([mask_upper, mask_lower], axis=-1)
    logits = logits + mask
    return logits


def add_eos_one_hot(logits, length, eosId, inf=1e9, name=None):
  with tf.name_scope(name, default_name="add_eos_one_hot"):
    batch_size, max_length, vocab_size = utils.get_shape_list(logits)
    one_hot_length = tf.one_hot(length, depth=max_length)
    one_hot_eos = tf.one_hot([eosId], depth=vocab_size)

    one_hot_length = tf.reshape(one_hot_length, [batch_size, max_length, 1])
    one_hot_eos = tf.reshape(one_hot_eos, [1, 1, vocab_size])

    mask = one_hot_length * one_hot_eos * (2 * inf)

    logits = logits + mask
    return logits


def prepare_target_input(target_shape, target_length, nmt_config):
  full_eos = tf.fill(target_shape, nmt_config.eosId)
  full_pad = tf.fill(target_shape, nmt_config.padId)

  target_eos = tf.one_hot(target_length, depth=target_shape[1])
  target_eos = tf.equal(target_eos, 1)

  output = tf.where(target_eos,
                    full_eos,
                    full_pad)
  return output


def create_scorer_graph(nat_model,
                        nmt_config,
                        features):
  original_target = features["target"]

  target_shape = utils.get_shape_list(features["source"])
  features["target"] = prepare_target_input(target_shape,
                                            features["target_length"],
                                            nmt_config)

  state, length_logits = nat_model.get_evaluation_encoder_func()(features)

  output_logits = nat_model.get_evaluation_decoder_func()(features, state)

  target_length_one_hot = tf.one_hot(features["target_length"],
                                     depth=nmt_config.max_position_embeddings)

  length_xentropy = tf.losses.softmax_cross_entropy(onehot_labels=target_length_one_hot,
                                                    logits=length_logits,
                                                    reduction=tf.losses.Reduction.NONE)

  batch_size, max_target_length = target_shape

  target_mask = tf.sequence_mask(features["target_length"],
                                 maxlen=max_target_length,
                                 dtype=tf.float32)

  flat_target = tf.reshape(original_target, [-1])
  one_hot_target = tf.one_hot(flat_target, depth=nmt_config.vocab_size)

  output_logits = tf.reshape(output_logits, [-1, nmt_config.vocab_size])

  xentropy = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_target,
                                             logits=output_logits,
                                             reduction=tf.losses.Reduction.NONE)

  xentropy = tf.reshape(xentropy, utils.get_shape_list(target_mask))
  xentropy = tf.reduce_sum(xentropy * target_mask, axis=1)

  score = -(length_xentropy + xentropy)

  return score


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  vocabulary = make_vocab(args.vocab_file)

  nmt_config = modeling.NmtConfig.from_json_file(args.nmt_config_file)

  tf.logging.info("Checkpoint Vocab Size: %d", nmt_config.vocab_size)
  tf.logging.info("True Vocab Size: %d", len(vocabulary))

  assert nmt_config.vocab_size == len(vocabulary)

  vocabulary[nmt_config.padId] = nmt_config.pad.encode()
  vocabulary[nmt_config.eosId] = nmt_config.eos.encode()
  vocabulary[nmt_config.unkId] = nmt_config.unk.encode()
  vocabulary[nmt_config.bosId] = nmt_config.bos.encode()

  # Build Graph
  with tf.Graph().as_default():
    # Build input queue
    with tf.device('/CPU:0'):
      features = get_inference_input(source_input_file=args.source_input_file,
                                     target_input_file=args.target_input_file,
                                     vocabulary=vocabulary,
                                     max_seq_length=args.max_seq_length,
                                     decode_batch_size=args.decode_batch_size,
                                     eos=nmt_config.eos.encode(),
                                     unkId=nmt_config.unkId)

    # Create placeholders
    scorer_placeholders = {
        "source": tf.placeholder(tf.int32, [args.decode_batch_size, args.max_seq_length], "source_0"),
        "source_length": tf.placeholder(tf.int32, [args.decode_batch_size], "source_length_0"),
        "target": tf.placeholder(tf.int32, [args.decode_batch_size, args.max_seq_length], "target_0"),
        "target_length": tf.placeholder(tf.int32, [args.decode_batch_size], "target_length_0"),
    }

    nat_model = nat_modeling.NatModel(config=nmt_config)

    if args.use_tpu:
      def scorer_computation(source, source_length, target, target_length):
        scorer_placeholders = {
            "source": source,
            "source_length": source_length,
            "target": target,
            "target_length": target_length,
        }

        scorer_op = create_scorer_graph(nat_model=nat_model,
                                        nmt_config=nmt_config,
                                        features=scorer_placeholders)
        return scorer_op

      scorer_op = tf.compat.v1.tpu.batch_parallel(scorer_computation,
                                                  [scorer_placeholders["source"],
                                                   scorer_placeholders["source_length"],
                                                   scorer_placeholders["target"],
                                                   scorer_placeholders["target_length"]],
                                                  num_shards=8)

    tvars = tf.trainable_variables()
    for init_checkpoint_single in args.init_checkpoint.split(","):
      init_vars = tf.train.list_variables(init_checkpoint_single)

      name_to_variable = {}
      assignment_map = {}

      for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
          name = m.group(1)
        name_to_variable[name] = var

      for x in init_vars:
        (name, var) = (x[0], x[1])
        if name in name_to_variable:
          assignment_map[name] = name
      tf.train.init_from_checkpoint(init_checkpoint_single, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    total_size = 0
    for var in tvars:
      tf.logging.info("  name = %s, shape = %s", var.name, var.shape)
      total_size += reduce(lambda x, y: x * y, var.get_shape().as_list())
    tf.logging.info("  total variable parameters: %d", total_size)

    target = ''
    config = None
    if args.use_tpu:
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu_name)
      target = tpu_cluster_resolver.get_master()
    else:
      config = tf.ConfigProto(allow_soft_placement=True)

    target_dir, _ = os.path.split(args.score_output_file)
    tf.gfile.MakeDirs(target_dir)

    with tf.Session(target=target, config=config) as sess:
      if args.use_tpu:
        sess.run(tf.contrib.tpu.initialize_system())

      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())

      with tf.gfile.Open(args.score_output_file, "w") as outfile:
        count = 0
        while True:
          try:
            feats = sess.run(features)
            feed_dict = {}
            for name in feats:
              feed_dict[scorer_placeholders[name]] = feats[name]
            scores = sess.run(scorer_op, feed_dict=feed_dict)[0]

            for score in scores:
              outfile.write("%f\n" % score)

            count += 1
            tf.logging.log(tf.logging.INFO, "Finished batch %d" % count)
          except tf.errors.OutOfRangeError:
            break

      if args.use_tpu:
        sess.run(tf.contrib.tpu.shutdown_system())


if __name__ == "__main__":
  main(parse_args())