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
Scoring with NMT
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import argparse
import tensorflow as tf

from models import utils, modeling


def parse_args():
  parser = argparse.ArgumentParser(description="Scoring with NMT")

  ## Required parameters
  parser.add_argument("--nmt_config_file", type=str, required=True)
  parser.add_argument("--source_input_file", type=str, required=True)
  parser.add_argument("--target_input_file", type=str, required=True)
  parser.add_argument("--score_output_file", type=str, required=True)
  parser.add_argument("--vocab_file", type=str, required=True)
  parser.add_argument("--init_checkpoint", type=str, required=True)

  ## Other parameters
  parser.add_argument("--max_seq_length", default=256, type=int)
  parser.add_argument("--decode_alpha", default=1.0, type=float)
  parser.add_argument("--decode_length", default=20, type=int)
  parser.add_argument("--decode_batch_size", default=32, type=int)

  ## TPU parameters
  parser.add_argument("--use_tpu", default=False, type=bool)
  parser.add_argument("--tpu_name", default=None, type=str)

  return parser.parse_args()


def get_evaluation_input(source_inputs,
                         target_inputs,
                         vocabulary,
                         max_seq_length,
                         decode_length,
                         decode_batch_size=32,
                         num_cpu_threads=2,
                         eos="<EOS>",
                         unkId=1):
  source_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(source_inputs))
  target_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(target_inputs))

  dataset = tf.data.Dataset.zip((source_dataset, target_dataset))

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
          tf.shape(src),
          tgt,
          tf.shape(tgt),
      ),
      num_parallel_calls=num_cpu_threads
  )

  dataset = dataset.padded_batch(
      batch_size=decode_batch_size,
      padded_shapes=(max_seq_length, 1, max_seq_length, 1),
      # padded_shapes=(tf.Dimension(None), 1),
      padding_values=(eos, 0, eos, 0))

  dataset = dataset.map(
      lambda src, src_len, tgt, tgt_len: {
          "source": src,
          "source_length": tf.squeeze(src_len, 1),
          "target": tgt,
          "target_length": tf.squeeze(tgt_len, 1),
      },
      num_parallel_calls=num_cpu_threads
  )

  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()

  src_table = tf.contrib.lookup.index_table_from_tensor(
      tf.constant(vocabulary),
      default_value=unkId
  )
  features["source"] = src_table.lookup(features["source"])
  features["target"] = src_table.lookup(features["target"])

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


def main(args):
  args.init_checkpoint = args.init_checkpoint.split(',')
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
    # Read input file
    source_inputs = []
    with tf.gfile.Open(args.source_input_file) as fd:
      for line in fd:
        source_inputs.append(line.strip())
    target_inputs = []
    with tf.gfile.Open(args.target_input_file) as fd:
      for line in fd:
        target_inputs.append(line.strip())

    true_length = len(source_inputs)

    while len(source_inputs) % args.decode_batch_size != 0:
      source_inputs.append('<UNK>')
      target_inputs.append('<UNK>')

    # Build input queue
    with tf.device('/CPU:0'):
      features = get_evaluation_input(source_inputs=source_inputs,
                                      target_inputs=target_inputs,
                                      vocabulary=vocabulary,
                                      max_seq_length=args.max_seq_length,
                                      decode_length=args.decode_length,
                                      decode_batch_size=args.decode_batch_size,
                                      eos=nmt_config.eos.encode(),
                                      unkId=nmt_config.unkId)

    # Create placeholders
    placeholders = {
        "source": tf.placeholder(tf.int32, [args.decode_batch_size, args.max_seq_length], "source_0"),
        "source_length": tf.placeholder(tf.int32, [args.decode_batch_size], "source_length_0"),
        "target": tf.placeholder(tf.int32, [args.decode_batch_size, args.max_seq_length], "target_0"),
        "target_length": tf.placeholder(tf.int32, [args.decode_batch_size], "target_length_0"),
    }

    model = [modeling.NmtModel(config=nmt_config) for _ in range(len(args.init_checkpoint))]
    for i, m in enumerate(model):
      m._scope = m._scope + str(i)

    model_fns = [m.get_evaluation_func() for m in model]

    if args.use_tpu:
      def computation(source, source_length, target, target_length):
        placeholders = {
            "source": source,
            "source_length": source_length,
            "target": target,
            "target_length": target_length,
        }
        scores = [model_fn(placeholders) for model_fn in model_fns]
        scores = tf.add_n(scores) / float(len(scores))
        return scores

      ops = tf.compat.v1.tpu.batch_parallel(computation,
                                            [placeholders["source"],
                                             placeholders["source_length"],
                                             placeholders["target"],
                                             placeholders["target_length"]],
                                            num_shards=8)
    else:
      scores = [model_fn(placeholders) for model_fn in model_fns]
      scores = tf.add_n(scores) / float(len(scores))
      ops = scores

    tvars = tf.trainable_variables()
    name_to_variable = {}
    for var in tvars:
      name = var.name
      m = re.match("^(.*):\\d+$", name)
      if m is not None:
        name = m.group(1)
      name_to_variable[name] = var

    for i, ckpt in enumerate(args.init_checkpoint):
      init_vars = tf.train.list_variables(ckpt)
      assignment_map = {}
      for x in init_vars:
        (name, var) = (x[0], x[1])
        m = re.match("^Tra\w*mer", name)
        if m is not None:
          temp = m.group(0)
          rename = temp + str(i) + name[len(temp):]
        else:
          rename = name
        if rename in name_to_variable:
          assignment_map[name] = rename
      tf.train.init_from_checkpoint(ckpt, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    total_size = 0
    for var in tvars:
      tf.logging.info("  name = %s, shape = %s", var.name, var.shape)
      total_size += reduce(lambda x, y: x * y, var.get_shape().as_list())
    tf.logging.info("  total variable parameters: %d", total_size)

    results = []

    target = ''
    config = None
    if args.use_tpu:
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu_name)
      target = tpu_cluster_resolver.get_master()
    else:
      config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(target=target, config=config) as sess:
      if args.use_tpu:
        sess.run(tf.contrib.tpu.initialize_system())

      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      while True:
        try:
          feats = sess.run(features)
          feed_dict = {}
          for name in feats:
            feed_dict[placeholders[name]] = feats[name]
          results.append(sess.run(ops, feed_dict=feed_dict))
          tf.logging.log(tf.logging.INFO, "Finished batch %d" % len(results))
        except tf.errors.OutOfRangeError:
          break

      if args.use_tpu:
        sess.run(tf.contrib.tpu.shutdown_system())

    target_dir, _ = os.path.split(args.score_output_file)
    tf.gfile.MakeDirs(target_dir)

    outputs = []

    for result in results:
      for item in result[0]:
        outputs.append(item)

    with tf.gfile.Open(args.score_output_file, "w") as outfile:
      for output in outputs[:true_length]:
        outfile.write("%f\n" % output)


if __name__ == "__main__":
  main(parse_args())