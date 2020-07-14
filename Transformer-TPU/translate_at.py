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
Translation with NMT
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import argparse
import tensorflow as tf

from models import beamsearch, modeling


def parse_args():
  parser = argparse.ArgumentParser(description="Translate with NMT")

  # Required parameters
  parser.add_argument("--nmt_config_file", type=str, required=True)
  parser.add_argument("--source_input_file", type=str, required=True)
  parser.add_argument("--target_output_file", type=str, required=True)
  parser.add_argument("--vocab_file", type=str, required=True)
  parser.add_argument("--init_checkpoint", type=str, required=True)

  # Other parameters
  parser.add_argument("--max_seq_length", default=256, type=int)
  parser.add_argument("--decode_alpha", default=1.0, type=float)
  parser.add_argument("--decode_length", default=20, type=int)
  parser.add_argument("--beam_size", default=4, type=int)
  parser.add_argument("--decode_batch_size", default=32, type=int)
  parser.add_argument("--output_bpe", default=False, type=bool)
  parser.add_argument("--top_beams", default=1, type=int)

  # TPU parameters
  parser.add_argument("--use_tpu", default=False, type=bool)
  parser.add_argument("--tpu_name", default=None, type=str)

  return parser.parse_args()


def sort_input_file(filename):
  inputs = []
  with tf.gfile.Open(filename) as fd:
    for line in fd:
      inputs.append(line.decode('utf-8').strip())

  input_lens = []
  for i, line in enumerate(inputs):
    input_lens.append((i, len(line.split())))

  sorted_input_lens = sorted(input_lens, key=lambda x: -x[1])
  sorted_keys = {}
  sorted_inputs = []

  for i, tup in enumerate(sorted_input_lens):
    index, _ = tup
    sorted_inputs.append(inputs[index])
    sorted_keys[index] = i

  return sorted_keys, sorted_inputs


def get_inference_input(inputs,
                        vocabulary,
                        max_seq_length,
                        decode_length,
                        decode_batch_size=32,
                        num_cpu_threads=2,
                        eos="<EOS>",
                        unkId=1,
                        use_tpu=False):
  dataset = tf.data.Dataset.from_tensor_slices(tf.constant(inputs))

  dataset = dataset.map(
      lambda x: tf.string_split([x]).values[:max_seq_length - decode_length - 1],
      num_parallel_calls=num_cpu_threads
  )

  dataset = dataset.map(
      lambda x: tf.concat([x, [tf.constant(eos)]], axis=0),
      num_parallel_calls=num_cpu_threads
  )

  dataset = dataset.map(
      lambda x: (
          x,
          tf.shape(x),
      ),
      num_parallel_calls=num_cpu_threads
  )

  if use_tpu:
    dataset = dataset.padded_batch(
        batch_size=decode_batch_size,
        padded_shapes=(max_seq_length, 1),
        padding_values=(eos, 0))
  else:
    dataset = dataset.padded_batch(
        batch_size=decode_batch_size,
        padded_shapes=(tf.Dimension(None), 1),
        padding_values=(eos, 0))

  dataset = dataset.map(
      lambda src, src_len: {
          "source": src,
          "source_length": tf.squeeze(src_len, 1),
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
    sorted_keys, sorted_inputs = sort_input_file(args.source_input_file)
    while len(sorted_inputs) % args.decode_batch_size != 0:
      sorted_inputs.append(nmt_config.pad)

    tf.logging.info("Total Sentence Size: %d", len(sorted_keys))

    # Build input queue
    with tf.device('/CPU:0'):
      features = get_inference_input(inputs=sorted_inputs,
                                     vocabulary=vocabulary,
                                     max_seq_length=args.max_seq_length,
                                     decode_length=args.decode_length,
                                     decode_batch_size=args.decode_batch_size,
                                     eos=nmt_config.eos.encode(),
                                     unkId=nmt_config.unkId,
                                     use_tpu=args.use_tpu)

    # Create placeholders
    if args.use_tpu:
      placeholders = {
          "source": tf.placeholder(tf.int32, [args.decode_batch_size, args.max_seq_length], "source_0"),
          "source_length": tf.placeholder(tf.int32, [args.decode_batch_size], "source_length_0")
      }
    else:
      placeholders = {
          "source": tf.placeholder(tf.int32, [args.decode_batch_size, None], "source_0"),
          "source_length": tf.placeholder(tf.int32, [args.decode_batch_size], "source_length_0")
      }

    model = modeling.NmtModel(config=nmt_config)

    model_fn = model.get_inference_func()

    if args.use_tpu:
      def computation(source, source_length):
        placeholders = {
            "source": source,
            "source_length": source_length,
        }
        predictions = beamsearch.create_inference_graph(model_fns=model_fn,
                                                        features=placeholders,
                                                        decode_length=args.decode_length,
                                                        beam_size=args.beam_size,
                                                        top_beams=args.top_beams,
                                                        decode_alpha=args.decode_alpha,
                                                        bosId=nmt_config.bosId,
                                                        eosId=nmt_config.eosId)
        return predictions[0], predictions[1]

      ops = tf.compat.v1.tpu.batch_parallel(computation,
                                            [placeholders["source"],
                                             placeholders["source_length"]],
                                            num_shards=8)
    else:
      predictions = beamsearch.create_inference_graph(model_fns=model_fn,
                                                      features=placeholders,
                                                      decode_length=args.decode_length,
                                                      beam_size=args.beam_size,
                                                      top_beams=args.top_beams,
                                                      decode_alpha=args.decode_alpha,
                                                      bosId=nmt_config.bosId,
                                                      eosId=nmt_config.eosId)
      ops = (predictions[0], predictions[1])

    init_vars = tf.train.list_variables(args.init_checkpoint)
    tvars = tf.trainable_variables()

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
    tf.train.init_from_checkpoint(args.init_checkpoint, assignment_map)

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

    target_dir, _ = os.path.split(args.target_output_file)
    tf.gfile.MakeDirs(target_dir)

    outputs = []

    for result in results:
      for item in result[0]:
        tmp = []
        for subitem in item.tolist():
          tmp.append(subitem)
        outputs.append(tmp)

    origin_outputs = []
    for index in range(len(sorted_keys)):
      origin_outputs.append(outputs[sorted_keys[index]])

    with tf.gfile.Open(args.target_output_file, "w") as outfile:
      for beam_group in origin_outputs:
        for output in beam_group:
          decoded = []
          for idx in output:
            symbol = vocabulary[idx]
            if symbol == nmt_config.eos.encode():
              break
            decoded.append(symbol)

          decoded = str.join(" ", decoded)

          if not args.output_bpe:
            decoded = decoded.replace("@@ ", "")

          outfile.write("%s\n" % decoded)


if __name__ == "__main__":
  main(parse_args())