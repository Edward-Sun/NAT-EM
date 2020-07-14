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
Translate with NAT
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import argparse
import numpy as np
import tensorflow as tf

from models import utils, modeling, nat_modeling


def parse_args():
  parser = argparse.ArgumentParser(description="Translate with NAT")

  # Required parameters
  parser.add_argument("--nmt_config_file", type=str, required=True)
  parser.add_argument("--source_input_file", type=str, required=True)
  parser.add_argument("--target_output_file", type=str, required=True)
  parser.add_argument("--vocab_file", type=str, required=True)
  parser.add_argument("--init_checkpoint", type=str, required=True)
  parser.add_argument("--teacher_config_file", default=None, type=str)

  # Other parameters
  parser.add_argument("--max_seq_length", default=256, type=int)
  parser.add_argument("--decode_alpha", default=1.0, type=float)
  parser.add_argument("--decode_length", default=0, type=int)
  parser.add_argument("--beam_size", default=4, type=int)
  parser.add_argument("--decode_batch_size", default=32, type=int)
  parser.add_argument("--output_bpe", default=False, type=bool)
  parser.add_argument("--use_length_predictor", default=True, type=bool)
  parser.add_argument("--ODD", default=True, type=bool)
  parser.add_argument("--dup_penalty", default=1e9, type=float)

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
                        decode_batch_size=32,
                        num_cpu_threads=2,
                        eos="<EOS>",
                        unkId=1,
                        use_tpu=False):
  dataset = tf.data.Dataset.from_tensor_slices(tf.constant(inputs))

  dataset = dataset.map(
      lambda x: tf.string_split([x]).values[:max_seq_length - 1],
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


def prepare_target_input(target_shape, target_length, nmt_config):
  full_eos = tf.fill(target_shape, nmt_config.eosId)
  full_pad = tf.fill(target_shape, nmt_config.padId)

  target_eos = tf.one_hot(target_length, depth=target_shape[1])
  target_eos = tf.equal(target_eos, 1)

  output = tf.where(target_eos, full_eos, full_pad)
  return output


def repeat(tensor, shape, duplicates):
  shape = list(shape)
  tensor_shape = utils.get_shape_list(tensor)
  repeat_shape = shape[:1] + [duplicates] + shape[1:]
  expected_shape = [a * b for a, b in zip(tensor_shape, [shape[0] * duplicates] + shape[1:])]
  return tf.reshape(tf.tile(tf.expand_dims(tensor, 1), repeat_shape), expected_shape)


def create_inference_graph(nat_model,
                           nmt_config,
                           features,
                           decode_length,
                           use_length_predictor,
                           beam_size):
  source_shape = utils.get_shape_list(features["source"])

  batch_size, max_seq_length = source_shape
  features["source"] = repeat(features["source"], (1, 1), beam_size)
  features["source_length"] = repeat(features["source_length"], (1,), beam_size)

  state, length_logits = nat_model.get_evaluation_encoder_func()(features)

  if use_length_predictor:
    features["target_length"] = tf.argmax(length_logits, axis=-1, output_type=tf.dtypes.int32)
    features["target_length"] = tf.where(tf.math.greater(features["target_length"], 128),
                                         features["source_length"] + decode_length,
                                         features["target_length"])
  else:
    features["target_length"] = features["source_length"] + decode_length

  features["target_length"] = tf.maximum(features["target_length"] - beam_size // 2,
                                         1 + tf.ones_like(features["target_length"]))

  features["target_length"] = features["target_length"] + tf.tile(tf.range(beam_size), (batch_size,))
  features["target_length"] = tf.minimum(features["target_length"],
                                         tf.maximum(128, max_seq_length + decode_length))

  target_shape = batch_size, tf.math.reduce_max(features["target_length"]) + 1
  features["target"] = prepare_target_input(target_shape, features["target_length"], nmt_config)

  output_logits = nat_model.get_evaluation_decoder_func()(features, state)

  output_logits = tf.nn.log_softmax(output_logits)

  predict_values, predict_indices = tf.math.top_k(output_logits, k=3)

  return predict_values, predict_indices, features["target_length"]


def create_rescorer_graph(score_fn,
                          features,
                          beam_size,
                          decode_alpha):
  source_shape = utils.get_shape_list(features["source"])
  batch_size, _ = source_shape
  target_shape = utils.get_shape_list(features["target"])
  _, max_seq_length = target_shape

  if beam_size > 1:
    features["source"] = repeat(features["source"], (1, 1), beam_size)
    features["source_length"] = repeat(features["source_length"], (1,), beam_size)

    log_probs = score_fn(features)

    length_penalty = tf.pow((5.0 + tf.to_float(features["target_length"])) / 6.0, decode_alpha)

    scores = log_probs / length_penalty

    scores = tf.reshape(scores, [batch_size, beam_size])

    positions = tf.argmax(scores, axis=-1, output_type=tf.int32)

    flat_offsets = tf.range(0, batch_size, dtype=tf.int32) * beam_size
    positions = positions + flat_offsets

    prediction = tf.gather(features["target"], positions)

    best_score = tf.reduce_max(scores, axis=-1)

    prediction = tf.reshape(prediction, [batch_size, 1, max_seq_length])
    best_score = tf.reshape(best_score, [batch_size, 1])
  else:
    prediction = tf.reshape(features["target"], [batch_size, 1, max_seq_length])
    best_score = tf.zeros([batch_size, 1])

  return prediction, best_score


def viterbi_decode(score, transition_params):
  """Decode the highest scoring sequence of tags outside of TensorFlow.
  This should only be used at test time.
  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indicies.
  """
  trellis = np.zeros_like(score)
  backpointers = np.zeros_like(score, dtype=np.int32)
  trellis[0] = score[0]

  for t in range(1, score.shape[0]):
    v = np.expand_dims(trellis[t - 1], 1) + transition_params[t-1]
    trellis[t] = score[t] + np.max(v, 0)
    backpointers[t] = np.argmax(v, 0)

  viterbi = [[np.argmax(trellis[-1])]]
  for bp in reversed(backpointers[1:]):
    viterbi.append([bp[viterbi[-1]]])
  viterbi.reverse()

  return viterbi


def deduplicate(predict_values, predict_indices, target_length, eosId, max_seq_length, ODD, dup_penalty):
  output = []
  for i in range(target_length.shape[0]):
    single_predict_values = predict_values[i]
    single_predict_indices = predict_indices[i]
    seq_len = target_length[i]
    if target_length:
      tile_predict_indices1 = np.tile(single_predict_indices[:seq_len-1].reshape((-1, 3, 1)), (1, 1, 3)
                                                                         ).reshape((-1, 3, 3))
      tile_predict_indices2 = np.tile(single_predict_indices[1:seq_len].reshape((-1, 3, 1)), (1, 3, 1)
                                                                          ).reshape((-1, 3, 3))
      transition_matrix = -dup_penalty * np.equal(tile_predict_indices1, tile_predict_indices2).astype(np.float)
      viterbi_output = viterbi_decode(single_predict_values[:seq_len], transition_matrix)
      viterbi_output = np.array(viterbi_output, dtype=np.int32)
      real_output = np.take_along_axis(single_predict_indices[:seq_len], viterbi_output, axis=1).reshape(-1)
    else:
      real_output = single_predict_indices[:seq_len][0]
    padded_output = np.pad(real_output, (0, max_seq_length - real_output.shape[0]),
                           'constant', constant_values=(eosId,))
    output.append(padded_output)

  output = np.array(output, dtype=np.int32)

  return output, target_length


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  vocabulary = make_vocab(args.vocab_file)

  nmt_config = modeling.NmtConfig.from_json_file(args.nmt_config_file)

  teacher_config = modeling.NmtConfig.from_json_file(args.teacher_config_file or args.nmt_config_file)

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

    # Build input queue
    with tf.device('/CPU:0'):
      features = get_inference_input(inputs=sorted_inputs,
                                     vocabulary=vocabulary,
                                     max_seq_length=args.max_seq_length,
                                     decode_batch_size=args.decode_batch_size,
                                     eos=nmt_config.eos.encode(),
                                     unkId=nmt_config.unkId,
                                     use_tpu=args.use_tpu)

    # Create placeholders
    if args.use_tpu:
      placeholders = {
          "source": tf.placeholder(tf.int32, [args.decode_batch_size, args.max_seq_length], "source_0"),
          "source_length": tf.placeholder(tf.int32, [args.decode_batch_size], "source_length_0"),
      }

      rescorer_placeholders = {
          "source": tf.placeholder(tf.int32, [args.decode_batch_size, args.max_seq_length], "source_1"),
          "source_length": tf.placeholder(tf.int32, [args.decode_batch_size], "source_length_1"),
          "target": tf.placeholder(tf.int32, [args.beam_size * args.decode_batch_size, args.max_seq_length], "target_1"),
          "target_length": tf.placeholder(tf.int32, [args.beam_size * args.decode_batch_size], "target_length_1"),
      }
    else:
      placeholders = {
          "source": tf.placeholder(tf.int32, [args.decode_batch_size, None], "source_0"),
          "source_length": tf.placeholder(tf.int32, [args.decode_batch_size], "source_length_0"),
      }

      rescorer_placeholders = {
          "source": tf.placeholder(tf.int32, [args.decode_batch_size, None], "source_1"),
          "source_length": tf.placeholder(tf.int32, [args.decode_batch_size], "source_length_1"),
          "target": tf.placeholder(tf.int32, [args.beam_size * args.decode_batch_size, None],
                                   "target_1"),
          "target_length": tf.placeholder(tf.int32, [args.beam_size * args.decode_batch_size], "target_length_1"),
      }

    model = modeling.NmtModel(config=teacher_config)

    score_fn = model.get_evaluation_func()

    nat_model = nat_modeling.NatModel(config=nmt_config)

    if args.use_tpu:
      def computation(source, source_length):
        placeholders = {
            "source": source,
            "source_length": source_length,
        }

        prediction_op = create_inference_graph(nat_model=nat_model,
                                               nmt_config=nmt_config,
                                               features=placeholders,
                                               decode_length=args.decode_length,
                                               use_length_predictor=args.use_length_predictor,
                                               beam_size=args.beam_size)
        return prediction_op

      prediction_op = tf.compat.v1.tpu.batch_parallel(computation,
                                                      [placeholders["source"],
                                                       placeholders["source_length"]],
                                                      num_shards=8)

      def rescorer_computation(source, source_length, target, target_length):
        rescorer_placeholders = {
            "source": source,
            "source_length": source_length,
            "target": target,
            "target_length": target_length,
        }

        rescorer_op = create_rescorer_graph(score_fn=score_fn,
                                            features=rescorer_placeholders,
                                            beam_size=args.beam_size,
                                            decode_alpha=args.decode_alpha)
        return rescorer_op

      rescorer_op = tf.compat.v1.tpu.batch_parallel(rescorer_computation,
                                                    [rescorer_placeholders["source"],
                                                     rescorer_placeholders["source_length"],
                                                     rescorer_placeholders["target"],
                                                     rescorer_placeholders["target_length"]],
                                                    num_shards=8)

    else:
      prediction_op = create_inference_graph(nat_model=nat_model,
                                             nmt_config=nmt_config,
                                             features=placeholders,
                                             decode_length=args.decode_length,
                                             use_length_predictor=args.use_length_predictor,
                                             beam_size=args.beam_size)

      rescorer_op = create_rescorer_graph(score_fn=score_fn,
                                          features=rescorer_placeholders,
                                          beam_size=args.beam_size,
                                          decode_alpha=args.decode_alpha)

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

          predict_values, predict_indices, target_length = sess.run(prediction_op, feed_dict=feed_dict)

          target, target_length = deduplicate(predict_values, predict_indices, target_length, nmt_config.eosId,
                                              np.amax(target_length), args.ODD, args.dup_penalty)

          feed_dict = {}
          for name in feats:
            feed_dict[rescorer_placeholders[name]] = feats[name]
          feed_dict[rescorer_placeholders["target"]] = target
          feed_dict[rescorer_placeholders["target_length"]] = target_length

          results.append(sess.run(rescorer_op, feed_dict=feed_dict))
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
        for subitem in item.tolist():
          outputs.append(subitem)

    origin_outputs = []
    for index in range(len(sorted_keys)):
      origin_outputs.append(outputs[sorted_keys[index]])

    with tf.gfile.Open(args.target_output_file, "w") as outfile:
      for output in origin_outputs:
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