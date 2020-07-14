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
"""NMT"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import argparse
import tensorflow as tf

from models import modeling
import optimization


def parse_args():
  parser = argparse.ArgumentParser(description="Train NMT")

  # Required parameters
  parser.add_argument("--nmt_config_file", type=str, required=True)
  parser.add_argument("--source_input_file", type=str, required=True)
  parser.add_argument("--target_input_file", type=str, required=True)
  parser.add_argument("--vocab_file", type=str, required=True)
  parser.add_argument("--output_dir", type=str, required=True)

  # Other parameters
  parser.add_argument("--init_checkpoint", default=None, type=str)
  parser.add_argument("--max_seq_length", default=256, type=int)
  parser.add_argument("--train_batch_size", default=32, type=int)
  parser.add_argument("--learning_rate", default=5e-5, type=float)
  parser.add_argument("--num_train_steps", default=100000, type=int)
  parser.add_argument("--num_warmup_steps", default=10000, type=int)
  parser.add_argument("--save_checkpoints_steps", default=1000, type=int)
  parser.add_argument("--update_freq", default=1, type=int)

  # TPU parameters
  parser.add_argument("--use_tpu", default=False, type=bool)
  parser.add_argument("--tpu_name", default=None, type=str)

  return parser.parse_args()


def get_model_fn(nmt_config, learning_rate, num_train_steps,
                 num_warmup_steps, use_tpu, update_freq):
  def model_fn(features, labels, mode, params):
    model = modeling.NmtModel(config=nmt_config)

    loss = model.get_training_func()(features)

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    tf.logging.info("**** Trainable Variables ****")
    tvars = tf.trainable_variables()
    total_size = 0
    for var in tvars:
      tf.logging.info("  name = %s, shape = %s", var.name, var.shape)
      total_size += reduce(lambda x, y: x * y, var.get_shape().as_list())
    tf.logging.info("  total variable parameters: %d", total_size)

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
          update_freq=update_freq)

      estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op)

      return estimator_spec
    else:
      raise ValueError("Only TRAIN modes are supported")

  return model_fn


def get_input_fn(source_input_file,
                 target_input_file,
                 vocabulary,
                 max_seq_length,
                 num_cpu_threads=4,
                 eos="<EOS>",
                 unkId=1):
  def input_fn(params):
    batch_size = params["batch_size"] // max_seq_length

    src_dataset = tf.data.TextLineDataset(source_input_file)
    tgt_dataset = tf.data.TextLineDataset(target_input_file)

    dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat()

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
        batch_size=batch_size,
        padded_shapes=(max_seq_length, max_seq_length, 1, 1),
        drop_remainder=True)

    dataset = dataset.map(
        lambda src, tgt, src_len, tgt_len: {
            "source": src,
            "target": tgt,
            "source_length": src_len,
            "target_length": tgt_len
        },
        num_parallel_calls=num_cpu_threads
    )

    src_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(vocabulary),
        default_value=unkId
    )
    tgt_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(vocabulary),
        default_value=unkId
    )

    dataset = dataset.map(
        lambda feature: _decode_record(feature, src_table, tgt_table),
        num_parallel_calls=num_cpu_threads
    )

    return dataset

  return input_fn


def _decode_record(feature, src_table, tgt_table):
  feature["source"] = src_table.lookup(feature["source"])
  feature["target"] = tgt_table.lookup(feature["target"])

  feature["source"] = tf.to_int32(feature["source"])
  feature["target"] = tf.to_int32(feature["target"])
  feature["source_length"] = tf.to_int32(feature["source_length"])
  feature["target_length"] = tf.to_int32(feature["target_length"])
  feature["source_length"] = tf.squeeze(feature["source_length"], 1)
  feature["target_length"] = tf.squeeze(feature["target_length"], 1)

  return feature


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

  nmt_config = modeling.NmtConfig.from_json_file(args.nmt_config_file, vocab_size=len(vocabulary))

  vocabulary[nmt_config.padId] = nmt_config.pad.encode()
  vocabulary[nmt_config.eosId] = nmt_config.eos.encode()
  vocabulary[nmt_config.unkId] = nmt_config.unk.encode()
  vocabulary[nmt_config.bosId] = nmt_config.bos.encode()

  tf.gfile.MakeDirs(args.output_dir)

  with tf.gfile.Open(os.path.join(args.output_dir, 'model_config.json'), "w") as fout:
    fout.write(nmt_config.to_json_string())

  tpu_cluster_resolver = None
  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  tpu_config = tf.contrib.tpu.TPUConfig(iterations_per_loop=1000,
                                        num_shards=8,
                                        per_host_input_for_training=is_per_host)
  if args.use_tpu and args.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu_name)

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=args.output_dir,
      save_checkpoints_steps=args.save_checkpoints_steps,
      keep_checkpoint_max=0,
      tpu_config=tpu_config)

  model_fn = get_model_fn(
      nmt_config=nmt_config,
      learning_rate=args.learning_rate,
      num_train_steps=args.num_train_steps,
      num_warmup_steps=args.num_warmup_steps,
      use_tpu=args.use_tpu,
      update_freq=args.update_freq)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=args.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=args.train_batch_size,
      warm_start_from=args.init_checkpoint)

  tf.logging.info("***** Running training *****")
  tf.logging.info("  Batch size = %d", args.train_batch_size)
  train_input_fn = get_input_fn(
      source_input_file=args.source_input_file,
      target_input_file=args.target_input_file,
      vocabulary=vocabulary,
      max_seq_length=args.max_seq_length,
      eos=nmt_config.eos.encode(),
      unkId=nmt_config.unkId)
  estimator.train(input_fn=train_input_fn, max_steps=args.num_train_steps)


if __name__ == "__main__":
  main(parse_args())