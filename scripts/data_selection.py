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
Select Data for Distillation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf


def parse_args():
  parser = argparse.ArgumentParser(description="Shuffle Dataset")

  parser.add_argument("--split", type=int, required=True, help="number of splits")
  parser.add_argument("--N", type=int, default=20, help="beam search size")
  parser.add_argument("--fsrc_path", type=str, required=True, help="Path of source repeated N times")
  parser.add_argument("--ftgt_path", type=str, required=True, help="Path of beam-search predictions")
  parser.add_argument("--fscore_path", type=str, required=True, help="Path of scores of predicted targets")
  parser.add_argument("--src_output", type=str, required=True, help="Path of selected sources")
  parser.add_argument("--tgt_output", type=str, required=True, help="Path of selected targets")

  return parser.parse_args()


def main(args):
  for split in range(0, args.split):
    suffix = "%02d" % split
    fsrc_path_suf = args.fsrc_path + suffix
    ftgt_path_suf = args.ftgt_path + suffix
    fscore_path_suf = args.fscore_path + suffix
    src_output_suf = args.src_output + suffix
    tgt_output_suf = args.tgt_output + suffix

    all_scores = []
    src_tgt = []
    with tf.gfile.Open(fsrc_path_suf) as fsrc, tf.gfile.Open(ftgt_path_suf) as ftgt, \
        tf.gfile.Open(fscore_path_suf) as fscore:
      with tf.gfile.Open(src_output_suf, 'w') as fsrc_out, tf.gfile.Open(tgt_output_suf, 'w') as ftgt_out:
        for src, tgt, line in zip(fsrc, ftgt, fscore):
          score = float(line.strip())
          src_tgt.append((src, tgt))
          all_scores.append(score)

          if len(all_scores) == args.N:
            index = 0
            for i in range(args.N):
              if all_scores[i] > all_scores[index]:
                index = i
            fsrc_out.write(src_tgt[index][0])
            ftgt_out.write(src_tgt[index][1])
            all_scores = []
            src_tgt = []

if __name__ == "__main__":
    main(parse_args())