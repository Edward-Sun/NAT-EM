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

num_operations=10000

tag=10k

data_path=data/iwslt14

train_de=${data_path}/train.de
train_en=${data_path}/train.en
valid_de=${data_path}/valid.de
valid_en=${data_path}/valid.en
test_de=${data_path}/test.de
test_en=${data_path}/test.en
train_file=${data_path}/train.mix
train_file_output=${data_path}/train.mix.${tag}.bpe
codes_file=${data_path}/train.mix.${tag}.code
vocab_file=${data_path}/train.mix.${tag}.vocab

train_de_output=${data_path}/train.de.${tag}.bpe
train_en_output=${data_path}/train.en.${tag}.bpe
valid_de_output=${data_path}/valid.de.${tag}.bpe
valid_en_output=${data_path}/valid.en.${tag}.bpe
test_de_output=${data_path}/test.de.${tag}.bpe
test_en_output=${data_path}/test.en.${tag}.bpe

cat ${train_de} ${train_en} > ${train_file}

subword-nmt learn-bpe -s ${num_operations} -o ${codes_file} < ${train_file}

subword-nmt apply-bpe -c ${codes_file} < ${train_file} > ${train_file_output}

subword-nmt get-vocab < ${train_file_output} > ${vocab_file}

subword-nmt apply-bpe --vocabulary ${vocab_file} --vocabulary-threshold 5 -c ${codes_file} < ${train_de} > ${train_de_output}
subword-nmt apply-bpe --vocabulary ${vocab_file} --vocabulary-threshold 5 -c ${codes_file} < ${train_en} > ${train_en_output}

subword-nmt apply-bpe --vocabulary ${vocab_file} --vocabulary-threshold 5 -c ${codes_file} < ${valid_de} > ${valid_de_output}
subword-nmt apply-bpe --vocabulary ${vocab_file} --vocabulary-threshold 5 -c ${codes_file} < ${valid_en} > ${valid_en_output}

subword-nmt apply-bpe --vocabulary ${vocab_file} --vocabulary-threshold 5 -c ${codes_file} < ${test_de} > ${test_de_output}
subword-nmt apply-bpe --vocabulary ${vocab_file} --vocabulary-threshold 5 -c ${codes_file} < ${test_en} > ${test_en_output}

python3 shuffle_dataset.py --input ${train_de_output} ${train_en_output}