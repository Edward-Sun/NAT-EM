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

tag=32k

data_path=data/wmt14_en_de

train_file_output=${data_path}/train.mix.${tag}.bpe
vocab_file=${data_path}/train.mix.${tag}.vocab

train_de_output=${data_path}/train.de.${tag}.bpe
train_en_output=${data_path}/train.en.${tag}.bpe

cat ${train_de_output} ${train_en_output} > ${train_file_output}

subword-nmt get-vocab < ${train_file_output} > ${vocab_file}

python3 shuffle_dataset.py --input ${train_de_output} ${train_en_output}