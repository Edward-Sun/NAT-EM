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

TPU_NAME=$0
BUCKT_PATH=$1

dataset=iwslt14
src=de
tgt=en
tag=10k

model=transformer_small

SOURCE=${BUCKT_PATH}/${dataset}/train.${src}.${tag}.bpe.shuf
TARGET=${BUCKT_PATH}/${dataset}/train.${tgt}.${tag}.bpe.shuf
NMT_DIR=${BUCKT_PATH}/${dataset}/${model}
VOCAB=${BUCKT_PATH}/${dataset}/train.mix.${tag}.vocab
CONFIG=${BUCKT_PATH}/${model}.json

MAX_LENGTH=175
BATCH_SIZE=35000
WARM_UP=4000
LEARNING_RATE=0.1

python train_at.py \
  --source_input_file=${SOURCE} \
  --target_input_file=${TARGET} \
  --vocab_file=${VOCAB} \
  --nmt_config_file=${CONFIG} \
  --max_seq_length=${MAX_LENGTH} \
  --learning_rate=${LEARNING_RATE} \
  --output_dir=${NMT_DIR} \
  --train_batch_size=${BATCH_SIZE} \
  --num_warmup_steps=${WARM_UP} \
  --num_train_steps=120000 \
  --save_checkpoints_steps=20000 \
  --use_tpu=True \
  --tpu_name=${TPU_NAME}