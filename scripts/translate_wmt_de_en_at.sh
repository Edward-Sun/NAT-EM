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

TASK="test"

TPU_NAME=$0
BUCKT_PATH=$1

dataset=wmt14_en_de
src=de
tgt=en
tag=32k

model=transformer_base_rev

CKPT=500000
ALPHA=1.4
BEAM_SIZE=5
BATCH_SIZE=128
DECODE_LENGTH=20

SOURCE=${BUCKT_PATH}/${dataset}/${TASK}.${src}.${tag}.bpe
NMT_DIR=${BUCKT_PATH}/${dataset}/${model}
CHECKPOINT=${NMT_DIR}/model.ckpt-${CKPT}
CONFIG=${NMT_DIR}/model_config.json
VOCAB=${BUCKT_PATH}/${dataset}/train.mix.${tag}.vocab

TARGET=${BUCKT_PATH}/${dataset}/decodes/${model}_c${CKPT}_a${ALPHA}_b${BEAM_SIZE}.${tgt}.txt

python translate_at.py \
  --source_input_file=${SOURCE} \
  --max_seq_length=250 \
  --target_output_file=${TARGET} \
  --vocab_file=${VOCAB} \
  --nmt_config_file=${CONFIG} \
  --decode_batch_size=${BATCH_SIZE} \
  --decode_alpha=${ALPHA} \
  --decode_length=${DECODE_LENGTH} \
  --init_checkpoint=${CHECKPOINT} \
  --beam_size=${BEAM_SIZE} \
  --use_tpu=True \
  --tpu_name=${TPU_NAME}