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

dataset=iwslt14
src=de
tgt=en
tag=10k

model=transformer_small
AT_CKPT=60000
NAT_CKPT=200000
ALPHA=1.4
BATCH_SIZE=128
DECODE_LENGTH=0

for BEAM_SIZE in 1 9; do
  full_model_name=${model}_nat

  SOURCE=${BUCKT_PATH}/${dataset}/${TASK}.${src}.${tag}.bpe
  TEACHER_DIR=${BUCKT_PATH}/${dataset}/${model}
  NMT_DIR=${BUCKT_PATH}/${dataset}/${full_model_name}
  CHECKPOINT=${TEACHER_DIR}/model.ckpt-${AT_CKPT},${NMT_DIR}/model.ckpt-${NAT_CKPT}
  CONFIG=${NMT_DIR}/model_config.json
  VOCAB=${BUCKT_PATH}/${dataset}/train.mix.${tag}.vocab

  TARGET=${BUCKT_PATH}/${dataset}/decodes/${full_model_name}_c${NAT_CKPT}_a${ALPHA}_b${BEAM_SIZE}.${tgt}.txt

  python translate_nat.py \
    --source_input_file=${SOURCE} \
    --max_seq_length=175 \
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
done