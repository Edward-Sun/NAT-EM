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

SRC_PREFIX=$0
TGT_PREFIX=$1

# for WMT, there are 46 splits of data
for NUM in {00..45}
do

src=${SRC_PREFIX}${NUM}
tgt=${TGT_PREFIX}${NUM}

gsutil cat ${src} \
  | awk '{while(++i<=20)print;i=0}' \
  | gsutil cp - ${tgt}

gsutil cat ${src} | wc -l
gsutil cat ${tgt} | wc -l

done