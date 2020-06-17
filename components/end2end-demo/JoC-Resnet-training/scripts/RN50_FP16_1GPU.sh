# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

# This script launches ResNet50 training in FP16 on 1 GPUs using 256 batch size (256 per GPU)
# Usage ./RN50_FP16_1GPU.sh <path to this repository> <path to dataset> <path to results directory>

python main.py --num_iter=180 --iter_unit=epoch --data_dir=$1 --batch_size=4096 --use_tf_amp --use_auto_loss_scaling --results_dir=$2
# python main.py --num_iter=90 --iter_unit=epoch --data_dir=$1 --batch_size=1024 --use_tf_amp --use_xla --results_dir=$2
