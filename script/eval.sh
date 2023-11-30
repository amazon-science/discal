#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

#!/bin/bash -x
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1

main=/fsx/users/hwanjuns/codes/Distilation/DialogLM/EMNLP-DisCal/DisCal/run_summarization.py
cuda=0,1,2,3,4,5,6,7

model_name_or_path=//fsx/shared/publication/discal/CNNDM-Results/BART-12-3/DisCal/checkpoint-5000
out_dir=/fsx/users/hwanjuns/tensorboard/DisCal/eval/output
train_file="/fsx/shared/publication/discal/CNNDM-Data/train.json"
validation_file=/fsx/shared/publication/discal/CNNDM-Data/validation.json
test_file=/fsx/shared/publication/discal/CNNDM-Data/test.json

min_target_length=55
max_target_length=142
max_source_length=1024
length_penalty=2.0
num_beams=4

CUDA_VISIBLE_DEVICES=${cuda}  \
python -m torch.distributed.launch --nproc_per_node 8 ${main} \
    --model_name_or_path $model_name_or_path \
    --output_dir ${out_dir} \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --train_file $train_file \
    --validation_file $validation_file \
    --test_file $test_file \
    --label_smoothing_factor 0.1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_source_length $max_source_length \
    --min_target_length $min_target_length \
    --max_target_length $max_target_length \
    --length_penalty $length_penalty \
    --num_beams $num_beams \
    --save_strategy steps \
    --evaluation_strategy steps \
    --fp16 \
    --overwrite_output_dir \
    --predict_with_generate

