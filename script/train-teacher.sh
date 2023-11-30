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
out_dir=/fsx/users/hwanjuns/tensorboard/DisCal/BART-12-12/teacher
cuda=0,1,2,3,4,5,6,7

min_target_length=55
max_target_length=142
max_source_length=1024
length_penalty=2.0
num_beams=4
lr=5e-5

model_name_or_path=facebook/bart-large
train_file=/fsx/users/hwanjuns/datasets/CNNDM/train.json
validation_file=/fsx/users/hwanjuns/datasets/CNNDM/validation.json
test_file=/fsx/users/hwanjuns/datasets/CNNDM/test.json


CUDA_VISIBLE_DEVICES=${cuda}  \
python -m torch.distributed.launch --nproc_per_node 8 ${main} \
    --model_name_or_path $model_name_or_path \
    --output_dir ${out_dir} \
    --do_train True \
    --do_eval True \
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
    --learning_rate $lr \
    --lr_scheduler_type cosine \
    --weight_decay 0.0001 \
    --warmup_steps 500 \
    --max_steps 20000 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --logging_steps 4000 \
    --save_steps 4000 \
    --eval_steps 4000 \
    --fp16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --report_to='tensorboard' 

