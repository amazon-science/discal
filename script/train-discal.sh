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

main=/fsx/users/hwanjuns/codes/Distilation/DialogLM/EMNLP-DisCal/DisCal/run_summarization_w_calib.py
cuda=0,1,2,3,4,5,6,7

teacher_model=/fsx/shared/publication/discal/CNNDM-Results/Teacher
model=/fsx/shared/publication/discal/CNNDM-Results/BART-12-3/plate
max_source_length=1024
max_target_length=142
min_target_length=55
generation_num_beams=4
generation_length_penalty=2.0
train_file="/fsx/shared/publication/discal/CNNDM-Data/train.json"
validation_file=/fsx/shared/publication/discal/CNNDM-Data/validation.json
test_file=/fsx/shared/publication/discal/CNNDM-Data/test.json
out_dir=/fsx/users/hwanjuns/tensorboard/DisCal/BART-12-3

# training params
learning_rate=5e-5
max_steps=5000
warm_steps=250
log_steps=1000

# teacher for generating summaries
dynamic_temperature=True
temperature_scaling=2.0

# calibration params for CNNDM
calibration=True
num_candidate_beams=6
mle_weight=0.01 
calibration_weight=1.0
abstract_weight=0.2


CUDA_VISIBLE_DEVICES=${cuda} \
python -m torch.distributed.launch --nproc_per_node 8 ${main} \
    --model_name_or_path ${model} \
    --teacher_model_name_or_path ${teacher_model} \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --dynamic_temperature $dynamic_temperature \
    --temperature_scaling $temperature_scaling \
    --num_candidate_beams $num_candidate_beams \
    --abstract_weight $abstract_weight \
    --mle_weight $mle_weight \
    --calibration_weight $calibration_weight \
    --calibration $calibration \
    --train_file $train_file \
    --validation_file $validation_file \
    --test_file $test_file \
    --text_column article \
    --summary_column highlights \
    --output_dir ${out_dir} \
    --label_smoothing_factor 0.1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_source_length $max_source_length  \
    --max_target_length $max_target_length \
    --generation_num_beams $generation_num_beams \
    --generation_min_length $min_target_length \
    --generation_max_length $max_target_length \
    --generation_length_penalty $generation_length_penalty \
    --learning_rate $learning_rate \
    --lr_scheduler_type cosine \
    --weight_decay 0.0001 \
    --warmup_steps $warm_steps \
    --max_steps $max_steps \
    --save_strategy steps \
    --evaluation_strategy steps \
    --logging_steps $log_steps \
    --save_steps $log_steps \
    --eval_steps $log_steps \
    --fp16 \
    --predict_with_generate \
    --overwrite_output_dir



