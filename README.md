# Enhancing Abstractiveness of Summarization Models through Calibrated Distillation (EMNLP-Findings 2023)
by Hwanjun Song, Igor Shalyminov, Hang Su, Siffi Singh, Kaisheng Yao, Saab Mansour

Paper: https://www.amazon.science/publications/enhancing-abstractiveness-of-summarization-models-through-calibrated-distillation

## Highlights
Sequence-level knowledge distillation reduces the size of Seq2Seq models for more efficient abstractive summarization. However, it often leads to a loss of abstractiveness in summarization. In this paper, we propose a novel approach named DisCal to enhance the level of abstractiveness (measured by $n$-gram overlap) without sacrificing the informativeness (measured by ROUGE) of generated summaries. DisCal exposes diverse pseudo summaries with two supervision to the student model. Firstly, the best pseudo summary is identified in terms of abstractiveness and informativeness and used for sequence-level distillation. Secondly, their ranks are used to ensure the student model to assign higher prediction scores to summaries with higher ranks. Our experiments show that DisCal outperforms prior methods in abstractive summarization distillation, producing highly abstractive and informative summaries.

## Training
Training with Discal needs step-by-step procedures. First, a teacher model should be trained to generate pseudo summaries using diverse beam search. Second, the teacher model can be fine-tuned using sequence-level knowledge distillation. Third, we re-train the student model using DisCal. We here provide the example script for the three procedures based on BART 12-12 (teacher) and BART 12-3 (student).

### Training a Teacher Model
<details>
<summary>Run this command: <code>sh script/train-teacher.sh</code></summary>
<pre><code>
main=run_summarization.py
cuda=0,1,2,3,4,5,6,7

min_target_length=55
max_target_length=142
max_source_length=1024
length_penalty=2.0
num_beams=4
lr=5e-5

model_name_or_path=facebook/bart-large
train_file=[path-to-train-data]
validation_file=[path-to-val-data]
test_file=[path-to-test-data]
out_dir=[path-to-teacher-output]

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
</code></pre>
</details>


### Training a Student Model
<details>
<summary>Run this command: <code>sh script/train-student.sh</code></summary>
<pre><code>
main=run_summarization.py
cuda=0,1,2,3,4,5,6,7

min_target_length=55
max_target_length=142
max_source_length=1024
length_penalty=2.0
num_beams=4
lr=5e-5

model_name_or_path=[path-to-teacher].
out_dir=[path-to-student-output]
train_file=[path-to-train-data]
validation_file=[path-to-val-data]
test_file=[path-to-test-data]
shrink_type=12-3

CUDA_VISIBLE_DEVICES=${cuda}  \
python -m torch.distributed.launch --nproc_per_node 8 ${main} \
    --model_name_or_path $model_name_or_path \
    --output_dir ${out_dir} \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --shrink_type $shrink_type \
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
</code></pre>
</details>

### Re-training with DisCal

<details>
<summary>Run this command: <code>sh script/train-discal.sh</code></summary>
<pre><code>
main=run_summarization_w_calib.py
cuda=0,1,2,3,4,5,6,7

teacher_model=[path-to-teacher]
model=[path-to-student]
out_dir=[path-to-discal-output]
train_file=[path-to-train-data]
validation_file=[path-to-val-data]
test_file=[path-to-test-data]

max_source_length=1024
max_target_length=142
min_target_length=55
generation_num_beams=4
generation_length_penalty=2.0

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
calibration_momentum=1.0
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
    --calibration_momentum $calibration_momentum \
    --monotonic_weighting $monotonic_weighting \
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




</code></pre>
</details>

## Evaluation
<details>
<summary>Run this command: <code>sh script/eval.sh</code></summary>
<pre><code>
main=run_summarization.py
cuda=0,1,2,3,4,5,6,7

model_name_or_path=[model-checkpoint]
out_dir=[path-to-output]
train_file=[path-to-train-data]
validation_file=[path-to-val-data]
test_file=[path-to-test-data]

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
</code></pre>
</details>

## Citation

Please consider citation if our paper is useful in your research.

```BibTeX
@Inproceedings{Song2023,
 author = {Hwanjun Song and Igor Shalyminov and Hang Su and Siffi Singh and Kaisheng Yao and Saab Mansour},
 title = {Enhancing abstractiveness of summarization models through calibrated distillation},
 year = {2023},
 url = {https://www.amazon.science/publications/enhancing-abstractiveness-of-summarization-models-through-calibrated-distillation},
 booktitle = {EMNLP-Findings 2023},
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

