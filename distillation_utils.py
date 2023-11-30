"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import Any, Optional, Union
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import numpy as np
from Bart import BartConfig, BartForConditionalGeneration


def get_teacher_model(model_args, training_args):
    
    # Model
    base_config = BartConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # for the evaluation using checkpoint model -> attention temperatured pseudo label
    base_config.temperature_scaling = training_args.temperature_scaling
    base_config.dynamic_temperature = training_args.dynamic_temperature

    model = BartForConditionalGeneration.from_pretrained(
    model_args.teacher_model_name_or_path,
    from_tf=bool(".ckpt" in model_args.teacher_model_name_or_path),
    config=base_config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
    )
    model.config.task_specific_params['summarization']['max_length'] = training_args.generation_max_length
    model.config.task_specific_params['summarization']['min_length'] = training_args.generation_min_length
    model.config.task_specific_params['summarization']['num_beams'] = training_args.generation_num_beams
    model.config.task_specific_params['summarization']['length_penalty'] = training_args.generation_length_penalty
    model.config.task_specific_params['summarization']['no_repeat_ngram_size'] = training_args.generation_no_repeat_ngram_size
    model.config.max_length = training_args.generation_max_length
    model.config.min_length = training_args.generation_min_length
    model.config.num_beams = training_args.generation_num_beams
    model.config.length_penalty = training_args.generation_length_penalty
    model.config.no_repeat_ngram_size = training_args.generation_no_repeat_ngram_size

    return model
        

def get_student_model(model_args, training_args):
    
    # Model
    base_config = BartConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # for the evaluation using checkpoint model -> attention temperatured pseudo label
    base_config.dynamic_temperature = False
    base_config.temperature_scaling = 1.0

    if training_args.shrink_type is None: # Need to erase shrink tpye to load dual head models.

        model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=base_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        )
        model.config.task_specific_params['summarization']['max_length'] = training_args.generation_max_length
        model.config.task_specific_params['summarization']['min_length'] = training_args.generation_min_length
        model.config.task_specific_params['summarization']['num_beams'] = training_args.generation_num_beams
        model.config.task_specific_params['summarization']['length_penalty'] = training_args.generation_length_penalty
        model.config.task_specific_params['summarization']['no_repeat_ngram_size'] = training_args.generation_no_repeat_ngram_size
        model.config.max_length = training_args.generation_max_length
        model.config.min_length = training_args.generation_min_length
        model.config.num_beams = training_args.generation_num_beams
        model.config.length_penalty = training_args.generation_length_penalty
        model.config.no_repeat_ngram_size = training_args.generation_no_repeat_ngram_size

        return model
    
    else:
        
        # single head teacher, and copy to dual head for initialization
        model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=base_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        )

        base_config = base_config.to_dict()

        if '12-12' in training_args.shrink_type:
            enc_step, dec_step = 1, 1
        elif '12-6' in training_args.shrink_type:
            enc_step, dec_step = 1, 2
        elif '12-3' in training_args.shrink_type:
            enc_step, dec_step = 1, 4
        elif '12-1' in training_args.shrink_type:
            enc_step, dec_step = 1, 12
        elif '6-6' in training_args.shrink_type:
            enc_step, dec_step = 2, 2
        elif '6-3' in training_args.shrink_type:
            enc_step, dec_step = 2, 4
        elif '6-1' in training_args.shrink_type:
            enc_step, dec_step = 2, 12

        base_config['encoder_layers'] //= enc_step
        base_config['decoder_layers'] //= dec_step
        base_config['max_length'] = training_args.generation_max_length
        base_config['min_length'] = training_args.generation_min_length
        base_config['num_beams'] = training_args.generation_num_beams
        base_config['length_penalty'] = training_args.generation_length_penalty
        base_config['no_repeat_ngram_size'] = training_args.generation_no_repeat_ngram_size
        base_config['task_specific_params']['summarization']['max_length'] = training_args.generation_max_length
        base_config['task_specific_params']['summarization']['min_length'] = training_args.generation_min_length
        base_config['task_specific_params']['summarization']['num_beams'] = training_args.generation_num_beams
        base_config['task_specific_params']['summarization']['length_penalty'] = training_args.generation_length_penalty
        base_config['task_specific_params']['summarization']['no_repeat_ngram_size'] = training_args.generation_no_repeat_ngram_size

        target_config = BartConfig.from_dict(base_config)
        shrink_model = BartForConditionalGeneration(target_config)

        def _copy_weight_shared_layer(base, target):
            target.load_state_dict(base.state_dict())

        def _copy_weight_encoding_layer(base, target):
            target.embed_tokens.load_state_dict(base.embed_tokens.state_dict())
            target.embed_positions.load_state_dict(base.embed_positions.state_dict())
            
            base_layers = [layer for layer in base.layers]
            target_layers = [layer for layer in target.layers]
            for i in range(len(target_layers)):
                target_layers[i].load_state_dict(base_layers[enc_step * i].state_dict())

        def _copy_weight_decoding_layer(base, target):
            target.embed_tokens.load_state_dict(base.embed_tokens.state_dict())
            target.embed_positions.load_state_dict(base.embed_positions.state_dict())
            
            base_layers = [layer for layer in base.layers]
            target_layers = [layer for layer in target.layers]
            for i in range(len(target_layers)):
                target_layers[i].load_state_dict(base_layers[dec_step * i].state_dict())

        def _copy_weight_head_layer(base, target):
            target.load_state_dict(base.state_dict())

        _copy_weight_shared_layer(model.model.shared, shrink_model.model.shared)
        _copy_weight_encoding_layer(model.model.encoder, shrink_model.model.encoder)
        _copy_weight_decoding_layer(model.model.decoder, shrink_model.model.decoder)
        _copy_weight_head_layer(model.lm_head, shrink_model.lm_head)

        return shrink_model