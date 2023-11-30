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

from Bart import BartConfig, BartForConditionalGeneration, BartModel, BartEncoder, BartDecoder


def get_bart_model(model_args, data_args):
    
    # Model
    base_config = BartConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # for the evaluation using checkpoint model -> attention temperatured pseudo label
    base_config.temperature_scaling = model_args.temperature_scaling

    model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=base_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.shrink_type is None:

        if hasattr(model.config, 'task_specific_params') and 'summarization' in model.config.task_specific_params:
            model.config.task_specific_params['summarization']['max_length'] = data_args.max_target_length
            model.config.task_specific_params['summarization']['min_length'] = data_args.min_target_length
            model.config.task_specific_params['summarization']['num_beams'] = data_args.num_beams
            model.config.task_specific_params['summarization']['length_penalty'] = data_args.length_penalty
            model.config.task_specific_params['summarization']['no_repeat_ngram_size'] = data_args.no_repeat_ngram_size
        model.config.max_length = data_args.max_target_length
        model.config.min_length = data_args.min_target_length
        model.config.num_beams = data_args.num_beams
        model.config.length_penalty = data_args.length_penalty
        model.config.no_repeat_ngram_size = data_args.no_repeat_ngram_size

        return model
    
    else:
        base_config = base_config.to_dict()

        if '12-12' in model_args.shrink_type:
            enc_step, dec_step = 1, 1
        elif '12-6' in model_args.shrink_type:
            enc_step, dec_step = 1, 2
        elif '12-3' in model_args.shrink_type:
            enc_step, dec_step = 1, 4
        elif '12-1' in model_args.shrink_type:
            enc_step, dec_step = 1, 12
        elif '6-6' in model_args.shrink_type:
            enc_step, dec_step = 2, 2
        elif '6-3' in model_args.shrink_type:
            enc_step, dec_step = 2, 4
        elif '6-1' in model_args.shrink_type:
            enc_step, dec_step = 2, 12

        base_config['encoder_layers'] //= enc_step
        base_config['decoder_layers'] //= dec_step
        base_config['max_length'] = data_args.max_target_length
        base_config['min_length'] = data_args.min_target_length
        base_config['num_beams'] = data_args.num_beams
        base_config['length_penalty'] = data_args.length_penalty
        base_config['no_repeat_ngram_size'] = data_args.no_repeat_ngram_size
        base_config['task_specific_params']['summarization']['max_length'] = data_args.max_target_length
        base_config['task_specific_params']['summarization']['min_length'] = data_args.min_target_length
        base_config['task_specific_params']['summarization']['num_beams'] = data_args.num_beams
        base_config['task_specific_params']['summarization']['length_penalty'] = data_args.length_penalty
        base_config['task_specific_params']['summarization']['no_repeat_ngram_size'] = data_args.no_repeat_ngram_size

        target_config = BartConfig.from_dict(base_config)
        shrink_model = BartForConditionalGeneration(target_config)

        def _copy_weight_like_distilbart(base, target):
            if isinstance(base, BartForConditionalGeneration) or isinstance(base, BartModel):
                for _base, _target in zip(base.children(), target.children()):
                    _copy_weight_like_distilbart(_base, _target)
            elif isinstance(base, BartEncoder):
                #print('alternating copy:', type(target).__name__)
                target.embed_tokens.load_state_dict(base.embed_tokens.state_dict())
                target.embed_positions.load_state_dict(base.embed_positions.state_dict())
                
                base_layers = [layer for layer in base.layers]
                target_layers = [layer for layer in target.layers]
                for i in range(len(target_layers)):
                    target_layers[i].load_state_dict(base_layers[enc_step * i].state_dict())
                    
            elif isinstance(base, BartDecoder):
                #print('alternating copy:', type(target).__name__)
                target.embed_tokens.load_state_dict(base.embed_tokens.state_dict())
                target.embed_positions.load_state_dict(base.embed_positions.state_dict())
                
                base_layers = [layer for layer in base.layers]
                target_layers = [layer for layer in target.layers]
                for i in range(len(target_layers)):
                    target_layers[i].load_state_dict(base_layers[dec_step * i].state_dict())
            else:
                #print('full copy:', type(target).__name__)
                target.load_state_dict(base.state_dict())

        _copy_weight_like_distilbart(model, shrink_model)

        return shrink_model
