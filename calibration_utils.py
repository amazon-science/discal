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

from transformers import Seq2SeqTrainer
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_apex_available
if is_apex_available():
    from apex import amp
from compare_mt.rouge.rouge_scorer import RougeScorer
from transformers.deepspeed import is_deepspeed_zero3_enabled
from copy import deepcopy
import numpy as np
import nltk


all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)


def compute_rouge(gt_summary, pred_summary):
    score = all_scorer.score(gt_summary, pred_summary)
    return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3.0


def compute_abstractiveness(src, pred_summary):
    novel_1 = novel_ngram_overlap(src, pred_summary, 1)
    novel_3 = novel_ngram_overlap(src, pred_summary, 3)
    novel_5 = novel_ngram_overlap(src, pred_summary, 5)

    # if 3 / 5 grams are not available.
    if novel_3 == -1:
        return novel_1 / 3.0
    if novel_5 == -1:
        return (novel_1 + novel_3) / 3.0

    return (novel_1 + novel_3 + novel_5) / 3.0


def novel_ngram_overlap(src, pred, num_gram):
    src = src.strip()
    pred = pred.strip()

    ngram_set = set()
    src_ngrams = nltk.ngrams(src.split(), num_gram)
    pred_ngrams = nltk.ngrams(pred.split(), num_gram)

    for ngram in src_ngrams:
        ngram_set.add(ngram)
    
    total_ngram = 0
    num_overlap = 0
    for ngram in pred_ngrams:
        if ngram in ngram_set:
            num_overlap += 1
        total_ngram += 1

    if total_ngram == 0:
        return -1 # error code.

    novel_ngram_ratio =  (1.0 - num_overlap / float(total_ngram)) 
    return novel_ngram_ratio


def compute_score(src, gt_summary, pred_summary):
    score = {}
    score['rouge'] = compute_rouge(gt_summary, pred_summary)
    score['length'] = compute_abstractiveness(src, pred_summary)
    return score


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    '''
        This is a custom seq2seq training, supporting an EMA-based generator for calibaration.
    '''
    def __init__(self, teacher_model=None, calibration_params=None, **kwargs,):
        super().__init__(**kwargs)

        # teacher model
        self.teacher_model = teacher_model

        # calibration parameters
        self.calibration = calibration_params['calibration']
        self.num_beams = calibration_params['num_beams']
        self.num_candidate_beams = calibration_params['num_candidate_beams']
        self.diverse_penalty= calibration_params['disversity_penalty']
        self.min_length = calibration_params['min_length']
        self.max_length = calibration_params['max_length']
        self.length_penalty = calibration_params['length_penalty']
        self.abstract_weight = calibration_params['abstract_weight']
        self.mle_weight = calibration_params['mle_weight']
        self.calibration_weight = calibration_params['calibration_weight']


    def compute_loss(self, model, inputs):

        ## pseudo and goldlen labels
        gt_labels = inputs.pop('labels')
        gt_label_ids = inputs.pop('decoder_input_ids')
        decoded_gt_labels = self.tokenizer.batch_decode(gt_label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        ####

        # encoding first.
        attention_mask = inputs["input_ids"] != self.model.config.pad_token_id
        encoder_outputs = model.module.get_encoder()(
            input_ids = inputs["input_ids"],
            attention_mask = attention_mask
            )
        
        decoded_gen_summaries = None
        if self.calibration and self.teacher_model is not None:
            # if calibration is turned on.
            # candidate summary generation using a teacher as generator 
            gen_summaries = self.teacher_model.generate(
                                                        input_ids=inputs['input_ids'], 
                                                        attention_mask=attention_mask,
                                                        num_return_sequences=self.num_candidate_beams, 
                                                        num_beams=self.num_candidate_beams,
                                                        num_beam_groups=self.num_candidate_beams, 
                                                        diversity_penalty=self.diverse_penalty,
                                                        max_length=self.max_length, 
                                                        min_length=self.min_length,
                                                        no_repeat_ngram_size=3,
                                                        length_penalty=self.length_penalty,
                                                        early_stopping=True
                                                        ) 
 
            # decoding to strings
            decoded_gen_summaries = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in gen_summaries]

            # decoder inference with {gold ref, candidate ref}
            calibration_inputs = {
                'inputs': inputs,
                'attention_mask': attention_mask,
                'encoder_outputs': encoder_outputs,
                'decoded_gold': decoded_gt_labels,
                'decoded_candidates': decoded_gen_summaries,
            }
            output, labels = self.calibrated_inference(model, **calibration_inputs)

            # loss computation
            # 1/ target mle loss
            # index '0' is the logit for gt labels, other indices are for the logits of candidiate references
            _target_index = 1 # best label
            logits = output["raw_outputs"][:, _target_index]
            gold = labels[:, _target_index, :]
            mle_loss = self.target_loss(logits, gold)

            # 2/ ranking loss
            # we refer to BRIO papers 
            similarity, gold_similarity = output['score'], output['summary_score']
            ranking_loss = RankingLoss(similarity, gold_similarity)

            # combined loss
            loss = self.mle_weight * mle_loss + self.calibration_weight * ranking_loss 
            #print('mle:', mle_loss, ', rank:', ranking_loss, self.mle_weight, self.calibration_weight)
          
        else:
            # if calibration is not turned on
            calibration_inputs = {
                'inputs': inputs,
                'attention_mask': attention_mask,
                'encoder_outputs': encoder_outputs,
                'decoded_gold': decoded_gt_labels
            }

            # decoder inference with only gold reference
            output, labels = self.uncalibrated_inference(model, **calibration_inputs)

            # get mle loss
            logits = output["raw_outputs"][:, 0]
            gold = labels[:, 0, :]
            mle_loss = self.target_loss(logits, gold)
            loss = mle_loss

        return loss
    
    
    def calibrated_inference(self, model, inputs, attention_mask, encoder_outputs, decoded_gold, decoded_candidates
                             , require_gold=True):
        '''
        Performs decoding inference with calibration.
            model: training model
            inputs: source input (doc)
            attention mask: encoder attention mask as input to decoder
            encoder_outputs: encoder outputs for the input
            decoded_gold: decoded golden reference, which is used to compute rouge score with psuedo references
            decoded_candidates: decoded candidate (pusedo) references from the teacher model, which is used to compute rouge and abstractivenss scores
            require gold: whether to return the logit of gold reference from the BART decoder, which can be used to compute final loss for optimization
        '''

        # train mode: we feed candidate references together with gold reference
        # non-train mode: no need for providing candidate references.
        if decoded_gold is not None:
            train_mode = True
        else:
            train_mode = False

        batch_size = inputs['input_ids'].shape[0]
        decoded_src = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        if train_mode:
            gen_summaries = []

            # sorting the candidate reference based on the specified scores (e.g., rouge)
            for s_idx in range(batch_size):
                src_doc = decoded_src[s_idx]
                gt_label = decoded_gold[s_idx].strip() 
                scored_summaries = []
                for ref_idx in range(self.num_candidate_beams):
                    ref_label = decoded_candidates[s_idx * self.num_candidate_beams + ref_idx].strip()
                    score = compute_score(src_doc, gt_label, ref_label)
                    scored_summaries.append((score, ref_label))
                    
                # normalize & merge scores
                agg_scores = {}
                for (score, summary) in scored_summaries:
                    for key, value in score.items():
                        if key not in agg_scores:
                            agg_scores[key] = []
                        agg_scores[key].append(value)
                
                for key, scores in agg_scores.items():
                    _sum = float(sum(scores))
                    if _sum != 0:
                        scores = [score/_sum for score in scores]
                    agg_scores[key] = np.array(scores)
                    
                final_scores = None
                for key, scores in agg_scores.items():
                    if key == 'rouge':
                        type_weight = 1.0 - self.abstract_weight
                    else:
                        type_weight = self.abstract_weight
                    if final_scores is None:
                        final_scores = (scores * type_weight)
                    else:
                        final_scores += (scores * type_weight)
                scored_summaries = [(final_score, summary) for final_score, (_, summary) in zip(final_scores, scored_summaries)]
                scored_summaries = sorted(scored_summaries, key=lambda tup: tup[0], reverse=True)

                scored_summaries = [summary for (score, summary) in scored_summaries]
                merged_summaries = [gt_label]
                merged_summaries.extend(scored_summaries)

                # gt label first, and then sorted candidate summaries (in desencding order, high -> low)
                gen_summaries.extend(merged_summaries)
            decoded_candidates = gen_summaries

        else:
            # for only gold reference (we turn off calibration during evaluation)
            gen_summaries = []
            for s_idx in range(batch_size):
                scored_summaries = []
                for ref_idx in range(self.num_beams):
                    ref_label = decoded_candidates[s_idx * self.num_beams + ref_idx].strip()
                    gen_summaries.append(ref_label)
            decoded_candidates = gen_summaries

        # tokenizing the candidates and golden reference strings -> decoder inputs
        with self.tokenizer.as_target_tokenizer(): 
            encoded_gen_summaries = self.tokenizer(decoded_candidates, max_length=self.max_length, padding="max_length", truncation=True)
        gen_decoder_input_labels = torch.tensor(encoded_gen_summaries["input_ids"]).to(inputs['input_ids'])
        gen_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=gen_decoder_input_labels)

        # reshape to inference with all the references in a batch
        cand_num = self.num_candidate_beams + 1 if train_mode else self.num_beams
        gen_decoder_input_ids = gen_decoder_input_ids.view(batch_size, cand_num, -1)
        gen_decoder_input_labels = gen_decoder_input_labels.view(batch_size, cand_num, -1)
        cand_mask = gen_decoder_input_labels != self.model.config.pad_token_id

        # interleaving the encoding outputs
        encoder_hidden_states = encoder_outputs[0]
        encoder_hidden_states = torch.repeat_interleave(encoder_hidden_states, cand_num, dim=0)
        attention_mask = torch.repeat_interleave(attention_mask, cand_num, dim=0)
        decoder_input_ids = gen_decoder_input_ids.view(-1, gen_decoder_input_ids.size(-1))
        decoder_attention_mask = cand_mask.view(-1, cand_mask.size(-1))

        # with label smoothing.
        new_inputs = {k: v for k, v in inputs.items()}
        if "labels" in new_inputs:
            new_inputs.pop("labels")
        new_inputs["encoder_outputs"] = [encoder_hidden_states]
        new_inputs["attention_mask"] = attention_mask
        new_inputs["decoder_input_ids"] = decoder_input_ids
        # fine-tuning and calibration, enable this "bi-directional attention - we see the next tokens as well"
        new_inputs["decoder_attention_mask"] = decoder_attention_mask

        outputs = model(**new_inputs)

        # outputs consisting of "logits" and "scores for ranking loss"
        output = self.model.score_forward(outputs,
                                    batch_size=batch_size,
                                    decoder_labels=gen_decoder_input_labels,
                                    length_penalty=self.length_penalty,
                                    require_gold=require_gold,
                                    adding=0)

        return output, gen_decoder_input_labels


    def uncalibrated_inference(self, model, inputs, attention_mask, encoder_outputs, decoded_gold):
        '''
        Decoder inference for only golden reference.
        '''

        batch_size = inputs['input_ids'].shape[0]
        gen_summaries = []
        for s_idx in range(batch_size):
            gt_label = decoded_gold[s_idx].strip()
            gen_summaries.append(gt_label)
        decoded_gen_summaries = gen_summaries

        with self.tokenizer.as_target_tokenizer(): 
            encoded_gen_summaries = self.tokenizer(decoded_gen_summaries, max_length=self.max_length, padding="max_length", truncation=True)
        gen_decoder_input_labels = torch.tensor(encoded_gen_summaries["input_ids"]).to(inputs['input_ids'])
        gen_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=gen_decoder_input_labels)

        cand_num = 1
        gen_decoder_input_ids = gen_decoder_input_ids.view(batch_size, cand_num, -1)
        gen_decoder_input_labels = gen_decoder_input_labels.view(batch_size, cand_num, -1)
        cand_mask = gen_decoder_input_labels != self.model.config.pad_token_id

        encoder_hidden_states = encoder_outputs[0]
        encoder_hidden_states = torch.repeat_interleave(encoder_hidden_states, cand_num, dim=0)
        attention_mask = torch.repeat_interleave(attention_mask, cand_num, dim=0)
        decoder_input_ids = gen_decoder_input_ids.view(-1, gen_decoder_input_ids.size(-1))
        decoder_attention_mask = cand_mask.view(-1, cand_mask.size(-1))

        new_inputs = {k: v for k, v in inputs.items()}
        if "labels" in new_inputs:
            new_inputs.pop("labels")
        new_inputs["encoder_outputs"] = [encoder_hidden_states]
        new_inputs["attention_mask"] = attention_mask
        new_inputs["decoder_input_ids"] = decoder_input_ids
        # for scrach training, disable this.
        #inputs["decoder_attention_mask"] = decoder_attention_mask
        
        outputs = model(**new_inputs)
        output = self.model.score_forward(outputs,
                                    batch_size=batch_size,
                                    decoder_labels=gen_decoder_input_labels,
                                    length_penalty=self.length_penalty,
                                    require_gold=True,
                                    adding=0)

        return output, gen_decoder_input_labels


    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # TODO (Joao): the following line is needed to keep a consistent result on SQUAD. Ideally, we should not block
        # users from preparing a dataset with `decoder_input_ids`.
        inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}

        generated_tokens = self.model.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            num_beams=self.num_beams,
            max_length=self.max_length,
            min_length=self.min_length,
            no_repeat_ngram_size=3,
            length_penalty=self.length_penalty,
            early_stopping=True
            ) 

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None
        
        return loss, generated_tokens, labels
    

    def target_loss(self, model_output, labels, shift_labels=False):
        # this is orignal loss function for seq2seq training with label smoothing.

        epsilon = self.args.label_smoothing_factor # 0.1 default
        ignore_index = self.model.config.pad_token_id

        logits = model_output
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(ignore_index)

        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])

        return (1 - epsilon) * nll_loss + epsilon * smoothed_loss


    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)

        #if is_sagemaker_mp_enabled():
        #    loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #    return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


def RankingLoss(score, summary_score=None, margin=0.001, gold_margin=0, gold_weight=0, no_gold=False, no_cand=False):
    '''
        score: joint probability scores for "beam_num" candidates
        summary_score: for gold summary
    '''
    ones = torch.ones_like(score) # 0.0
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)

    # candidate loss
    n = score.size(1) # num candidates
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)

            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss

    if no_gold:
        return TotalLoss
    
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)

    return TotalLoss


