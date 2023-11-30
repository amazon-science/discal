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

import json
import re
from collections import defaultdict

import nltk
import numpy as np
import evaluate

PREDICTION_SPANS = ['ISSUE', 'OUTCOME', 'NEXTSTEP']
HF_ROUGE = evaluate.load('rouge')
HF_BERTSCORE = evaluate.load('bertscore')
HF_F1 = evaluate.load('f1')


def extract_text(span_index, span_index_all, pred):
    if span_index == None:
        return ""

    span_index_all_after = [(start, end) for (start, end) in span_index_all if start > span_index[0]]
    start_index = span_index[1] + 1
    if len(span_index_all_after) >= 1:
        end_index = span_index_all_after[0][0] - 1
    else:
        end_index = len(pred)
    return pred[start_index:end_index]


def refine_pred(pred, log=False):
    """
    Refine generated prediction from DialogLM. Refined prediction follows the format "[ISSUE] %s [OUTCOME] %s [NEXTSTEP] %s".
    """
    if log:
        print("pred:")
        print(pred)
    pred_index = {}
    pred_index["all"] = []
    for span_type in PREDICTION_SPANS:
        pred_index[span_type] = []
        count = 0
        for match in re.finditer(r'\[%s\]' % span_type, pred):
            count += 1
            if log:
                print("match", count, match.group(), "start index", match.start(), "End index", match.end())
            pred_index[span_type].append((match.start(), match.end()))
            pred_index["all"].append((match.start(), match.end()))

    pred_index["all"].sort(key=lambda x: x[0])
    pred_by_type = {}
    for span_type in PREDICTION_SPANS:
        if pred_index[span_type] == []:
            span_index = None
        else:
            # choose the first occurance of special token
            span_index = pred_index[span_type][0]
        pred_by_type[span_type] = extract_text(span_index, pred_index["all"], pred)
    pred_by_type["COMBINED"] = "[ISSUE] %s [OUTCOME] %s [NEXTSTEP] %s" % (
        pred_by_type["ISSUE"], pred_by_type["OUTCOME"], pred_by_type["NEXTSTEP"])
    return pred_by_type


def filter_out_span_tokens(pred):
    for span in PREDICTION_SPANS:
        pred = pred.replace(f'[{span}]', '')
    return {'SUMMARY': pred}


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def length(predictions, pad_token_id):
    return {"gen_len": np.array([np.count_nonzero(np.array(pred) != pad_token_id) for pred in predictions])}


def huggingface_rouge_wrapper(preds, labels, use_stemmer=False):
    # Some simple post-processing
    preds, span_labels = postprocess_text(preds, labels)

    metric_values = HF_ROUGE.compute(
        predictions=preds,
        references=labels,
        use_stemmer=use_stemmer,
        use_aggregator=False,  # using manual aggregation if set via aggregate agrument
    )
    return {key: np.array(values) for key, values in metric_values.items()}


def huggingface_bertscore_wrapper(preds, labels, model_type='bert-large-uncased', attribute='recall'):
    # Some simple post-processing
    preds, span_labels = postprocess_text(preds, labels)

    metric_values = HF_BERTSCORE.compute(
        predictions=preds,
        references=labels,
        model_type=model_type,
    )
    return {f'bertscore_{attribute}': np.array(metric_values[attribute])}


def huggingface_f1_wrapper(preds, labels):
    assert len(preds) == len(labels), "Unequal number of references and predictions for F1 calculation"
    return HF_F1.compute(predictions=preds, references=labels)


def compute_metrics(eval_preds, tokenizer, data_args, aggregate=False):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    postprocess_fn = refine_pred if data_args.eval_type == 'spans_combined' else filter_out_span_tokens
    eval_spans = PREDICTION_SPANS if data_args.eval_type == 'spans_combined' else ['SUMMARY']
    pred_spans, label_spans = {span: [] for span in eval_spans}, {span: [] for span in eval_spans}

    for pred in decoded_preds:
        pred_parsed = postprocess_fn(pred)
        for span in eval_spans:
            pred_spans[span].append(pred_parsed[span])

    for label in decoded_labels:
        label_parsed = postprocess_fn(label)
        for span in eval_spans:
            label_spans[span].append(label_parsed[span])
    result = defaultdict(lambda: 0)
    for span in eval_spans:
        current_span_preds, current_span_labels = pred_spans[span], label_spans[span]
        current_span_result = {}
        for metric in [huggingface_rouge_wrapper, huggingface_bertscore_wrapper]:
            current_span_result.update(metric(current_span_preds, current_span_labels))
        for key, value in current_span_result.items():
            result[f"{span}_{key}"] = value
            # averaging over spans
            result[f"AVG_{key}"] += np.array(value) / len(eval_spans)
        # here, we need to tokenize the spans again in order to determine generated length in tokens
        current_span_length = length(
            [tokenizer.encode(pred) for pred in current_span_preds],
            tokenizer.pad_token_id,
        )
        for key, value in current_span_length.items():
            result[f"{span}_{key}"] = value

        span_is_empty = lambda x: len(x) == 0 or x.strip().lower() == 'nan'
        preds_nonempty_binary = [not span_is_empty(pred) for pred in current_span_preds]
        labels_nonempty_binary = [not span_is_empty(label) for label in current_span_labels]
        for key, value in huggingface_f1_wrapper(preds_nonempty_binary, labels_nonempty_binary).items():
            # duplicating f1 score for aggregation later
            result[f"{span}_nonempty_{key}"] = np.array([value for _ in range(len(preds_nonempty_binary))])
        result[f"{span}_preds_nonempty_binary"] = np.array(preds_nonempty_binary)
        result[f"{span}_labels_nonempty_binary"] = np.array(labels_nonempty_binary)
    if aggregate:
        for key, value in result.items():
            result[key] = np.mean(value)
        # averaging over spans and Rouge-X metrics (useful for checkpoint selection)
        result['AVG_rouge'] = np.mean([value for key, value in result.items() if key.startswith('AVG_')])
    return dict(result)


def aggregate_metrics_by_references(metric_references):
    assert len(metric_references), "No metrics to aggregate"

    result = {}
    for key in metric_references[0]:
        metric_concat = np.array([metric[key] for metric in metric_references])
        result[f"{key}_max"] = np.mean(np.max(metric_concat, axis=0))
        result[f"{key}_mean"] = np.mean(np.mean(metric_concat, axis=0))
    return result


def offline_compute_metrics(eval_preds, aggregate=False, eval_type='plain'):
    '''
        This function is a modified "compute_metrics" for offline evaluation.
        Can be merged later with "compute_metrics" function
    '''
    assert eval_type in ['spans_combined', 'plain']
    decoded_preds, decoded_labels = eval_preds

    refine_fn = refine_pred if eval_type == 'spans_combined' else filter_out_span_tokens
    eval_spans = PREDICTION_SPANS if eval_type == 'spans_combined' else ['SUMMARY']
    pred_spans, label_spans = {span: [] for span in eval_spans}, {span: [] for span in eval_spans}

    for pred in decoded_preds:
        pred_parsed = refine_fn(pred)
        for span in eval_spans:
            pred_spans[span].append(pred_parsed[span])

    for label in decoded_labels:
        label_parsed = refine_fn(label)
        for span in eval_spans:
            label_spans[span].append(label_parsed[span])

    result = defaultdict(lambda: 0)
    for span in eval_spans:
        current_span_preds, current_span_labels = pred_spans[span], label_spans[span]
        current_span_result = {}
        for metric in [huggingface_rouge_wrapper, huggingface_bertscore_wrapper]:
            current_span_result.update(metric(current_span_preds, current_span_labels))
        for key, value in current_span_result.items():
            result[f"{span}_{key}"] = value
            # averaging over spans
            result[f"AVG_{key}"] += np.array(value) / len(eval_spans)

        span_is_empty = lambda x: len(x) == 0 or x.strip().lower() == 'nan'
        preds_nonempty_binary = [not span_is_empty(pred) for pred in current_span_preds]
        labels_nonempty_binary = [not span_is_empty(label) for label in current_span_labels]
        for key, value in huggingface_f1_wrapper(preds_nonempty_binary, labels_nonempty_binary).items():
            # duplicating f1 score for aggregation later
            result[f"{span}_nonempty_{key}"] = np.array([value for _ in range(len(preds_nonempty_binary))])
        result[f"{span}_preds_nonempty_binary"] = np.array(preds_nonempty_binary)
        result[f"{span}_labels_nonempty_binary"] = np.array(labels_nonempty_binary)

    if aggregate:
        for key, value in result.items():
            result[key] = np.mean(value)
        # averaging over spans and Rouge-X metrics (useful for checkpoint selection)
        result['AVG_rouge'] = np.mean([value for key, value in result.items() if key.startswith('AVG_')])

    return dict(result)


def offline_evaluation(gt_path, pred_csv_path, eval_type='plain'):
    """
        This function returns a dictionary of including all the evaluation metircs in an offline fashion.

        Args:
            gt_path: path for ground-truth test data
            pred_csv_path: path for generated summaries (.csv format)
        Return:
            a dictionary including all metric results
    """
    assert eval_type in ['spans_combined', 'plain']

    collection = {}
    for line in open(gt_path, 'r'):
        line = json.loads(line)
        conv_id = line['Convo_ID']

        if conv_id not in collection:
            collection[conv_id] = {'tgt': []}

        summary = line['tgt']
        summary = summary.strip()
        collection[conv_id]['tgt'].append(summary)

    for line in open(pred_csv_path, 'r'):
        line = json.loads(line)
        conv_id = line['Convo_ID']
        summary = line['tgt']

        summary = summary.strip()
        collection[conv_id]['pred'] = summary

    total_scores = {}
    for conv_id, item in collection.items():

        multiple_tgts = item['tgt']
        pred = item['pred']
        
        # Given an input, obtains score for multiple reference
        eval_preds = ([pred] * len(multiple_tgts), multiple_tgts)
        scores = offline_compute_metrics(eval_preds, eval_type=eval_type)

        if len(total_scores) == 0:
            for key, value in scores.items():
                total_scores[key + '_max'] = [np.max(value)]
                total_scores[key + '_mean'] = [np.mean(value)]
        else:
            for key, value in scores.items():
                total_scores[key + '_max'].append(np.max(value))
                total_scores[key + '_mean'].append(np.mean(value))

    return {key: np.mean(value) for key, value in total_scores.items()}