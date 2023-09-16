#!/usr/bin/env python3
"""
Question answering
"""


import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer


def question_answer(question, reference):
    model_url = "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
    model = hub.load(model_url)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + reference_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_ids = tf.constant([input_ids])

    start_logits, end_logits = model(input_ids)

    start_idx = tf.argmax(start_logits, axis=1)[0].numpy().item()
    end_idx = tf.argmax(end_logits, axis=1)[0].numpy().item()

    answer_tokens = reference_tokens[start_idx:end_idx + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer
