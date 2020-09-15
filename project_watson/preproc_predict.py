import numpy as np
import pandas as pd
import requests
import json
import os
from transformers import BertTokenizer, TFBertModel
import matplotlib.pyplot as plt
import tensorflow as tf

TEST_FILE_PATH = './'
TEST_FILE_NAME = 'test.csv' # needs to be a csv file
URL_PREDICTION = 'http://localhost:8501/v1/models/model:predict' # needs to updated accordingly

def encode_sentence(s):
    """ Encode One sentence s using the tokenizer defined in this .py"""
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)

def bert_encode(hypotheses, premises, tokenizer):
    """Returns a mathematical representation of the text inputs using the defined
    tokenizer. The model expects 'input_word_ids', 'input_mask', 'input_type_ids'"""

    sentence1 = tf.ragged.constant([encode_sentence(s) for s in np.array(hypotheses)])
    sentence2 = tf.ragged.constant([encode_sentence(s) for s in np.array(premises)])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()

    inputs = {'input_word_ids': input_word_ids.to_tensor(),
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

    return inputs

def format_input(test, tokenizer, max_len=50):
    """Reduces #tokens to 50 (max_len), as expected by the model to predict."""
    test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer)

    # take the 1st 50 tokens of the 20 first lines and convert to list. Why 20 ? No reason, could be 1...
    test_input_slice = {'input_word_ids': tf.slice(test_input['input_word_ids'], [0, 0], [20, max_len]).numpy().tolist(),
                        'input_mask' : tf.slice(test_input['input_mask'], [0, 0], [20, max_len]).numpy().tolist(),
                        'input_type_ids' : tf.slice(test_input['input_type_ids'], [0, 0], [20, max_len]).numpy().tolist()}

    #format data as expected by the model for a prediction AND KEEP ONLY THE 1ST LINE (TO DO : refactor)
    # ==> {'instances': [{'input_word_ids': [101, 10111, ... 11762] , 'input_mask': [1, 1, ... 1], 'input_type_ids': [0, 0, ... 0]

    data = {}
    for k in test_input_slice.keys():
        data[k] = test_input_slice[k][0]

    data = {'instances': [data]}

    return data

def predict(data, url=URL_PREDICTION ):
    r = requests.post(url, data=json.dumps(data))

if __name__ == '__main__':
    model_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    max_len = 50
    test = pd.read_csv(TEST_FILE_PATH + TEST_FILE_NAME)

    data = format_input(test=test, max_len=50, tokenizer=tokenizer)

    predict(data)
