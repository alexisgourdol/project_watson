####### INUTILE À SUPPRIMER

from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np


def create_tokenizer(model_name="jplu/tf-xlm-roberta-base"):
    '''
    method to initialize the tokenizer of the nlp task.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def encode_sentence(s, model_name="jplu/tf-xlm-roberta-base"):
    tokenizer = create_tokenizer(model_name)
    tokens = list(tokenizer.tokenize(s))
    tokens.append("[SEP]")
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(hypotheses, premises, tokenizer):
    num_examples = len(hypotheses)

    sentence1 = tf.ragged.constant([encode_sentence(s) for s in np.array(hypotheses)])
    sentence2 = tf.ragged.constant([encode_sentence(s) for s in np.array(premises)])

    cls = [tokenizer.convert_tokens_to_ids(["[CLS]"])] * sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()

    inputs = {
        "input_word_ids": input_word_ids.to_tensor(),
        "input_mask": input_mask,
        "input_type_ids": input_type_ids,
    }

    return inputs


def build_model(model_name, max_len):
    bert_encoder = TFAutoModel.from_pretrained(model_name)
    input_word_ids = tf.keras.Input(
        shape=(max_len,), dtype=tf.int32, name="input_word_ids"
    )
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.Input(
        shape=(max_len,), dtype=tf.int32, name="input_type_ids"
    )

    embedding = bert_encoder([input_word_ids, input_mask, input_type_ids])[0]
    output = tf.keras.layers.Dense(3, activation="softmax")(embedding[:, 0, :])

    model = tf.keras.Model(
        inputs=[input_word_ids, input_mask, input_type_ids], outputs=output
    )
    model.compile(
        tf.keras.optimizers.Adam(lr=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    params = dict(
        model_name="jplu/tf-xlm-roberta-base",
        max_len=50,
    )
    model = build_model(**params)
    model.summary()
