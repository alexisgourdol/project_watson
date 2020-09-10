""" Main lib for project_watson Project
"""

import multiprocessing
import time
import warnings
from tempfile import mkdtemp

import joblib
import mlflow
import pandas as pd
import numpy as np

import sys

from project_watson.data import get_data, get_snli
from project_watson.params import MLFLOW_URI
from project_watson.utils import simple_time_tracker
from project_watson.model import *

from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


class Configuration():
    """
    All configuration for running an experiment
    """
    def __init__(
        self,
        model_name,
        translation = True,
        max_length = 64,
        padding = True,
        batch_size = 128,
        epochs = 5,
        learning_rate = 1e-5,
        metrics = ["sparse_categorical_accuracy"],
        verbose = 1,
        train_splits = 5,
        accelerator = "TPU",
        myluckynumber = 13
    ):
        # seed and accelerator
        self.SEED = myluckynumber

        # paths
        self.PATH_TRAIN = "project_watson/data/train.csv"
        self.PATH_TEST  = "project_watson/data/test.csv"

        # splits
        self.TRAIN_SPLITS = train_splits

        # mapping of language
        self.LANGUAGE_MAP = {
            "English"   : 0,
            "Chinese"   : 1,
            "Arabic"    : 2,
            "French"    : 3,
            "Swahili"   : 4,
            "Urdu"      : 5,
            "Vietnamese": 6,
            "Russian"   : 7,
            "Hindi"     : 8,
            "Greek"     : 9,
            "Thai"      : 10,
            "Spanish"   : 11,
            "German"    : 12,
            "Turkish"   : 13,
            "Bulgarian" : 14
        }

        self.INVERSE_LANGUAGE_MAP = {v: k for k, v in self.LANGUAGE_MAP.items()}

        # model configuration
        self.MODEL_NAME = model_name
        self.TRANSLATION = translation
        # self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        # model hyperparameters
        self.MAX_LENGTH = max_length
        self.PAD_TO_MAX_LENGTH = padding
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate
        self.METRICS = metrics
        self.VERBOSE = verbose

        # initializing accelerator
        # self.initialize_accelerator()

    # def initialize_accelerator(self):
    #     """
    #     Initializing accelerator
    #     """
    #     # checking TPU first
    #     if self.ACCELERATOR == "TPU":
    #         print("Connecting to TPU")
    #         try:
    #             tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    #             print(f"Running on TPU {tpu.master()}")
    #         except ValueError:
    #             print("Could not connect to TPU")
    #             tpu = None

    #         if tpu:
    #             try:
    #                 print("Initializing TPU")
    #                 tf.config.experimental_connect_to_cluster(tpu)
    #                 tf.tpu.experimental.initialize_tpu_system(tpu)
    #                 self.strategy = tf.distribute.experimental.TPUStrategy(tpu)
    #                 self.tpu = tpu
    #                 print("TPU initialized")
    #             except:
    #                 e = sys.exc_info()[0]
    #                 print( "Error TPU not initialized: %s" % e )
    #         else:
    #             print("Unable to initialize TPU")
    #             self.ACCELERATOR = "GPU"

    #     # default for CPU and GPU
    #     if self.ACCELERATOR != "TPU":
    #         print("Using default strategy for CPU and single GPU")
    #         self.strategy = tf.distribute.get_strategy()

    #     # checking GPUs
    #     if self.ACCELERATOR == "GPU":
    #         print(f"GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")

    #     # defining replicas
    #     self.AUTO = tf.data.experimental.AUTOTUNE
    #     self.REPLICAS = self.strategy.num_replicas_in_sync
    #     print(f"REPLICAS: {self.REPLICAS}")

    def train(self):
        try:
            print("Initializing TPU")
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            print("TPU initialized")
        except:
            e = sys.exc_info()[0]
            print( "Error TPU not initialized: %s" % e )

        params = dict(
        model_name="bert-base-multilingual-cased",
        max_len=50,
        )

        self.model = build_model(**params)

        df = get_data()
        X = df.drop(columns=['label'], axis=1)
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
        tokenizer = create_tokenizer()
        train_input = bert_encode(X_train.premise.values, X_train.hypothesis.values, tokenizer)
        test_input = bert_encode(X_test.premise.values, X_test.hypothesis.values, tokenizer)

        self.model.fit(train_input, y_train, epochs = 5, verbose = 2, batch_size = 32, validation_split = 0.3,learning_rate = 1e-5,)

    def pred(self, X_pred):
        predictions = [np.argmax(i) for i in model.predict(X_pred)]
        return  predictions

    def accuracy(self, y_pred, y_true):
        return sum(y_pred == y_true) / len(y_pred)


if __name__ == '__main__':

    conf = Configuration(
        model_name = 'bert-base-multilingual-cased',
        translation = True,
        max_length = 64,
        padding = True,
        batch_size = 128,
        epochs = 3,
        metrics = ["sparse_categorical_accuracy"],
        verbose = 1,
        )

    conf.train()

    # test = pd.read_csv("data/test.csv")
    # test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer)

    # y_pred = conf.pred(test_input)
    # acc = conf.accuracy(y_pred, y_true)
