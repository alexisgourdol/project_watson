import multiprocessing
import time
import warnings
from tempfile import mkdtemp
from google.cloud import storage
import joblib
import mlflow
import pandas as pd
from project_watson.data import get_data, encode_sentence
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from psutil import virtual_memory
from project_watson.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, MLFLOW_URI
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

class TrainerEdgar(object):
    # Mlflow parameters identifying the experiment, you can add all the parameters you wish
    ESTIMATOR = "Linear"
    EXPERIMENT_NAME = "Project_Watson"

    def __init__(self, train, **kwargs):
        """
        FYI:
        __init__ is called every time you instatiate Trainer
        Consider kwargs as a dict containig all possible parameters given to your constructor
        Example:
            TT = Trainer(nrows=1000, estimator="Linear")
               ==> kwargs = {"nrows": 1000,
                            "estimator": "Linear"}
        :param X:
        :param y:
        :param kwargs:
        """
        self.model_name = model_name
        self.kwargs = kwargs
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to nlflow
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)  # cf doc above
        self.model_params = None  # for
        self.train = train
        self.split = self.kwargs.get("split", True)  # cf doc above
        self.nrows = self.train
        self.log_kwargs_params()
        self.log_machine_specs()
        self.model
        self.tokenizer = create_tokenizer(model_name)


    def create_tokenizer(model_name="jplu/tf-xlm-roberta-base"):
        '''
        method to initialize the tokenizer of the nlp task.
        '''
        tokenizer = BertTokenizer.from_pretrained(model_name)
        return tokenizer


    def bert_encode(self, hypotheses, premises):
        num_examples = len(hypotheses)

        sentence1 = tf.ragged.constant([encode_sentence(s) for s in np.array(hypotheses)])
        sentence2 = tf.ragged.constant([encode_sentence(s) for s in np.array(premises)])

        cls = [self.tokenizer.convert_tokens_to_ids(["[CLS]"])] * sentence1.shape[0]
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


    def build_model(model_name):
        bert_encoder = TFBertModel.from_pretrained(model_name)
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


    @simple_time_tracker
    def train(self, gridsearch=False):
        tic = time.time()
        self.model = build_model(self.model_name)
        train_input = bert_encode(self.train.premise.values, self.train.hypothesis.values)
        self.model.fit(train_input, self.train.label.values)

        # mlflow logs
        self.mlflow_log_metric("train_time", int(time.time() - tic))

    def evaluate(self):
        rmse_train = self.compute_rmse(self.X_train, self.y_train)
        self.mlflow_log_metric("rmse_train", rmse_train)
        if self.split:
            rmse_val = self.compute_rmse(self.X_val, self.y_val, show=True)
            self.mlflow_log_metric("rmse_val", rmse_val)
            print(colored("rmse train: {} || rmse val: {}".format(rmse_train, rmse_val), "blue"))
        else:
            print(colored("rmse train: {}".format(rmse_train), "blue"))

    def compute_rmse(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(5), "blue"))
        rmse = compute_rmse(y_pred, y_test)
        return round(rmse, 3)

    def save_model(self, upload=True, auto_remove=True):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

        # Add upload of model.joblib to storage here
        storage_location = '{}/{}/{}/{}'.format(
            'models',
            MODEL_NAME,
            MODEL_VERSION,
            'model.joblib')
        client = storage.Client().bucket(BUCKET_NAME)
        blob = client.blob(storage_location)
        blob.upload_from_filename('model.joblib')

        print("uploaded model.joblib to gcp cloud storage under \n => {}".format(storage_location))
    ### MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def log_estimator_params(self):
        reg = self.get_estimator()
        self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        params = reg.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

    def log_machine_specs(self):
        cpus = multiprocessing.cpu_count()
        mem = virtual_memory()
        ram = int(mem.total / 1000000000)
        self.mlflow_log_param("ram", ram)
        self.mlflow_log_param("cpus", cpus)


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    # Get and clean data
    experiment = "taxifare_set_YOURNAME"
    params = dict(
        nrows=1000,
        upload=True,  # upload model.job lib to strage if set to True
        local=False,  # set to False to get data from GCP Storage
        model_name = 'bert-base-multilingual-cased', #model_name
        mlflow=True,  # set to True to log params to mlflow
        experiment_name=experiment,
    )
    print("############   Loading Data   ############")
    train = pd.read_csv("/kaggle/input/contradictory-my-dear-watson/train.csv")
    train_input = bert_encode(train.premise.values, train.hypothesis.values, tokenizer)
    del df
    print("shape: {}".format(X_train.shape))
    print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))
    # Train and save model, locally and
    t = TrainerEdgar(train=train, **params)
    del X_train, y_train
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    t.save_model()
