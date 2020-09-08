"""The aim of this file is to define data import functions"""
import pandas as pd
from google.cloud import storage

from TaxiFareModel.utils import simple_time_tracker

from TaxiFareModel.params import (
    BUCKET_NAME,
    BUCKET_TRAIN_DATA_PATH,
    MODEL_NAME,
    MODEL_VERSION,
    MLFLOW_URI,
)


@simple_time_tracker
def get_data(nrows=10000, local=False, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    client = storage.Client()
    if local:
        path = "data/train.csv"
    else:
        path = "gs://{}/{}".format(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH)
    df = pd.read_csv(path, nrows=nrows)
    return df


if __name__ == "__main__":
    params = dict(
        nrows=1000,
        local=False,  # set to False to get data from GCP (Storage or BigQuery)
    )
    df = get_data(**params)
