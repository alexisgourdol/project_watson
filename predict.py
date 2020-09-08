import os
from math import sqrt

import joblib
import pandas as pd
from TaxiFareModel.params import MODEL_NAME, MODEL_VERSION, BUCKET_NAME
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error

PATH_INSIDE_BUCKET = "project_watsondata/train_1k.csv"


def download_model(model_directory="PipelineTest", bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = "{}/{}/{}/{}".format(
        "models", MODEL_NAME, MODEL_VERSION, "model.joblib"
    )
    blob = client.blob(storage_location)
    blob.download_to_filename("model.joblib")
    print(f"=> pipeline downloaded from storage")
    model = joblib.load("model.joblib")
    if rm:
        os.remove("model.joblib")
    return model


def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {"MAE": MAE, "RMSE": RMSE}
    return res


def generate_submission_csv(folder="Pipeline", kaggle_upload=False):
    df_test = get_test_data()
    pipeline = download_model(folder)
    # Check if model savec was the ouptut of RandomSearch or Gridsearch
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df_test)
    else:
        y_pred = pipeline.predict(df_test)
    df_test["fare_amount"] = y_pred
    df_sample = df_test[["key", "fare_amount"]]
    name = f"predictions_{folder}.csv"
    df_sample.to_csv(name, index=False)
    print("prediction saved under kaggle format")
    if kaggle_upload:
        kaggle_message_submission = name[:-4]
        command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
        os.system(command)


if __name__ == "__main__":
    folder = "Pipeline"
    # model = download_model(folder)
    generate_submission_csv(folder, kaggle_upload=True)
