import os

from google.cloud import storage
from termcolor import colored

from project_watson.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION


def storage_upload(model_version, bucket=BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_version,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.joblib')
    print(colored("=> model.joblib uploaded to bucket {} inside {}".format(BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove('model.joblib')
