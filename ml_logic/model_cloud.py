import os
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
import pickle as pkl
from google.cloud import storage

import mlflow
from mlflow.tracking import MlflowClient
import time


def cloud_upload_model(bucket_name,storage_filename,local_filename):

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(storage_filename)
    blob.upload_from_filename(local_filename)

def cloud_download_model(bucket_name,storage_filename,local_filename):

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(storage_filename)
    blob.download_to_filename(local_filename)

    return local_filename
