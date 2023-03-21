import boto3
from io import BytesIO
import joblib
import requests
import json
import logging
from requests.auth import HTTPBasicAuth
import time


logging.basicConfig()
logger = logging.getLogger()
logging.getLogger("botocore").setLevel(logging.ERROR)
logger.setLevel(logging.INFO)


def write_joblib(file, path):
    """
    Function to write a joblib file to an s3 bucket or local directory.
    Arguments:
    * file: The file that you want to save
    * path: an s3 bucket or local directory path.
    """

    # Path is an s3 bucket
    if path[:5] == "s3://":
        s3_bucket, s3_key = path.split("/")[2], path.split("/")[3:]
        s3_key = "/".join(s3_key)
        with BytesIO() as f:
            joblib.dump(file, f)
            f.seek(0)
            boto3.client("s3").upload_fileobj(Bucket=s3_bucket, Key=s3_key, Fileobj=f)

    # Path is a local directory
    else:
        with open(path, "wb") as f:
            joblib.dump(file, f)


def read_joblib(path):
    """
    Function to load a joblib file from an s3 bucket or local directory.
    Arguments:
    * path: an s3 bucket or local directory path where the file is stored
    Outputs:
    * file: Joblib file loaded
    """

    # Path is an s3 bucket
    if path[:5] == "s3://":
        s3_bucket, s3_key = path.split("/")[2], path.split("/")[3:]
        s3_key = "/".join(s3_key)
        with BytesIO() as f:
            boto3.client("s3").download_fileobj(Bucket=s3_bucket, Key=s3_key, Fileobj=f)
            f.seek(0)
            file = joblib.load(f)

    # Path is a local directory
    else:
        with open(path, "rb") as f:
            file = joblib.load(f)

    return file


def send_message(payload, webhook_url, webhook_id, webhook_key):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    resp = requests.post(
        url=webhook_url,
        data=json.dumps(payload),
        auth=HTTPBasicAuth(webhook_id, webhook_key),
        headers=headers,
    )
    logger.info(resp.content)
    logger.info(f"Status: {resp.status_code}, Message: {resp.reason}")


def timer(func):
    # Decorator wrapper function to time process
    def wrapper(*args, **kwargs):
        before = time.time()
        response = func(*args, **kwargs)
        print("Process takes {:.2f} seconds".format((time.time() - before)))
        return response

    return wrapper
