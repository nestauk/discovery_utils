"""
discovery_utils.discovery_utils.getters.crunchbase.py

Module for easy access to downloaded CB data on S3.

"""
import json
import logging
import os

from datetime import datetime
from datetime import timedelta

import boto3
import pandas as pd

from botocore.client import BaseClient
from dotenv import load_dotenv

from discovery_utils.utils import s3 as s3
from discovery_utils.utils import timestamps as ts


# logging level
logging.basicConfig(level=logging.INFO)

load_dotenv()

# Retrieve AWS file information from environment variables
BUCKET_NAME_RAW = os.getenv("BUCKET_NAME_RAW")
S3_PATH_RAW = os.getenv("S3_PATH_RAW")
FILE_NAMES_RAW = json.loads(os.getenv("FILE_NAMES_RAW"))

# Retrieve AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")


# Left these here for when they are used using AWS CLI
# os.environ["AWS_ACCESS_KEY"]
# os.environ["AWS_SECRET_KEY"]


# VERSATILE FUNCTION FOR GENERALISED USE


def _s3_client(aws_access_key_id: str = AWS_ACCESS_KEY, aws_secret_access_key: str = AWS_SECRET_KEY) -> BaseClient:
    """Initialize S3 client"""
    S3 = boto3.client("s3", aws_access_key_id, aws_secret_access_key)
    return S3


def get_table(
    timestamp: str,
    file_name: str,
    s3_client: BaseClient,
    file_names_raw: list = FILE_NAMES_RAW,
    bucket_name_raw: str = BUCKET_NAME_RAW,
) -> pd.DataFrame:
    """Specific table with named CB data from a specific version, dynamically selecting the exact file match."""

    # Validate file name is in the list of known file names without extensions
    if file_name not in file_names_raw:
        raise ValueError(f"File '{file_name}' not found in FILE_NAMES_RAW.")

    # Construct the directory path for the given timestamp
    dir_path = S3_PATH_RAW + "Crunchbase_" + timestamp + "/"

    # Retrieve all file names in the directory
    all_files = s3._get_bucket_filenames(bucket_name_raw, dir_path)

    # Match the file name to the exact file in the directory
    s3_key = s3._match_one_file(all_files, file_name, dir_path)

    response = s3._download_obj(s3_client, bucket_name_raw, s3_key, download_as="dataframe")

    return response


# FUNCTIONS FOR SPECIFIC USE: TO GET LATEST OR SECOND LATEST SNAPSHOT, OR TODAY'S OR YESTERDAY'S SNAPSHOT
# USES THE VERSATILE FUNCTION ABOVE


def get_tdy_table(
    file_name: str,
    s3_client: BaseClient,
) -> pd.DataFrame:
    """Specific table from today's CB snapshot"""
    # Get today's date in YYYY-MM-DD format
    today_timestamp = datetime.now().strftime("%Y-%m-%d")

    return get_table(today_timestamp, file_name, s3_client)


def get_ytd_table(
    file_name: str,
    s3_client: BaseClient,
) -> pd.DataFrame:
    """Specific table from yesterday's CB snapshot"""
    # Get yesterday's date in YYYY-MM-DD format
    yesterday_timestamp = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    return get_table(yesterday_timestamp, file_name, s3_client)


# THE LATEST AND SECOND LATEST SNAPSHOTS NEED TESTING
def latest_table(
    file_name: str,
    s3_client: BaseClient,
    bucket_name_raw: str = BUCKET_NAME_RAW,
    file_names_raw: list = FILE_NAMES_RAW,
) -> pd.DataFrame:
    """Specific table from latest CB snapshot"""
    # Get the latest timestamp
    directories = s3._list_directories(s3_client, bucket_name_raw, S3_PATH_RAW)
    timestamps = ts._timestamp_list(directories)
    latest_timestamp = ts._directory(directories, timestamps[0])

    return get_table(latest_timestamp, file_name, s3_client, file_names_raw, bucket_name_raw)


def secondlatest_table(
    file_name: str,
    s3_client: BaseClient,
    bucket_name_raw: str = BUCKET_NAME_RAW,
    file_names_raw: list = FILE_NAMES_RAW,
) -> pd.DataFrame:
    """Specific table from latest CB snapshot"""
    # Get the latest timestamp
    directories = s3._list_directories(s3_client, bucket_name_raw, S3_PATH_RAW)
    timestamps = ts._timestamp_list(directories)
    secondlatest_timestamp = ts._directory(directories, timestamps[1])

    return get_table(secondlatest_timestamp, file_name, s3_client, file_names_raw, bucket_name_raw)
