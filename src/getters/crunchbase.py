"""
discovery_utils.src.getters.crunchbase.py

Module for easy access to downloaded CB data on S3.

"""
import pandas as pd
import boto3
import os
from dotenv import load_dotenv
from botocore.client import BaseClient
from src.utils import s3 as s3
from src.utils import timestamps as ts
from datetime import datetime, timedelta
import logging
import json

# logging level
logging.basicConfig(level=logging.INFO)

load_dotenv()

# Retrieve AWS file information from environment variables
BUCKET_NAME_RAW = os.getenv("BUCKET_NAME_RAW")
S3_PATH_RAW = os.getenv("S3_PATH_RAW")
FILE_NAMES_RAW = json.loads(os.getenv("FILE_NAMES_RAW"))


# VERSATILE FUNCTION FOR GENERALISED USE

def get_table(timestamp: str,
              file_name: str,
              s3_client: BaseClient,
              file_names_raw: list = FILE_NAMES_RAW,
              bucket_name_raw: str = BUCKET_NAME_RAW
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


# FUNCTIONS FOR SPECIFIC USE: TO GET TODAY'S OR YESTERDAY'S SNAPSHOT
# USES THE VERSATILE FUNCTION ABOVE

def get_tdy_table(file_name: str,
                  s3_client: BaseClient,
                  ) -> pd.DataFrame:
    """Specific table from today's CB snapshot"""
    # Get today's date in YYYY-MM-DD format
    today_timestamp = datetime.now().strftime('%Y-%m-%d')
    
    return get_table(today_timestamp, file_name, s3_client)

def get_ytd_table(file_name: str,
                   s3_client: BaseClient,
                   ) -> pd.DataFrame:
    """Specific table from yesterday's CB snapshot"""
    # Get yesterday's date in YYYY-MM-DD format
    yesterday_timestamp = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    return get_table(yesterday_timestamp, file_name, s3_client)