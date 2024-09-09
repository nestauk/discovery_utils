import json
import logging
import os
import xml.etree.ElementTree as ET

from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd

from botocore.client import BaseClient
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from discovery_utils import PROJECT_DIR
from discovery_utils.utils import s3 as s3


load_dotenv()

# Retrieve AWS file information from environment variables
S3_BUCKET = os.getenv("S3_BUCKET")
# S3_PATH_RAW = os.getenv("S3_PATH_RAW")
# FILE_NAMES_RAW = json.loads(os.getenv("FILE_NAMES_RAW"))

# Retrieve AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HansardGetter:
    def __init__(self):
        self.s3_client = s3.s3_client()
        self.bucket = S3_BUCKET
        self.s3_prefix = "data/policy_scanning_data/"
        logger.info(f"Initialized HansardGetter with bucket: {self.bucket}")

    def s3_client(aws_access_key_id: str = AWS_ACCESS_KEY, aws_secret_access_key: str = AWS_SECRET_KEY) -> BaseClient:
        """Initialize S3 client"""
        S3 = boto3.client("s3", aws_access_key_id, aws_secret_access_key)
        return S3

    def get_parquet(self):
        key = "enriched/HansardDebates.parquet"
        logger.info(f"Downloading debates parquet file: {key}")
        try:
            s3_key = os.path.join(self.s3_prefix, key)
            response = s3._download_obj(self.s3_client, self.bucket, s3_key, download_as="dataframe")
            logger.info(f"Successfully downloaded and read parquet file: {key}")
            return response
        except Exception as e:
            logger.error(f"Error downloading parquet file {key}: {str(e)}")
            raise

    def get_labelstore(self, keywords=True):
        """Debates keyword label store"""

        if keywords == True:
            key = "enriched/HansardDebates_LabelStore_keywords.csv"
        else:
            key = "enriched/HansardDebates_LabelStore_ml.csv"

        logger.info(f"Downloading label store: {key}")

        try:
            s3_key = os.path.join(self.s3_prefix, key)
            response = s3._download_obj(self.s3_client, self.bucket, s3_key, download_as="dataframe")
            logger.info(f"Successfully downloaded and read labelstore: {key}")
            return response
        except Exception as e:
            logger.error(f"Error downloading CSV file {key}: {str(e)}")
            raise

    def get_latest_debates(self):
        logger.info("Downloading new XML files since last sync")
        temp_dir = PROJECT_DIR / "tmp/debates"
        os.makedirs(temp_dir, exist_ok=True)
        s3_key = os.path.join(self.s3_prefix, "house_of_commons", "files_to_sync.log")
        try:
            files_to_sync = (
                self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)["Body"].read().decode("utf-8").splitlines()
            )
            logger.info(f"Found {len(files_to_sync)} files to sync")
            downloaded_files = []
            for file in files_to_sync:
                local_file_path = os.path.join(temp_dir, os.path.basename(file))
                logger.debug(f"Downloading XML file: {file} to {local_file_path}")
                s3_key = os.path.join(self.s3_prefix, "house_of_commons", file)
                self.s3_client.download_file(self.bucket, s3_key, local_file_path)
                downloaded_files.append(local_file_path)
            logger.info(f"Successfully downloaded {len(downloaded_files)} XML files")
            return downloaded_files
        except Exception as e:
            logger.error(f"Error downloading XML files: {str(e)}")
            raise

    def get_all_debates(self):
        logger.info("Retrieving all XML debate files from S3")
        temp_dir = Path("tmp/debates")
        temp_dir.mkdir(parents=True, exist_ok=True)

        s3_prefix = os.path.join(self.s3_prefix, "house_of_commons")

        try:
            # List all objects in the S3 directory
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix)

            downloaded_files = []
            for page in pages:
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".xml"):
                        file_name = os.path.basename(obj["Key"])
                        local_file_path = temp_dir / file_name
                        logger.debug(f"Downloading XML file: {obj['Key']} to {local_file_path}")

                        self.s3_client.download_file(self.bucket, obj["Key"], str(local_file_path))
                        downloaded_files.append(str(local_file_path))

            logger.info(f"Successfully downloaded {len(downloaded_files)} XML files")
            return downloaded_files

        except ClientError as e:
            logger.error(f"ClientError downloading XML files: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading XML files: {str(e)}")
            raise

    def get_people_metadata(self):
        s3_key = os.path.join(self.s3_prefix, "house_of_commons", "people.json")
        temp_dir = PROJECT_DIR / "tmp/"
        try:
            response = s3._download_obj(self.s3_client, self.bucket, s3_key, download_as="dict")
            logger.info(f"Successfully downloaded json file: {s3_key}")
            with open(os.path.join(temp_dir, "people.json"), "w") as fp:
                json.dump(response, fp)
        except Exception as e:
            logger.error(f"Error downloading CSV file {s3_key}: {str(e)}")
            raise
