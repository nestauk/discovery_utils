"""
This module provides functionality to retrieve Hansard debate data from S3.
It includes methods for downloading parquet files, label stores, and XML debate files.
"""

import json
import logging
import os

from pathlib import Path
from typing import List

import pandas as pd

from botocore.exceptions import ClientError
from dotenv import load_dotenv

from discovery_utils.utils import s3


load_dotenv()
S3_BUCKET = os.getenv("S3_BUCKET")

logger = logging.getLogger(__name__)


class HansardGetter:
    """
    A class to handle retrieval of Hansard debate data from S3.
    """

    def __init__(self):
        """
        Initialize the HansardGetter with S3 bucket and prefix information.
        """

        self.bucket = S3_BUCKET
        self.prefix = "data/policy_scanning_data"
        self.s3_client = s3.s3_client()
        self.temp_dir = Path("tmp/debates")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.all_debates_retrieved = False

    def _get_s3_key(self, key: str) -> str:
        """Combine prefix and key for S3 operations."""
        return f"{self.prefix}/{key}"

    def get_debates_parquet(self) -> pd.DataFrame:
        """
        Download and return the Hansard debates parquet file from S3.

        Returns:
            pd.DataFrame: The Hansard debates data.

        Raises:
            ClientError: If there's an error downloading the file from S3.
        """
        key = self._get_s3_key("enriched/HansardDebates.parquet")
        logger.info(f"Downloading debates parquet file: {key}")
        try:
            return s3._download_obj(self.s3_client, self.bucket, key, download_as="dataframe")
        except ClientError as e:
            logger.error(f"Error downloading debates parquet file: {e}")
            raise

    def get_labelstore(self, keywords: bool = True) -> pd.DataFrame:
        """
        Download and return the Hansard debates label store from S3.
        If the labelstore doesn't exist, return an empty DataFrame.

        Args:
            keywords (bool): If True, download keyword labels. Otherwise, download ML labels.

        Returns:
            pd.DataFrame: The label store data.

        Raises:
            ClientError: If there's an error downloading the file from S3.
        """
        filename = "HansardDebates_LabelStore_keywords.csv" if keywords else "HansardDebates_LabelStore_ml.csv"
        key = self._get_s3_key(f"enriched/{filename}")
        logger.info(f"Attempting to download label store: {key}")
        try:
            return s3._download_obj(self.s3_client, self.bucket, key, download_as="dataframe")
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.warning(f"Labelstore {key} does not exist. Returning empty DataFrame.")
                return pd.DataFrame()
            else:
                logger.error(f"Error downloading label store: {e}")
                raise

    def delete_labelstore(self, keywords: bool = True) -> None:
        """
        Delete a labelstore from S3.

        Args:
            keywords (bool): If True, delete keyword labels. Otherwise, delete ML labels.
        """
        filename = "HansardDebates_LabelStore_keywords.csv" if keywords else "HansardDebates_LabelStore_ml.csv"
        key = self._get_s3_key(f"enriched/{filename}")
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=key)
            logger.info(f"Successfully deleted labelstore: {key}")
        except ClientError as e:
            logger.error(f"Error deleting labelstore {key}: {e}")
            raise

    def get_latest_debates(self) -> List[str]:
        """
        Download the latest XML debate files from S3 based on a synchronisation log.
        Latest files are files that were added on the most recent run of the data collection pipeline.

        Returns:
            List[str]: A list of paths to downloaded XML files.

        Raises:
            ClientError: If there's an error downloading the files from S3.
        """
        if self.all_debates_retrieved:
            logger.info("All debates have already been retrieved. No new files to process.")
            return []

        logger.info("Downloading new XML files since last sync")
        sync_file_key = self._get_s3_key("house_of_commons/files_to_sync.log")
        try:
            files_to_sync = (
                self.s3_client.get_object(Bucket=self.bucket, Key=sync_file_key)["Body"]
                .read()
                .decode("utf-8")
                .splitlines()
            )

            logger.info(f"Found {len(files_to_sync)} files to sync")

            downloaded_files = []
            for file in files_to_sync:
                local_file_path = self.temp_dir / os.path.basename(file)
                s3_key = self._get_s3_key(f"house_of_commons/{file}")
                logger.debug(f"Downloading XML file: {file} to {local_file_path}")
                self.s3_client.download_file(self.bucket, s3_key, str(local_file_path))
                downloaded_files.append(str(local_file_path))

            logger.info(f"Successfully downloaded {len(downloaded_files)} XML files")
            return downloaded_files
        except ClientError as e:
            logger.error(f"Error downloading latest debates: {e}")
            raise

    def get_all_debates(self) -> List[str]:
        """
        Download all XML debate files from S3.

        Returns:
            List[str]: A list of paths to downloaded XML files.

        Raises:
            ClientError: If there's an error downloading the files from S3.
        """
        logger.info("Retrieving all XML debate files from S3")
        prefix = self._get_s3_key("house_of_commons")
        try:
            all_files = s3._get_bucket_filenames(self.bucket, prefix)
            xml_files = [f for f in all_files if f.endswith(".xml")]

            downloaded_files = []
            for file in xml_files:
                local_file_path = self.temp_dir / os.path.basename(file)
                logger.debug(f"Downloading XML file: {file} to {local_file_path}")
                self.s3_client.download_file(self.bucket, file, str(local_file_path))
                downloaded_files.append(str(local_file_path))

            logger.info(f"Successfully downloaded {len(downloaded_files)} XML files")
            self.all_debates_retrieved = True
            return downloaded_files
        except ClientError as e:
            logger.error(f"Error downloading all debates: {e}")
            raise

    def get_people_metadata(self) -> dict:
        """
        Download and save the people metadata JSON file from S3.

        Returns:
            dict: The people metadata.

        Raises:
            ClientError: If there's an error downloading the file from S3.
        """
        logger.info("Downloading people metadata")
        key = self._get_s3_key("house_of_commons/people.json")
        try:
            metadata = s3._download_obj(self.s3_client, self.bucket, key, download_as="dict")

            temp_dir = Path("tmp")
            temp_dir.mkdir(exist_ok=True)
            with open(temp_dir / "people.json", "w") as fp:
                json.dump(metadata, fp)

            logger.info("Successfully downloaded and saved people metadata")
            return metadata
        except ClientError as e:
            logger.error(f"Error downloading people metadata: {e}")
            raise
