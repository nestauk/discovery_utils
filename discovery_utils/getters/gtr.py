"""
discovery_utils.getters.gtr.py

Getters for Gateway to Research data
"""
import logging
import os
import re

import pandas as pd

from discovery_utils.utils import s3


S3_BUCKET = os.environ["S3_BUCKET"]
S3_PREFIX = "data/GtR/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GtrGetter:
    """Class to get Gateway to Research data from S3"""

    def __init__(self, use_latest_version: bool = True, data_version: str = None) -> None:
        """Initialise GtrGetter

        Args:
            use_latest_version (bool, optional): Use the latest version of the data. Defaults to True.
            data_version (str, optional): Version of data to use, should follow format "Crunchbase_YYYY-MM-DD".
        """
        self.s3_client = s3.s3_client()
        self.bucket = S3_BUCKET
        self.s3_prefix = S3_PREFIX
        self.use_latest_version = use_latest_version
        if (data_version is None) and self.use_latest_version:
            self.data_version = self._get_latest_data_version()
        else:
            self.data_version = data_version
        self._projects = None
        self._organisations = None
        self._persons = None
        self._funds = None

    def _get_latest_data_version(self) -> str:
        """Find the latest version based on S3 folder timestamps."""
        try:
            logger.info(f"Checking for latest version of data in S3 bucket: {self.bucket}")
            response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=S3_PREFIX, Delimiter="/")
            folders = [content["Prefix"] for content in response.get("CommonPrefixes", [])]

            # Extract the date part from the folder names using regex
            date_folders = []
            date_pattern = re.compile(r"GtR_(\d{4}\d{2}\d{2})")

            for folder in folders:
                match = date_pattern.search(folder)
                if match:
                    date_folders.append(match.group(1))  # Extract just the date part

            # Sort the folders by date and get the latest one
            if date_folders:
                latest_date = sorted(date_folders)[-1]
                latest_version = f"GtR_{latest_date}"
                logger.info(f"Latest version found: {latest_version}")
                return latest_version
            else:
                raise ValueError("No valid folders found.")
        except Exception as e:
            logger.error(f"Error fetching or processing folder names from S3: {str(e)}")
            raise

    def _get_table(self, key: str) -> pd.DataFrame:
        """Download parquet table from S3"""
        logger.info(f"Downloading parquet file: {key}")
        try:
            response = s3._download_obj(
                self.s3_client,
                self.bucket,
                key,
                download_as="dataframe",
            )
            logger.info(f"Successfully downloaded and read parquet file: {key}")
            return response
        except Exception as e:
            logger.error(f"Error downloading parquet file {key}: {str(e)}")
            raise

    def _get_gtr_table(self, table: str) -> pd.DataFrame:
        """Get Gateway to Research data from S3

        Args:
            table (str): Table name to download

        Returns:
            pd.DataFrame
        """
        key = f"{S3_PREFIX}{self.data_version}/{table}.parquet"
        return self._get_table(key)

    @property
    def projects(self) -> pd.DataFrame:
        """Get projects data"""
        if self._projects is None:
            self._projects = self._get_gtr_table("projects")
        return self._projects

    @property
    def organisations(self) -> pd.DataFrame:
        """Get organisations data"""
        if self._organisations is None:
            self._organisations = self._get_gtr_table("organisations")
        return self._organisations

    @property
    def persons(self) -> pd.DataFrame:
        """Get persons data"""
        if self._persons is None:
            self._persons = self._get_gtr_table("persons")
        return self._persons

    @property
    def funds(self) -> pd.DataFrame:
        """Get funds data"""
        if self._funds is None:
            self._funds = self._get_gtr_table("funds")
        return self._funds
