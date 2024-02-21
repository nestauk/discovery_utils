import logging
import os

from datetime import datetime
from datetime import timedelta
from io import BytesIO
from unittest.mock import patch

import boto3
import pandas as pd
import pytest

from moto import mock_s3

from src.getters import crunchbase as cb


# Set logging
logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"  # nosec: B105
    os.environ["AWS_SECURITY_TOKEN"] = "testing"  # nosec: B105
    os.environ["AWS_SESSION_TOKEN"] = "testing"  # nosec: B105


@pytest.fixture(scope="function")
def mock_s3_env(aws_credentials):
    """Create a mock S3 bucket and upload dummy data."""
    with mock_s3():
        conn = boto3.client("s3", region_name="us-east-1")
        conn.create_bucket(Bucket=os.environ["BUCKET_NAME_RAW"])

        # Import dummy data and read the CSV file into a DataFrame
        csv_file_path = "tests/dummydata.csv"
        csv_data = pd.read_csv(csv_file_path)

        # Convert DataFrame to CSV content in bytes
        csv_content = csv_data.to_csv(index=False).encode("utf-8")

        # Convert DataFrame to Parquet format in memory using a BytesIO buffer
        parquet_buffer = BytesIO()
        csv_data.to_parquet(parquet_buffer, index=False)
        parquet_content = parquet_buffer.getvalue()  # Get the binary content

        # Define snapshot dates and their corresponding file extensions
        snapshots = {"2024-01-30": ".csv", "2024-01-31": ".parquet", "2024-02-05": ".csv", "2024-02-06": ".parquet"}

        # Define the base file names without extensions
        base_file_names = [
            "acquisitions",
            "category_groups",
            "checksum",
            "degrees",
            "event_appearances",
            "events",
            "funding_rounds",
            "funds",
            "investment_partners",
            "investments",
            "investors",
            "ipos",
            "jobs",
            "org_parents",
            "organization_descriptions",
            "organizations",
            "people_descriptions",
            "people",
        ]

        # Loop through snapshots and base file names, creating files with the appropriate content
        for date, extension in snapshots.items():
            for base_file_name in base_file_names:
                file_name = f"{base_file_name}{extension}"  # Append the correct extension
                s3_key = f"{os.environ['S3_PATH_RAW']}Crunchbase_{date}/{file_name}"
                if extension == ".csv":
                    conn.put_object(Bucket=os.environ["BUCKET_NAME_RAW"], Key=s3_key, Body=csv_content)
                    logging.info(f"Created {s3_key}")
                elif extension == ".parquet":
                    conn.put_object(Bucket=os.environ["BUCKET_NAME_RAW"], Key=s3_key, Body=parquet_content)
                    logging.info(f"Created {s3_key}")

        yield


def test_get_table_valid_file(mock_s3_env):
    # Arrange
    timestamp = "2024-01-31"
    file_name = "organizations"  # No extension needed
    s3_client = boto3.client("s3", region_name="us-east-1")

    # Act
    result = cb.get_table(timestamp, file_name, s3_client)

    # Assert
    assert isinstance(result, pd.DataFrame)
    # Additional assertions based on the expected DataFrame structure


def test_get_table_invalid_file(mock_s3_env):
    # Arrange
    timestamp = "2024-01-31"
    file_name = "organization"  # This file name does not exist in FILE_NAMES_RAW
    s3_client = boto3.client("s3", region_name="us-east-1")

    # Act and Assert
    with pytest.raises(ValueError, match=f"File '{file_name}' not found in FILE_NAMES_RAW."):
        cb.get_table(timestamp, file_name, s3_client)


# Mock for get_table function to avoid actual S3 interaction
@patch("src.getters.crunchbase.get_table")
@mock_s3
def test_get_tdy_table(mock_get_table):
    # Arrange
    mock_s3_client = boto3.client("s3", region_name="us-east-1")
    file_name = "organizations"
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Act
    cb.get_tdy_table(file_name, mock_s3_client)

    # Assert
    mock_get_table.assert_called_once_with(today_str, file_name, mock_s3_client)


@patch("src.getters.crunchbase.get_table")
@mock_s3
def test_get_ytd_table(mock_get_table):
    # Arrange
    mock_s3_client = boto3.client("s3", region_name="us-east-1")
    file_name = "organizations"
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Act
    cb.get_ytd_table(file_name, mock_s3_client)

    # Assert
    mock_get_table.assert_called_once_with(yesterday_str, file_name, mock_s3_client)
