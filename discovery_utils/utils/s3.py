import io
import json
import os
import pickle  # nosec
import warnings

from fnmatch import fnmatch
from pathlib import Path
from typing import List

import boto3
import dotenv
import numpy as np
import pandas as pd

from botocore.client import BaseClient


dotenv.load_dotenv()

# Retrieve AWS file information from environment variables
BUCKET_NAME_RAW = os.getenv("BUCKET_NAME_RAW")
# Retrieve AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")


def s3_client(aws_access_key_id: str = AWS_ACCESS_KEY, aws_secret_access_key: str = AWS_SECRET_KEY) -> BaseClient:
    """Initialize and return an S3 client"""
    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def _get_bucket_filenames(bucket_name: str, dir_name: str = "") -> List[str]:
    """Get a list of all files in bucket directory.

    Taken from Nesta DS Utils.

    Args:
        bucket_name (str): S3 bucket name
        dir_name (str, optional): Directory or sub-directory within an S3 bucket.
            Defaults to '' (the top level bucket directory).

    Returns:
        List[str]: List of file names in bucket directory.
    """
    s3_resources = boto3.resource("s3")
    bucket = s3_resources.Bucket(bucket_name)
    return [object_summary.key for object_summary in bucket.objects.filter(Prefix=dir_name)]


def _match_one_file(all_files: list, file_name: str, dir_path: str) -> str:
    matched_files = [f for f in all_files if Path(f).stem == file_name]

    if not matched_files:
        raise FileNotFoundError(f"No file exactly matching '{file_name}' found in {dir_path}")
    elif len(matched_files) > 1:
        raise ValueError(
            f"Multiple files exactly matching '{file_name}' found in {dir_path}, requiring clarification."
        )

    return matched_files[0]


def _list_directories(s3_client: BaseClient, bucket: str, prefix: str) -> list:
    """List S3 directories under the specified prefix."""
    # List all objects with the specified prefix
    objects = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")

    # Extract the directory names from the CommonPrefixes list
    directories = [prefix["Prefix"] for prefix in objects.get("CommonPrefixes", [])]

    return directories


def _fileobj_to_df(fileobj: io.BytesIO, path_from: str, **kwargs) -> pd.DataFrame:
    """Convert bytes file object into dataframe.

    Args:
        fileobj (io.BytesIO): Bytes file object.
        path_from (str): Path of loaded data.

    Returns:
        pd.DataFrame: Data as dataframe.
    """
    if fnmatch(path_from, "*.csv"):
        return pd.read_csv(fileobj, **kwargs)
    elif fnmatch(path_from, "*.parquet"):
        return pd.read_parquet(fileobj, **kwargs)


def _fileobj_to_dict(fileobj: io.BytesIO, **kwargs) -> dict:
    """Convert bytes file object into dictionary.

    Args:
        fileobj (io.BytesIO): Bytes file object.
        path_from (str): Path of loaded data.

    Returns:
        dict: Data as dictionary.
    """
    return json.loads(fileobj.getvalue().decode(), **kwargs)


def _fileobj_to_list(fileobj: io.BytesIO, path_from: str, **kwargs) -> list:
    """Convert bytes file object into list.

    Args:
        fileobj (io.BytesIO): Bytes file object.
        path_from (str): Path of loaded data.

    Returns:
        list: Data as list.
    """
    if fnmatch(path_from, "*.csv"):
        list_data = pd.read_csv(fileobj, **kwargs)["0"].to_list()
    elif fnmatch(path_from, "*.txt"):
        list_data = fileobj.read().decode().splitlines()
    elif fnmatch(path_from, "*.json"):
        list_data = json.loads(fileobj.getvalue().decode())

    return list_data


def _fileobj_to_str(fileobj: io.BytesIO) -> str:
    """Convert bytes file object into string.

    Args:
        fileobj (io.BytesIO): Bytes file object.
        path_from (str): Path of loaded data.

    Returns:
        str: Data as string.
    """
    return fileobj.getvalue().decode("utf-8")


def _fileobj_to_np_array(fileobj: io.BytesIO, path_from: str, **kwargs) -> np.ndarray:
    """Convert bytes file object into numpy array.

    Args:
        fileobj (io.BytesIO): Bytes file object.
        path_from (str): Path of loaded data.

    Returns:
        np.ndarray: Data as numpy array.
    """
    if fnmatch(path_from, "*.csv"):
        np_array_data = np.genfromtxt(fileobj, delimiter=",", **kwargs)

    return np_array_data


def _download_obj(
    s3_client: BaseClient,
    bucket: str,
    path_from: str,
    download_as: str = None,
    kwargs_boto: dict = None,
    kwargs_reading: dict = None,
) -> any:  # noqa: B006
    """Download data to memory from S3 location.

    Adapted from Nesta DS Utils

    Args:
        s3_client (BaseClient): The S3 client.
        bucket (str): Bucket's name.
        path_from (str): Path to data in S3.
        download_as (str, optional): Type of object downloading. Choose between
        ('dataframe', 'geodf', 'dict', 'list', 'str', 'np.array'). Not needed for 'pkl files'.
        kwargs_boto (dict, optional): Dictionary of kwargs for boto3 function 'download_fileobj'.
            Default is None, which results in an empty dict.
        kwargs_reading (dict, optional): Dictionary of kwargs for reading data.
            Default is None, which results in an empty dict.

    Returns:
        any: Downloaded data.
    """
    if kwargs_boto is None:
        kwargs_boto = {}
    if kwargs_reading is None:
        kwargs_reading = {}

    if not path_from.endswith(
        (
            ".csv",
            ".parquet",
            ".json",
            ".txt",
            ".pkl",
            ".geojson",
            ".xlsx",
            ".xlsm",
        )
    ):
        raise NotImplementedError("This file type is not currently supported for download in memory.")
    fileobj = io.BytesIO()
    s3_client.download_fileobj(bucket, path_from, fileobj, **kwargs_boto)
    fileobj.seek(0)
    if not download_as:
        if path_from.endswith((".pkl",)):
            return pickle.load(fileobj, **kwargs_reading)  # nosec
        else:
            raise ValueError("'download_as' is required for this file type.")
    elif download_as == "dataframe":
        if path_from.endswith((".csv", ".parquet", ".xlsx", ".xlsm")):
            return _fileobj_to_df(fileobj, path_from, **kwargs_reading)
        else:
            raise NotImplementedError(
                "Download as dataframe currently supported only " "for 'csv','parquet','xlsx' and 'xlsm'."
            )
    elif download_as == "dict":
        if path_from.endswith((".json",)):
            return _fileobj_to_dict(fileobj, path_from, **kwargs_reading)
        elif path_from.endswith((".geojson",)):
            warnings.warn(
                "Please check geojson has a member with the name 'type', the value of the member must be one of the following:"
                "'Point', 'MultiPoint', 'LineString', 'MultiLineString', 'Polygon', 'MultiPolygon', 'GeometryCollection',"
                "'Feature' and 'FeatureCollection'. Else downloaded dictionary will not be valid geojson."
            )
            return _fileobj_to_dict(fileobj, path_from, **kwargs_reading)
        else:
            raise NotImplementedError("Download as dictionary currently supported only " "for 'json' and 'geojson'.")
    elif download_as == "list":
        if path_from.endswith((".csv", ".txt", ".json")):
            return _fileobj_to_list(fileobj, path_from, **kwargs_reading)
        else:
            raise NotImplementedError("Download as list currently supported only " "for 'csv', 'txt' and 'json'.")
    elif download_as == "str":
        if path_from.endswith((".txt",)):
            return _fileobj_to_str(fileobj)
        else:
            raise NotImplementedError("Download as string currently supported only " "for 'txt'.")
    else:
        raise ValueError(
            "'download_as' not provided. Choose between ('dataframe', 'geodf', "
            "'dict', 'list', 'str', 'np.array'). Not needed for 'pkl files'.'"
        )
