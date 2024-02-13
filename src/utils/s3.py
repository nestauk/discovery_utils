import boto3
from botocore.client import BaseClient
import io
import pickle
from fnmatch import fnmatch
import warnings
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path


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
    return [
        object_summary.key for object_summary in bucket.objects.filter(Prefix=dir_name)
    ]
    
def _match_one_file(all_files: list, file_name: str, dir_path: str) -> str:
    matched_files = [f for f in all_files if Path(f).stem == file_name]
    
    if not matched_files:
        raise FileNotFoundError(f"No file exactly matching '{file_name}' found in {dir_path}")
    elif len(matched_files) > 1:
        raise ValueError(f"Multiple files exactly matching '{file_name}' found in {dir_path}, requiring clarification.")
    
    return matched_files[0]

def _list_directories(s3_client: BaseClient, bucket: str, prefix: str) -> list:
    """List S3 directories under the specified prefix."""
    # List all objects with the specified prefix
    objects = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')

    # Extract the directory names from the CommonPrefixes list
    directories = [prefix['Prefix'] for prefix in objects.get('CommonPrefixes', [])]

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
    elif fnmatch(path_from, "*.xlsx") or fnmatch(path_from, "*.xlsm"):
        if feature_enabled["excel"]:
            return pd.read_excel(fileobj, **kwargs)
        else:
            raise ModuleNotFoundError(
                "Please install 'io_extras' extra from nesta_ds_utils or 'openpyxl' to download excel files."
            )


def _fileobj_to_dict(fileobj: io.BytesIO, path_from: str, **kwargs) -> dict:
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
    elif fnmatch(path_from, "*.parquet"):
        np_array_data = pq.read_table(fileobj, **kwargs)["data"].to_numpy()

    return np_array_data


def _download_obj(
    s3_client: BaseClient,
    bucket: str,
    path_from: str,
    download_as: str = None,
    kwargs_boto: dict = {},
    kwargs_reading: dict = {},
) -> any:
    """Download data to memory from S3 location.
    Adapted from Nesta DS Utils: https://github.com/nestauk/nesta_ds_utils/blob/f71ae556f1bbe44b1dff4f1ec502ac0ba45371f5/nesta_ds_utils/loading_saving/S3.py#L331.

    Args:
        bucket (str): Bucket's name.
        path_from (str): Path to data in S3.
        download_as (str, optional): Type of object downloading. Choose between
        ('dataframe', 'geodf', 'dict', 'list', 'str', 'np.array'). Not needed for 'pkl files'.
        kwargs_boto (dict, optional): Dictionary of kwargs for boto3 function 'download_fileobj'.
        kwargs_reading (dict, optional): Dictionary of kwargs for reading data.

    Returns:
        any: Downloaded data.
    """
    if not path_from.endswith(
        tuple(
            [".csv", ".parquet", ".json", ".txt", ".pkl", ".geojson", ".xlsx", ".xlsm"]
        )
    ):
        raise NotImplementedError(
            "This file type is not currently supported for download in memory."
        )
    fileobj = io.BytesIO()
    s3_client.download_fileobj(bucket, path_from, fileobj, **kwargs_boto)
    fileobj.seek(0)
    if not download_as:
        if path_from.endswith(tuple([".pkl"])):
            return pickle.load(fileobj, **kwargs_reading)
        else:
            raise ValueError("'download_as' is required for this file type.")
    elif download_as == "dataframe":
        if path_from.endswith(tuple([".csv", ".parquet", ".xlsx", ".xlsm"])):
            return _fileobj_to_df(fileobj, path_from, **kwargs_reading)
        else:
            raise NotImplementedError(
                "Download as dataframe currently supported only "
                "for 'csv','parquet','xlsx' and 'xlsm'."
            )
    elif download_as == "geodf":
        if path_from.endswith(tuple([".geojson"])):
            if _gis_enabled:
                return _fileobj_to_gdf(fileobj, path_from, **kwargs_reading)
            else:
                raise ModuleNotFoundError(
                    "Please install 'gis' extra from nesta_ds_utils or 'geopandas' to download geodataframes."
                )
        else:
            raise NotImplementedError(
                "Download as geodataframe currently supported only " "for 'geojson'."
            )
    elif download_as == "dict":
        if path_from.endswith(tuple([".json"])):
            return _fileobj_to_dict(fileobj, path_from, **kwargs_reading)
        elif path_from.endswith(tuple([".geojson"])):
            warnings.warn(
                "Please check geojson has a member with the name 'type', the value of the member must be one of the following:"
                "'Point', 'MultiPoint', 'LineString', 'MultiLineString', 'Polygon', 'MultiPolygon', 'GeometryCollection',"
                "'Feature' and 'FeatureCollection'. Else downloaded dictionary will not be valid geojson."
            )
            return _fileobj_to_dict(fileobj, path_from, **kwargs_reading)
        else:
            raise NotImplementedError(
                "Download as dictionary currently supported only "
                "for 'json' and 'geojson'."
            )
    elif download_as == "list":
        if path_from.endswith(tuple([".csv", ".txt", ".json"])):
            return _fileobj_to_list(fileobj, path_from, **kwargs_reading)
        else:
            raise NotImplementedError(
                "Download as list currently supported only "
                "for 'csv', 'txt' and 'json'."
            )
    elif download_as == "str":
        if path_from.endswith(tuple([".txt"])):
            return _fileobj_to_str(fileobj)
        else:
            raise NotImplementedError(
                "Download as string currently supported only " "for 'txt'."
            )
    elif download_as == "np.array":
        if path_from.endswith(tuple([".csv", ".parquet"])):
            return _fileobj_to_np_array(fileobj, path_from, **kwargs_reading)
        else:
            raise NotImplementedError(
                "Download as numpy array currently supported only "
                "for 'csv' and 'parquet'."
            )
    else:
        raise ValueError(
            "'download_as' not provided. Choose between ('dataframe', 'geodf', "
            "'dict', 'list', 'str', 'np.array'). Not needed for 'pkl files'.'"
        )