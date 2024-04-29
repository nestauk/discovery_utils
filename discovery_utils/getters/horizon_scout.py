import numpy as np
import pandas as pd

from discovery_utils.horizon_scout.make_training_data import DataPaths
from discovery_utils.utils import s3


client = s3.s3_client()


def convert_to_array(embedding_str: str) -> np.ndarray:
    """
    Convert string representation of a NumPy array into an actual NumPy array.

    Args:
        embedding_str (str): String representation of a NumPy array.

    Returns:
        np.ndarray: The converted NumPy array.
    """
    return np.fromstring(embedding_str.strip("[]"), sep=" ")


def get_training_data(mission: str) -> pd.DataFrame:
    """Load training data from S3 and convert embedding strings to NumPy arrays

    Args:
        mission (str): Nesta mission ('AHL', 'AFS' or 'ASF')

    Returns:
        pd.DataFrame: Training dataset with embeddings converted to NumPy arrays.
    """
    # Download the DataFrame from S3
    training_data = s3._download_obj(
        s3_client=client,
        bucket=s3.BUCKET_NAME_RAW,
        path_from=DataPaths.paths[f"{mission}_TRAIN"],
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )

    # Convert embeddings from string to NumPy array if they are stored as strings
    if "embedding" in training_data.columns and isinstance(training_data.iloc[0]["embedding"], str):
        training_data["embedding"] = training_data["embedding"].apply(convert_to_array)

    return training_data
