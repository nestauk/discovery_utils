import io
import pickle  # nosec B403

import numpy as np
import pandas as pd
import torch

from sklearn.svm import SVC
from transformers import PreTrainedModel

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


def get_svc_model(mission: str, svc_model_filename: str) -> SVC:
    """Load SCV from S3

    Args:
        mission (str): Nesta mission ('AHL', 'AFS' or 'ASF')
        model_filename (str): Filename on S3 e.g. 'AHL_svc_0.882_20240517-1602.pkl'

    Returns:
        SVC: Support Vector Classifier for specified mission and filename
    """
    return s3._download_obj(
        s3_client=client,
        bucket=s3.BUCKET_NAME_RAW,
        path_from=f"models/horizon_scout/{mission}/{svc_model_filename}",
        download_as=None,
    )


class CPU_Unpickler(pickle.Unpickler):
    """Unpickler that loads torch tensors on CPU

    Example:
        model = CPU_Unpickler(fileobj).load()
    """

    def find_class(self, module, name):  # noqa: ANN001, ANN201
        """Load torch tensors on CPU"""
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def get_bert_model(mission: str, bert_model_filename: str, gpu: bool = True) -> PreTrainedModel:
    """Load BERT model from S3

    Args:
        mission (str): Nesta mission ('AHL', 'AFS' or 'ASF')
        model_filename (str): Filename on S3 e.g. 'AHL_bert_model_0.965_20240530-2244.pkl'
        gpu (bool): Whether to load the model on GPU

    Returns:
        PreTrainedModel: BERT model
    """
    path_from = f"models/horizon_scout/{mission}/{bert_model_filename}"
    if gpu:
        return s3._download_obj(
            s3_client=client,
            bucket=s3.BUCKET_NAME_RAW,
            path_from=path_from,
            download_as=None,
        )
    else:
        fileobj = io.BytesIO()
        client.download_fileobj(s3.BUCKET_NAME_RAW, path_from, fileobj)
        fileobj.seek(0)
        return CPU_Unpickler(fileobj).load()
