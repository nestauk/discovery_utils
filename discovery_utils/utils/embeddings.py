import os
import shutil

import lancedb
import pandas as pd

from lancedb import LanceDBConnection
from sentence_transformers import SentenceTransformer

from discovery_utils import PROJECT_DIR
from discovery_utils import logging
from discovery_utils.utils import s3


LOCAL_VECTOR_DB_PATH = "tmp/vector_db"


def add_embeddings(df: pd.DataFrame, text_col: str = "text", model_name: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    """
    Embeds text from a specified column in a DataFrame using a sentence transformer model.

    Adds the embedding and model name as new columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the text data.
        text_col (str): Column name of the DataFrame where the text is stored.
        model_name (str): Name of the sentence transformer model to use.

    Returns:
        pd.DataFrame: The original DataFrame with two new columns: 'embedding' and 'embedding_model'.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df[text_col].tolist(), show_progress_bar=True)
    df["embedding"] = list(embeddings)
    df["embedding_model"] = model_name
    return df


def load_lancedb_embeddings(
    embeddings: str,
) -> LanceDBConnection:
    """
    Load the lancedb embeddings

    Args:
        embeddings (str): Name of the embeddings to load
    """
    # Load the lanceDB
    download_lancedb_embeddings(embeddings)
    db = lancedb.connect(PROJECT_DIR / f"{LOCAL_VECTOR_DB_PATH}/{embeddings}")
    logging.info(f"Connected with database {embeddings}. Available tables: {db.table_names()}")
    return db


def download_lancedb_embeddings(
    embeddings: str,
    overwrite: bool = False,
    s3_path: str = "data/vector_db",
    local_path: str = LOCAL_VECTOR_DB_PATH,
) -> None:
    """
    Download and unzip lancedb embeddings

    Args:
        embeddings (str): Name of the embeddings to download
        overwrite (bool): Whether to overwrite existing files
        s3_path (str): Path in S3 where the embeddings are stored
        local_path (str): Local path to store the embeddings
    """
    # Check if folder already exists
    _local_path = f"{local_path}/{embeddings}/"

    if (not overwrite) and (os.path.exists(_local_path)):
        logging.info(f"Folder {_local_path} already exists. Set overwrite=True to download again.")
    else:
        s3_client = s3.s3_client()
        s3_key = f"{s3_path}/{embeddings}.zip"
        local_key = PROJECT_DIR / f"{local_path}/{embeddings}.zip"

        os.makedirs(_local_path, exist_ok=True)

        try:
            s3_client.download_file(os.environ["S3_BUCKET"], s3_key, local_key)
            logging.info(f"Downloaded {s3_key} to {local_key}")
        except Exception as e:
            logging.error(f"Error downloading {s3_key}: {str(e)}")
            raise

        shutil.unpack_archive(local_key, _local_path)
        logging.info(f"Unzipped {local_key} to {_local_path}")
