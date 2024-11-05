import os
import shutil

from pathlib import Path

import lancedb
import pandas as pd

from lancedb import LanceDBConnection
from sentence_transformers import SentenceTransformer

from discovery_utils import PROJECT_DIR
from discovery_utils import logging
from discovery_utils.utils import s3


LOCAL_VECTOR_DB_PATH = PROJECT_DIR / "tmp/vector_db"


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


def load_lancedb_embeddings(embeddings: str, local_path: str = LOCAL_VECTOR_DB_PATH) -> LanceDBConnection:
    """
    Load the lancedb embeddings

    Args:
        embeddings (str): Name of the embeddings to load
    """
    # Load the lanceDB
    if local_path is None:
        raise ValueError("Vector database path is not set")
    download_lancedb_embeddings(embeddings, overwrite=False, local_path=local_path)
    db = lancedb.connect(f"{local_path}/{embeddings}")
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
        local_key = f"{local_path}/{embeddings}.zip"

        Path(_local_path).mkdir(parents=True, exist_ok=True)

        try:
            s3_client.download_file(os.environ["S3_BUCKET"], s3_key, local_key)
            logging.info(f"Downloaded {s3_key} to {local_key}")
        except Exception as e:
            logging.error(f"Error downloading {s3_key}: {str(e)}")
            raise

        shutil.unpack_archive(local_key, str(_local_path))
        logging.info(f"Unzipped {local_key} to {_local_path}")


class VectorDB:
    """Class to handle the LanceDB connection and search"""

    def __init__(self, db_path: Path, db_name: str, table_name: str, model: str) -> None:
        """Initialise the VectorDB class

        Args:
            vector_db_path (Path): Path to the local LanceDB embeddings
        """
        self._vector_db_path = db_path
        self._vector_db_name = db_name
        self._vector_db_table_name = table_name
        self._vector_model_name = model
        self._vector_model = None
        self._vector_db_connection = None
        self._vector_db = None

    @property
    def vector_db(self) -> LanceDBConnection:
        """Get the LanceDB connection"""
        if self._vector_db is None:
            self._vector_db_connection = load_lancedb_embeddings(self._vector_db_name, local_path=self._vector_db_path)
            self._vector_db = self._vector_db_connection.open_table(self._vector_db_table_name)
            # Enable full text searches
            try:
                self._vector_db.create_fts_index("text")
            except Exception as e:
                logging.error(f"Error creating FTS index: {str(e)}")
        return self._vector_db

    @property
    def vector_model(self) -> SentenceTransformer:
        """Get the sentence transformer model"""
        if self._vector_model is None:
            self._vector_model = SentenceTransformer(self._vector_model_name)
        return self._vector_model

    def text_search(self, query: str, n_results: int = 10) -> pd.DataFrame:
        """Search the LanceDB for the query"""
        return self.vector_db.search(query, query_type="fts").select(["id", "text"]).limit(n_results).to_pandas()

    def vector_search(self, query: str, n_results: int = 10) -> pd.DataFrame:
        """Search the LanceDB for the query"""
        query_embedding = self.vector_model.encode([query])[0]
        return self.vector_db.search(query_embedding).select(["id", "text"]).limit(n_results).to_pandas()
