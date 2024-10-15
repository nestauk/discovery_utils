"""
Load Crunchbase data

Initialise lancedb
Create the embeddings
Upload to S3
*
Separate utils for loading the embeddings from S3
"""
import argparse
import os
import shutil

import lancedb
import pandas as pd
import torch

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel
from lancedb.pydantic import Vector
from pydantic import ValidationError

from discovery_utils import PROJECT_DIR
from discovery_utils import logging
from discovery_utils.getters import crunchbase
from discovery_utils.utils import s3


logger = logging.getLogger("crunchbase_embeddings")

BATCH_SIZE = 1000
MODEL = "all-MiniLM-L6-v2"
DB_PATH = PROJECT_DIR / "tmp/vector_db/crunchbase-lancedb"
DB_TABLE_NAME = "company_embeddings"

S3_BUCKET = os.environ["S3_BUCKET"]
S3_PATH = "data/vector_db/"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.set_defaults(test=False, dummy_data=False)
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--dummy_data", action="store_true", help="Use dummy data")
    args = parser.parse_args()

    test = args.test
    dummy_data = args.dummy_data
    if dummy_data:
        test = True

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Using CPU.")
        device = "cpu"
    else:
        device = "cuda"

    s3_client = s3.s3_client()
    if test:
        s3_key = f"{S3_PATH}{DB_PATH.name}_test.zip"
    else:
        s3_key = f"{S3_PATH}{DB_PATH.name}.zip"

    # Initialise the database
    DB_PATH.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(DB_PATH)
    logger.info(f"Initialised LanceDB at {DB_PATH}")

    model = get_registry().get("sentence-transformers").create(name=MODEL, device=device)
    logger.info(f"Using embeddings model {MODEL}")

    # Define the LanceDB model schema
    class Embedding(LanceModel):
        """Schema for Crunchbase company embeddings"""

        id: str
        name: str
        homepage_url: str
        country_code: str
        region: str
        city: str
        status: str
        employee_count: str
        text: str = model.SourceField()  # the text to embed
        vector: Vector(model.ndims()) = model.VectorField()  # the vector embedding

    # Create a new table in LanceDB with the defined schema
    table = db.create_table(DB_TABLE_NAME, schema=Embedding, mode="overwrite")

    # Load the data
    CB = crunchbase.CrunchbaseGetter()

    # Combine all text fields and fetch other useful categorical fields
    if dummy_data:
        # Use dummy data
        logger.info("Running in test mode: Using dummy data")
        texts_df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "name": ["Company 1", "Company 2", "Company 3"],
                "short_description": [
                    "Company 1 short description",
                    "Company 2 short description",
                    "Company 3 short description",
                ],
                "homepage_url": ["https://company1.com", "https://company2.com", "https://company3.com"],
                "country_code": ["US", "GB", "DE"],
                "region": ["North America", "Europe", "Europe"],
                "city": ["New York", "London", "Berlin"],
                "status": ["operating", "closed", "ipo"],
                "employee_count": ["1000", "500", "2000"],
                "text": ["Company 1 description", "Company 2 description", "Company 3 description"],
            }
        )
    else:
        texts_df = (
            CB.organisations[
                [
                    "id",
                    "name",
                    "short_description",
                    "homepage_url",
                    "country_code",
                    "region",
                    "city",
                    "status",
                    "employee_count",
                ]
            ]
            .merge(CB.descriptions[["id", "description"]], on="id", how="left")
            .fillna({"description": "", "short_description": "", "name": ""})
            .astype(str)
            .assign(text=lambda x: x["name"] + " " + x["short_description"] + " " + x["description"])
            .drop(columns=["description"])
        )
    # Remove companies with no descriptions
    texts_df = texts_df[texts_df["text"].str.strip() != ""]

    if test:
        n_test = min(100, len(texts_df))
        texts_df = texts_df.sample(n_test)
        logger.info(f"Running in test mode: sampling {n_test} records")

    data_to_add = texts_df.to_dict(orient="records")
    logger.info(f"Adding {len(data_to_add)} records to the database")
    for i in range(0, len(data_to_add), BATCH_SIZE):
        batch = data_to_add[i : i + BATCH_SIZE]
        try:
            table.add(batch)
            logger.info(f"Added batch {i // BATCH_SIZE + 1} / {len(data_to_add) // BATCH_SIZE}")
        except ValidationError as e:
            logger.error(f"Error adding batch {i // BATCH_SIZE + 1}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in batch {i // BATCH_SIZE + 1}: {e}")

    compressed_db_path = f"{DB_PATH}.zip"
    shutil.make_archive(DB_PATH, "zip", DB_PATH)
    logger.info(f"Compressed LanceDB to {compressed_db_path}")

    try:
        logger.info(f"Uploading {compressed_db_path} to S3 bucket {S3_BUCKET}")
        s3_client.upload_file(compressed_db_path, S3_BUCKET, s3_key)
        logger.info(f"Successfully uploaded {compressed_db_path} to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        logger.error(f"Error uploading database to S3: {e}")
