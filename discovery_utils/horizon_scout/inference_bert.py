from typing import List

import pandas as pd
import torch

from botocore.exceptions import ClientError
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from discovery_utils import logging
from discovery_utils.getters.horizon_scout import get_bert_model
from discovery_utils.horizon_scout.utils import get_current_datetime
from discovery_utils.utils import s3


logging.basicConfig(level=logging.INFO)

client = s3.s3_client()


def bert_add_predictions(
    df: pd.DataFrame,
    text_col: str,
    bert_models: List[PreTrainedModel],
    tokenizer: PreTrainedTokenizer,
    missions: List[str],
    batch_size: int,
) -> pd.DataFrame:
    """Add BERT predictions to input DataFrame containing text to classify as relevant to Nesta's missions or not.

    Processes in batches, classifying text using a BERT model for each mission.

    Args:
        df (pd.DataFrame): DataFrame containing the text to be classified.
        text_col (str): Name of the column containing text data.
        bert_models (List[PreTrainedModel]): List of BERT models.
        tokenizer (PreTrainedTokenizer): BERT tokenizer.
        missions (List[str]): List of mission names.
        batch_size (int): Size of the batch for processing data.

    Returns:
        pd.DataFrame: DataFrame with new columns containing classification predictions.
    """
    df = df.dropna(subset=text_col)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_models = [model.to(device) for model in bert_models]
    total_batches = (len(df) + batch_size - 1) // batch_size

    predictions = {f"{mission.lower()}_bert_preds": [] for mission in missions}

    for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Processing batches"):
        batch_texts = df[text_col].iloc[i : i + batch_size].tolist()
        tokenized = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
            device
        )

        for bert_model, mission in zip(bert_models, missions):
            with torch.no_grad():
                outputs = bert_model(**tokenized)
                pred = outputs.logits.argmax(dim=1).cpu().numpy()
            predictions[f"{mission.lower()}_bert_preds"].extend(pred)

    for mission in missions:
        df[f"{mission.lower()}_bert_preds"] = pd.Series(predictions[f"{mission.lower()}_bert_preds"])

    return df


def preds_to_labels(df: pd.DataFrame, missions: List[str]) -> pd.DataFrame:
    """
    Create a new 'label' column based on the values of 'ahl_bert_preds', 'asf_bert_preds', and 'afs_bert_preds'.

    Args:
        df (pd.DataFrame): Input DataFrame with 'ahl_bert_preds',
            'asf_bert_preds', and 'afs_bert_preds' columns.
        missions (List[str]): List of mission names.

    Returns:
        pd.DataFrame: DataFrame with an additional 'label' column.
    """

    def make_label(row: pd.Series) -> str:
        return ",".join([mission for mission in missions if row[f"{mission.lower()}_bert_preds"] == 1])

    df["label"] = df.apply(make_label, axis=1)
    return df


def check_file_exists(path: str) -> bool:
    """Check if a file exists in an S3 bucket.

    Args:
        path (str): Path of the file within the bucket.

    Returns:
        bool: True if the file exists, False otherwise.
    """

    try:
        client.head_object(Bucket=s3.BUCKET_NAME_RAW, Key=path)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise e


MISSIONS = ["AHL", "ASF", "AFS"]
MODEL_PATHS = [
    "AHL_bert_model_0.896_20240618-2156.pkl",
    "ASF_bert_model_0.925_20240618-2146.pkl",
    "AFS_bert_model_0.971_20240618-2152.pkl",
]
BATCH_SIZE = 512

LABEL_STORE_PATH = "data/crunchbase/enriched/label_store.csv"
NEW_ORGS_PATH = "data/crunchbase/enriched/organizations_new_only.parquet"
FULL_ORGS_PATH = "data/crunchbase/enriched/organizations_full.parquet"

LABEL_TYPE = "BERT classifiers 20240618"
DATASET_NAME = "Crunchbase"

MODE = "Full"
MODE = "New"


if __name__ == "__main__":
    # Set data path
    if MODE == "Full":
        data_path = FULL_ORGS_PATH
    elif MODE == "New":
        data_path = NEW_ORGS_PATH

    # Load data
    logging.info("Downloading Crunchbase companies")
    companies = s3._download_obj(
        s3_client=client,
        bucket=s3.BUCKET_NAME_RAW,
        path_from=data_path,
        download_as="dataframe",
    )[["id", "short_description"]]

    # Load models
    bert_models = [get_bert_model(mission, model_path) for mission, model_path in zip(MISSIONS, MODEL_PATHS)]
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    # Perform inference
    logging.info("Adding BERT mission predictions")
    companies = bert_add_predictions(
        df=companies,
        text_col="short_description",
        bert_models=bert_models,
        tokenizer=tokenizer,
        missions=MISSIONS,
        batch_size=BATCH_SIZE,
    )

    # Convert binary predictions to labels
    companies = preds_to_labels(companies, MISSIONS)

    # Add label type col
    companies["label_type"] = LABEL_TYPE

    # Add label date col
    current_datetime = get_current_datetime()
    companies["label_date"] = current_datetime

    # Add dataset col
    companies["dataset"] = DATASET_NAME

    # Append to datastore, if file does not exist then create
    if check_file_exists(LABEL_STORE_PATH):
        label_store = s3._download_obj(
            s3_client=client,
            bucket=s3.BUCKET_NAME_RAW,
            path_from=LABEL_STORE_PATH,
            download_as="dataframe",
        )
        label_store = pd.concat([label_store, companies], ignore_index=True).reset_index(drop=True)
        s3.upload_obj(label_store, s3.BUCKET_NAME_RAW, LABEL_STORE_PATH, kwargs_writing={"index": False})
    else:
        s3.upload_obj(companies, s3.BUCKET_NAME_RAW, LABEL_STORE_PATH, kwargs_writing={"index": False})
