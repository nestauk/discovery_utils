from typing import List

import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from discovery_utils import logging
from discovery_utils.getters.horizon_scout import get_bert_model
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
    total_batches = (len(df) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Processing batches"):
        batch_texts = df[text_col].iloc[i : i + batch_size].tolist()
        tokenized = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

        batch_indices = range(i, min(i + batch_size, len(df)))

        for bert_model, mission in zip(bert_models, missions):
            col_name = f"{mission.lower()}_bert_preds"

            # Ensure model and tokenized tensors are on the same device
            if torch.cuda.is_available():
                bert_model = bert_model.to("cuda")
                tokenized = {key: value.to("cuda") for key, value in tokenized.items()}

            # Make predictions
            with torch.no_grad():
                outputs = bert_model(**tokenized)
                predictions = outputs.logits.argmax(dim=1).cpu().numpy()

            df.loc[batch_indices, col_name] = predictions

    return df


SAVE_PATH = "data/crunchbase/enriched/organizations_new_only_with_bert_preds.csv"
MISSIONS = ["AHL", "ASF", "AFS"]
MODEL_PATHS = [
    "AHL_bert_model_0.967_20240601-1441.pkl",
    "ASF_bert_model_0.93_20240601-1434.pkl",
    "AFS_bert_model_0.979_20240601-1438.pkl",
]
BATCH_SIZE = 2048


if __name__ == "__main__":
    logging.info("Downloading new Crunchbase companies")
    new_companies = s3._download_obj(
        s3_client=client,
        bucket=s3.BUCKET_NAME_RAW,
        path_from="data/crunchbase/enriched/organizations_new_only.parquet",
        download_as="dataframe",
    )

    bert_models = [get_bert_model(mission, model_path) for mission, model_path in zip(MISSIONS, MODEL_PATHS)]
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    logging.info("Adding BERT mission predictions")
    new_companies_with_predictions = bert_add_predictions(
        df=new_companies,
        text_col="short_description",
        bert_models=bert_models,
        tokenizer=tokenizer,
        missions=MISSIONS,
        batch_size=BATCH_SIZE,
    )

    logging.info("Saving new Crunchbase companies with BERT mission predictions to %s", SAVE_PATH)
    s3.upload_obj(
        new_companies_with_predictions,
        s3.BUCKET_NAME_RAW,
        SAVE_PATH,
    )
