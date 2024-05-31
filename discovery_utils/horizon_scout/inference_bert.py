from typing import List

import pandas as pd

from discovery_utils import logging
from discovery_utils.getters.horizon_scout import get_bert_classifier
from discovery_utils.utils import s3


logging.basicConfig(level=logging.INFO)

client = s3.s3_client()


def bert_predictions(mission: str, model_filename: str, df: pd.DataFrame, text_col: str, label2id: dict) -> list:
    """Use BERT classifier to make a list of predictions

    Args:
       mission (str): Nesta mission ('AHL', 'AFS' or 'ASF').
       model_filename (str): Filename on S3 e.g. 'AHL_bert_0.965_20240530-2244.pkl'
       df (pd.DataFrame): Input DataFrame containing the text data.
       text_col (str): Column name for the text in the DataFrame.
       label2id (dict): Dict that maps label to id.

    Returns:
       list: List of predictions
    """
    # Load the classifier
    bert_classifier = get_bert_classifier(mission, model_filename)

    # Make predictions on the text data
    texts = df[text_col].tolist()
    predictions = bert_classifier(texts)

    # Convert predictions to 0s and 1s using label2id
    predicted_labels = [label2id[pred["label"]] for pred in predictions]

    return predicted_labels


def bert_add_predictions(
    missions: List[str],
    model_filenames: List[str],
    df: pd.DataFrame,
    text_col: str,
    label2id: dict,
    predictions_cols: List[str],
) -> pd.DataFrame:
    """Use BERT classifier to add predictions to input DataFrame

    Args:
       mission (str): Nesta mission ('AHL', 'AFS' or 'ASF').
       model_filename (str): Filename on S3 e.g. 'AHL_bert_0.965_20240530-2244.pkl'
       df (pd.DataFrame): Input DataFrame containing the text data.
       text_col (str): Column name for the text in the DataFrame.
       label2id (dict): Dict that maps label to id.
       predictions_col (str): Column name to assign the predictions to.

    Returns:
       pd.DataFrame: DataFrame with predictions from each mission classifier added.
    """
    for mission, model_filename, predictions_col in zip(missions, model_filenames, predictions_cols):
        df[predictions_col] = bert_predictions(mission, model_filename, df, text_col, label2id)
    return df


MISSIONS = [
    "AHL",
    "ASF",
    "AFS",
]
MODEL_FILENAMES = [
    "AHL_bert_0.965_20240530-2244.pkl",
    "ASF_bert_0.931_20240530-2235.pkl",
    "AFS_bert_0.979_20240530-2240.pkl",
]
PREDICTIONS_COLS = ["ahl_bert_preds", "asf_bert_preds", "afs_bert_preds"]
LABEL2ID = {"NOT RELEVANT": 0, "RELEVANT": 1}
SAVE_PATH = "data/crunchbase/enriched/organizations_new_only_with_bert_preds.csv"

if __name__ == "__main__":
    logging.info("Downloading new Crunchbase companies")
    new_companies = s3._download_obj(
        s3_client=client,
        bucket=s3.BUCKET_NAME_RAW,
        path_from="data/crunchbase/enriched/organizations_new_only.parquet",
        download_as="dataframe",
    )

    logging.info("Adding BERT mission predictions")
    new_companies_with_predictions = bert_add_predictions(
        missions=MISSIONS,
        model_filenames=MODEL_FILENAMES,
        df=new_companies,
        text_col="short_description",
        label2id=LABEL2ID,
        predictions_cols=PREDICTIONS_COLS,
    )

    logging.info("Saving new Crunchbase companies with BERT mission predictios to %s", SAVE_PATH)
    s3.upload_obj(new_companies_with_predictions, s3.BUCKET_NAME_RAW, SAVE_PATH)
