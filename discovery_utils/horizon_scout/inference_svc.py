import numpy as np
import pandas as pd

from discovery_utils.getters.horizon_scout import get_svc_model
from discovery_utils.utils.embeddings import add_embeddings


def svc_add_predictions(
    mission: str,
    svc_model_filename: str,
    df: pd.DataFrame,
    text_col: str = "text",
    embedding_col: str = None,
    predictions_col: str = "predictions",
) -> pd.DataFrame:
    """Use SVC to add predictions to input DataFrame.

    Args:
        mission (str): Nesta mission ('AHL', 'AFS' or 'ASF').
        model_filename (str): Filename on S3 e.g. 'AHL_svc_0.882_20240517-1602.pkl'.
        df (pd.DataFrame): Input DataFrame containing the text data.
        text_col (str): Column name for the text in the DataFrame.
        embedding_col (str): Column name for the embeddings in the DataFrame.
        prediction_col (str): Column name to assign to the SVC predictions.

    Returns:
        pd.DataFrame: The original DataFrame with four additional columns:
            'embedding', 'embedding_model', "predictions", "prediction_model".
    """
    svc = get_svc_model(mission, svc_model_filename)
    df = df.dropna(subset=text_col)
    if not embedding_col:
        df = add_embeddings(df, text_col=text_col)
        X = np.vstack(df.embedding.values)
    else:
        X = np.vstack(df[embedding_col].values)
    df[predictions_col] = svc.predict(X=X)
    df["prediction_model"] = svc_model_filename
    return df
