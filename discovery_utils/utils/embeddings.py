import pandas as pd

from sentence_transformers import SentenceTransformer


def add_embeddings(df: pd.DataFrame, text_col: str = "text", model_name: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    """
    Embeds text from a specified column in a DataFrame using a sentence transformer model.

    Adds the embedding and model name as new columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the text data.
        text_col (str): Column name of the DataFrame where the text is stored.
        model_name (str): Name of the sentence transformer model to use.

    Returns:
        pd.DataFrame: The original DataFrame with two new columns: 'embedding' and 'model_name'.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df[text_col].tolist(), show_progress_bar=True)
    df["embedding"] = list(embeddings)
    df["model_name"] = model_name
    return df
