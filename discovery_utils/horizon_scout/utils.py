from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from discovery_utils import logging


logging.basicConfig(level=logging.INFO)


def display_confusion_matrix(actual: np.array, predictions: np.array) -> None:
    """
    Log the confusion matrix along with precision, recall, and accuracy metrics.

    Args:
        actual (np.ndarray): Array of actual target values.
        predictions (np.ndarray): Array of predicted target values.
    """
    cm = confusion_matrix(actual, predictions)
    classes = ["Neg", "Pos"]

    logging.info("            Predicted:")
    logging.info("            | %s | %s |", classes[0], classes[1])
    logging.info("-----------------------------")
    for i, row in enumerate(cm):
        logging.info("Actual %s | %4d | %4d |", classes[i], row[0], row[1])

    precision = precision_score(actual, predictions, average="binary")
    recall = recall_score(actual, predictions, average="binary")
    accuracy = accuracy_score(actual, predictions)

    logging.info("Precision: %.3f", precision)
    logging.info("Recall: %.3f", recall)
    logging.info("Accuracy: %.3f", accuracy)


def get_current_datetime() -> str:
    """Get the current date and time in the format yyyymmdd-hhmmss."""
    return datetime.now().strftime("%Y%m%d-%H%M")


def make_train_val_datasets(
    dataset: pd.DataFrame,
    random_state: int,
    training_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the input DataFrame into  training and validation datasets.

    First shuffle the dataset using a specified random state.
    Then partitions the dataset into training and validation sets.

    Args:
        dataset (pd.DataFrame): The complete dataset to be split.
        random_state (int): A seed used by the random number generator for shuffling the data.
        training_frac (float): The fraction of the dataset to allocate to the training set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
        (training_dataset, val_dataset).
    """
    shuffled_dataset = dataset.sample(frac=1, random_state=random_state)
    train_size = int(len(shuffled_dataset) * training_frac)
    train_dataset = shuffled_dataset[:train_size]
    val_dataset = shuffled_dataset[train_size:]
    return (train_dataset.reset_index(drop=True), val_dataset.reset_index(drop=True))
