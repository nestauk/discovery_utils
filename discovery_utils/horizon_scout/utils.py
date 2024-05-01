from datetime import datetime

import numpy as np

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
