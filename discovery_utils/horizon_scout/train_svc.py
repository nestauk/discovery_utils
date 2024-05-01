from typing import Tuple

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import GridSearchCV

from discovery_utils import logging
from discovery_utils.getters.horizon_scout import get_training_data
from discovery_utils.horizon_scout.utils import get_current_datetime
from discovery_utils.utils import s3


logging.basicConfig(level=logging.INFO)


client = s3.s3_client()


def make_x_y(train_dataset: pd.DataFrame) -> Tuple[np.array, np.array]:
    """
    Create X (text embeddings) and y (target) arrays.

    Args:
        train_dataset (pd.DataFrame): The training dataset containing features and target.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays: (X, y).
    """
    X = np.vstack(train_dataset.embedding.values)
    y = train_dataset.relevant.values
    return (X, y)


MISSION = "AHL"


if __name__ == "__main__":
    # Make X and y for each mission
    X, y = make_x_y(get_training_data(MISSION))

    param_grid = [
        {
            "C": [1.1, 1.2, 1.3],
            "kernel": ["poly"],
            "degree": [3],
            "gamma": ["scale"],
            "coef0": [0.0000001, 0.000001, 0.000002],
        }
    ]

    # Carry out Cross Validation Grid Search to find best parameters
    svc = svm.SVC()
    grid_search = GridSearchCV(svc, param_grid, cv=5, verbose=3)
    grid_search.fit(X, y)

    best_score = round(grid_search.best_score_, 3)

    logging.info("Best parameters found: %s", grid_search.best_params_)
    logging.info("Best accuracy: %s", best_score)

    # Calculate datetime
    current_datetime = get_current_datetime()

    # Save best model to s3
    s3.upload_obj(
        grid_search.best_estimator_,
        s3.BUCKET_NAME_RAW,
        f"models/horizon_scout/{MISSION}/{MISSION}_svc_{best_score}_{current_datetime}.pkl",
    )
