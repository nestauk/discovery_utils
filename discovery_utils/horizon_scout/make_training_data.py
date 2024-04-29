from typing import Dict
from typing import List

import pandas as pd

from discovery_utils.utils import s3
from discovery_utils.utils.embeddings import add_embeddings


client = s3.s3_client()


class DataPaths:
    """Store the data paths"""

    base_path: str = "data/horizon_scout"
    classifier_inputs: str = f"{base_path}/classifier_training_data_inputs"
    classifier_outputs: str = f"{base_path}/classifier_training_data_outputs"
    paths: Dict[str, str] = {
        "AFS_POS": f"{classifier_inputs}/positive_samples_afs.csv",
        "AHL_POS": f"{classifier_inputs}/positive_samples_ahl.csv",
        "ASF_POS": f"{classifier_inputs}/positive_samples_asf.csv",
        "CB_ORGS": "data/crunchbase/Crunchbase_2024-04-17/organizations.parquet",
        "GTR_PROJECTS": "data/GtR/GtR_20240109/gtr_projects.json",
        "AFS_EXTRA": f"{classifier_inputs}/relevance_labels_eval_annotated_afs.jsonl",
        "AFS_TRAIN": f"{classifier_outputs}/training_data_afs.csv",
        "AHL_TRAIN": f"{classifier_outputs}/training_data_ahl.csv",
        "ASF_TRAIN": f"{classifier_outputs}/training_data_asf.csv",
    }


def load_and_process_positive_training_data(s3_path: str) -> pd.DataFrame:
    """
    Load and preprocess the positive training data from a CSV file from S3.

    Args:
        s3_path (str): The S3 file path to the CSV containing positive samples.

    Returns:
        pd.DataFrame: A DataFrame containing the processed positive
            training data, with duplicates and NA values removed,
            and a 'relevant' column set to 1.
    """
    return (
        s3._download_obj(s3_client=client, bucket=s3.BUCKET_NAME_RAW, path_from=s3_path, download_as="dataframe")[
            ["id", "text"]
        ]
        .dropna(subset=["id"])
        .drop_duplicates(subset=["id"])
        .assign(relevant=1)
        .reset_index(drop=True)
    )


def make_negative_training_data(data: pd.DataFrame, random_state: int, n_samples: int) -> pd.DataFrame:
    """
    Sample and label the negative training data from a given DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to sample from.
        random_state (int): A seed for the random number generator to ensure reproducibility.
        n_samples (int): The number of samples to draw.

    Returns:
        pd.DataFrame: A DataFrame of sampled negative data, with a 'relevant' column set to 0.
    """
    return data.sample(n=n_samples, random_state=random_state).assign(relevant=0).reset_index(drop=True)


def make_training_data(
    positive_training_data: pd.DataFrame,
    negative_data_list: List[pd.DataFrame],
    negative_data_frac_list: List[float],
    random_state: int,
) -> pd.DataFrame:
    """
    Combine the positive and negative training data into a single dataset.

    Args:
        positive_training_data (pd.DataFrame): DataFrame containing the positive samples.
        negative_data_list (List[pd.DataFrame]): A list of DataFrames from which negative samples will be drawn.
        negative_data_frac_list (List[float]): A list of fractions dictating the proportion of negative samples
                                               to draw relative to the number of positive samples.
        random_state (int): Seed for the random number generator used in sampling.

    Returns:
        pd.DataFrame: A DataFrame containing both positive and negative training data.
    """
    training_data = [positive_training_data]
    positive_training_data_len = len(positive_training_data)
    negative_data_n_samples_list = [int(positive_training_data_len * frac) for frac in negative_data_frac_list]
    for negative_data, negative_data_n_samples in zip(negative_data_list, negative_data_n_samples_list):
        negative_training_data = make_negative_training_data(negative_data, random_state, negative_data_n_samples)
        training_data.append(negative_training_data)
    return pd.concat(training_data).reset_index(drop=True)


if __name__ == "__main__":
    # Load positive training data
    afs_positive_training_data = load_and_process_positive_training_data(DataPaths.paths["AFS_POS"])
    ahl_positive_training_data = load_and_process_positive_training_data(DataPaths.paths["AHL_POS"])
    asf_positive_training_data = load_and_process_positive_training_data(DataPaths.paths["ASF_POS"])

    # Load crunchbase orgs data which will used to sample negative training data
    cb_orgs = (
        s3._download_obj(
            s3_client=client,
            bucket=s3.BUCKET_NAME_RAW,
            path_from=DataPaths.paths["CB_ORGS"],
            download_as="dataframe",
        )
        .query("short_description.notna()")[["id", "short_description"]]
        .rename(columns={"short_description": "text"})
    )

    # Load GtR projects which will used to sample negative training data
    gtr_projects_dict = s3._download_obj(
        s3_client=client,
        bucket=s3.BUCKET_NAME_RAW,
        path_from=DataPaths.paths["GTR_PROJECTS"],
        download_as="dict",
    )
    gtr_projects = pd.DataFrame([{"id": item["id"], "text": item["abstractText"]} for item in gtr_projects_dict])

    # Create training data for each mission
    afs_training_data = make_training_data(
        positive_training_data=afs_positive_training_data,
        negative_data_list=[ahl_positive_training_data, asf_positive_training_data, cb_orgs, gtr_projects],
        negative_data_frac_list=[0.15, 0.15, 0.35, 0.35],
        random_state=1,
    )

    ahl_training_data = make_training_data(
        positive_training_data=ahl_positive_training_data,
        negative_data_list=[afs_positive_training_data, asf_positive_training_data, cb_orgs, gtr_projects],
        negative_data_frac_list=[0.15, 0.15, 0.35, 0.35],
        random_state=2,
    )

    asf_training_data = make_training_data(
        positive_training_data=asf_positive_training_data,
        negative_data_list=[afs_positive_training_data, ahl_positive_training_data, cb_orgs, gtr_projects],
        negative_data_frac_list=[0.15, 0.15, 0.35, 0.35],
        random_state=3,
    )

    # Create extra training data for AFS from ISS 3
    afs_extra_training_data = (
        pd.DataFrame(
            s3._download_obj(
                s3_client=client, bucket=s3.BUCKET_NAME_RAW, path_from=DataPaths.paths["AFS_EXTRA"], download_as="list"
            )
        )[["id", "text", "prediction"]]
        .query("prediction != 'Not-specified'")
        .replace({"prediction": {"Not-relevant": 0, "Relevant": 1}})
        .rename(columns={"prediction": "relevant"})
    )

    # Add extra training data to AFS
    afs_training_data = pd.concat([afs_training_data, afs_extra_training_data])

    # Add embeddings to each dataset
    afs_training_data = add_embeddings(afs_training_data)
    ahl_training_data = add_embeddings(ahl_training_data)
    asf_training_data = add_embeddings(asf_training_data)

    # Save training datasets to s3
    s3.upload_obj(afs_training_data, s3.BUCKET_NAME_RAW, DataPaths.paths["AFS_TRAIN"])
    s3.upload_obj(ahl_training_data, s3.BUCKET_NAME_RAW, DataPaths.paths["AHL_TRAIN"])
    s3.upload_obj(asf_training_data, s3.BUCKET_NAME_RAW, DataPaths.paths["ASF_TRAIN"])
