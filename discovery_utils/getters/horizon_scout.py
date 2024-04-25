import pandas as pd

from discovery_utils.horizon_scout.make_training_data import DataPaths
from discovery_utils.utils import s3


client = s3.s3_client()


def get_training_data(mission: str) -> pd.DataFrame:
    """Load training data from S3

    Args:
        mission (str): Nesta mission ('AHL', 'AFS' or 'ASF')

    Returns:
        (pd.DataFrame): Training dataset relevant to the specified mission
                        ('AHL', 'AFS' or 'ASF')

    """
    return s3._download_obj(
        s3_client=client,
        bucket=s3.BUCKET_NAME_RAW,
        path_from=DataPaths.paths[f"{mission}_TRAIN"],
        download_as="dataframe",
    )
