"""
discovery_utils.getters.crunchbase.py

Module for easy access to downloaded CB data on S3.

"""
import pandas as pd
import boto3


# Define global variables
BUCKET_NAME = "discovery-iss"
S3_PATH = "data/crunchbase/"
S3 = boto3.client("s3")
FILE_NAMES = [
    "acquisitions.csv",
    "category_groups.csv",
    "checksum.csv",
    "degrees.csv",
    "event_appearances.csv",
    "events.csv",
    "funding_rounds.csv",
    "funds.csv",
    "investment_partners.csv",
    "investments.csv",
    "investors.csv",
    "ipos.csv",
    "jobs.csv",
    "org_parents.csv",
    "organization_descriptions.csv",
    "organizations.csv",
    "people_descriptions.csv",
    "people.csv"
]


# Building block functions - not to be used directly

def latest_version() -> str:
    """Directory name for the latest version of CB data."""
    return sorted(S3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PATH)["Contents"])[-1]["Key"]

def secondlatest_version() -> str:
    """Directory name for the second latest version of CB data."""
    return sorted(S3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PATH)["Contents"])[-2]["Key"]

def get_crunchbase_table(file_name, version) -> pd.DataFrame:
    """Table with named CB data"""
    # Read the file contents from S3
    response = S3.get_object(Bucket=BUCKET_NAME, Key=S3_PATH + version + file_name)
    data = response["Body"].read().decode("utf-8")
    
    return pd.read_csv(data)


# Master functions - to be used directly

def latest_table(file_name) -> pd.DataFrame:
    """Specific table from latest CB snapshot"""
    return get_crunchbase_table(file_name, latest_version())

def secondlatest_table(file_name) -> pd.DataFrame:
    """Specific table from second latest CB snapeshot"""
    return get_crunchbase_table(file_name, secondlatest_version())