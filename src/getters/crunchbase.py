"""
discovery_utils.src.getters.crunchbase.py

Module for easy access to downloaded CB data on S3.

"""
import pandas as pd
import boto3
import logging
import re


# Define global variables
BUCKET_NAME = "discovery-iss"
S3_PATH = "data/crunchbase/"
AWS_ACCESS_KEY =
AWS_SECRET_KEY = 
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


# Initialize S3 client
S3 = boto3.client(
    "s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY
    )


# Building block functions - not to be used directly

def list_s3_directories() -> list[str]:
    """List S3 directories under the specified prefix."""
    # List all objects with the specified prefix
    objects = S3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PATH, Delimiter='/')

    # Extract the directory names from the CommonPrefixes list
    directories = [prefix['Prefix'] for prefix in objects.get('CommonPrefixes', [])]

    return directories

def extract_timestamp(directory: str) -> str or None:
    """ Use regular expression to extract the timestamp from a directory name"""
    match = re.search(r'\d{8}_\d{6}', directory)
    if match:
        timestamp_str = match.group()
        return timestamp_str
    else:
        return None
    
def directory(position: int) -> str:
    """Directory name for a given snapshot of CB data."""
    
    # Get the list of directories
    directories = list_s3_directories()
    
    # Get a list of timestamps
    timestamps = []
    for directory in directories:
        timestamps.append(extract_timestamp(directory))
    
    # Sort the timestamps in descending order
    sorted_timestamps = sorted(timestamps, reverse=True)
    
    # Return the most recent directory
    for directory in directories:
        if sorted_timestamps[position] in directory:
            return directory

def get_crunchbase_table(file_name: str, version: str) -> pd.DataFrame:
    """Table with named CB data"""
    s3_key = S3_PATH + version + file_name
    logging.info(f"Fetching data from S3 key: {s3_key}")

    # Read the file contents from S3
    response = S3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
    data = response["Body"].read().decode("utf-8")
    
    return pd.read_csv(data)


# Master functions - to be used directly

def latest_table(file_name: str) -> pd.DataFrame:
    """Specific table from latest CB snapshot"""
    return get_crunchbase_table(file_name, directory(0))

def secondlatest_table(file_name) -> pd.DataFrame:
    """Specific table from second latest CB snapeshot"""
    return get_crunchbase_table(file_name, directory(1))


# Example use
# Get the latest organizations table

table = latest_table("organizations.csv")
# Or
table = latest_table(FILE_NAMES[15])