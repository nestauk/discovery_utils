"""
Parliament Oral Evidence Sessions Getter with S3 Integration

This module downloads oral evidence session transcripts from
Parliament select committees using the Committees API,
with option to store results locally or in S3.
"""

import base64
import json
import logging
import os
import sys
import tempfile
import time

from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import boto3
import pandas as pd
import requests

from botocore.exceptions import ClientError

from discovery_utils.utils.s3 import BUCKET_NAME_RAW
from discovery_utils.utils.s3 import s3_client
from discovery_utils.utils.s3 import upload_obj


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SelectCommitteesToS3:
    """Class to handle S3 interactions for select committees data."""

    def __init__(self, prefix: str):
        """
        Initialise the S3 handler.

        Args:
            prefix: S3 prefix (folder path)
        """
        self.bucket = BUCKET_NAME_RAW
        self.prefix = prefix
        self.s3_client = s3_client()

        self.cumulative_file_key = f"{prefix}/select_committees.parquet"
        self.runs_prefix = f"{prefix}/runs"
        self.htmls_prefix = f"{prefix}/htmls"

        logger.info(f"Initialized S3 handler for bucket: {self.bucket}, prefix: {prefix}")

    def create_run_directory(self, run_date: datetime) -> str:
        """
        Create a run directory for the current run.

        Args:
            run_date: Date of the run

        Returns:
            Run directory prefix
        """
        date_str = run_date.strftime("%Y%m%d")
        return f"{self.runs_prefix}/{date_str}"

    def get_existing_evidence_ids(self) -> Set[str]:
        """
        Get set of IDs for evidence sessions that already exist in S3.

        Returns:
            Set of existing evidence IDs
        """
        existing_ids = set()

        # Try to get IDs from cumulative file
        try:
            # Check if cumulative file exists
            self.s3_client.head_object(Bucket=self.bucket, Key=self.cumulative_file_key)

            # Download to temp file and read
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = tmp.name
                self.s3_client.download_file(self.bucket, self.cumulative_file_key, tmp_path)
                df = pd.read_parquet(tmp_path)

                # Clean up
                os.unlink(tmp_path)

                # Extract IDs
                if "id" in df.columns:
                    existing_ids = set(df["id"].dropna().astype(str))
                    logger.info(f"Found {len(existing_ids)} existing evidence IDs in cumulative file")

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.info(f"No cumulative file found in S3")
            else:
                logger.warning(f"Error checking cumulative file: {e}")

        # Also check for any HTML files
        try:
            # List objects to find HTML files
            paginator = self.s3_client.get_paginator("list_objects_v2")
            html_count = 0

            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.htmls_prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if key.endswith(".html"):
                            html_count += 1
                            # Try to extract ID from filename
                            # Format: committee{committee_id}_{date}_{timestamp}_{title}.html
                            try:
                                # Extract just the evidence ID part if present
                                html_basename = os.path.basename(key)
                                # Look for evidence ID encoded in the filename
                                parts = html_basename.split("_")
                                if len(parts) >= 3:
                                    for part in parts:
                                        if part.isdigit():
                                            existing_ids.add(part)
                                            break
                            except Exception as e:
                                logger.debug(f"Could not extract ID from filename {key}: {e}")

            if html_count > 0:
                logger.info(f"Found {html_count} HTML files in S3")

        except Exception as e:
            logger.warning(f"Error checking for HTML files in S3: {e}")

        return existing_ids

    def upload_html(self, local_path: str, filename: str) -> str:
        """
        Upload an HTML file to S3.

        Args:
            local_path: Local path to the HTML file
            filename: Desired filename in S3

        Returns:
            S3 URI for the uploaded file
        """
        s3_key = f"{self.htmls_prefix}/{filename}"

        try:
            self.s3_client.upload_file(local_path, self.bucket, s3_key)
            logger.info(f"Uploaded HTML to S3: s3://{self.bucket}/{s3_key}")
            return f"s3://{self.bucket}/{s3_key}"
        except Exception as e:
            logger.error(f"Failed to upload HTML to S3: {e}")
            return ""

    def check_html_exists(self, filename: str) -> bool:
        """
        Check if an HTML file already exists in S3.

        Args:
            filename: HTML filename to check

        Returns:
            True if the HTML exists, False otherwise
        """
        s3_key = f"{self.htmls_prefix}/{filename}"

        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    def upload_run_data(self, df: pd.DataFrame, run_dir: str) -> str:
        """
        Upload data for a specific run to S3.

        Args:
            df: DataFrame with run data
            run_dir: Run directory prefix

        Returns:
            S3 URI for the uploaded file
        """
        if df.empty:
            logger.warning("Empty DataFrame, not uploading run data")
            return ""

        filename = f"select_committees.parquet"
        s3_key = f"{run_dir}/{filename}"

        try:
            # Upload the DataFrame
            upload_obj(df, self.bucket, s3_key)
            logger.info(f"Uploaded run data to S3: s3://{self.bucket}/{s3_key}")
            return f"s3://{self.bucket}/{s3_key}"
        except Exception as e:
            logger.error(f"Error uploading run data to S3: {e}")
            return ""

    def upload_artifact(self, artifact_data: dict, run_dir: str) -> str:
        """
        Upload pipeline artifact for a run.

        Args:
            artifact_data: Artifact data dictionary
            run_dir: Run directory prefix

        Returns:
            S3 URI for the uploaded artifact
        """
        filename = "ingestion_artifact.json"
        s3_key = f"{run_dir}/{filename}"

        try:
            # Upload the artifact
            upload_obj(artifact_data, self.bucket, s3_key)
            logger.info(f"Uploaded artifact to S3: s3://{self.bucket}/{s3_key}")
            return f"s3://{self.bucket}/{s3_key}"
        except Exception as e:
            logger.error(f"Error uploading artifact to S3: {e}")
            return ""

    def update_cumulative_file(self, new_data: pd.DataFrame) -> bool:
        """
        Update the cumulative select committees file with new data.

        Args:
            new_data: DataFrame with new select committees data

        Returns:
            True if update was successful, False otherwise
        """
        if new_data.empty:
            logger.warning("No new data to update cumulative file with")
            return False

        try:
            # Check if cumulative file exists
            cumulative_exists = True
            try:
                self.s3_client.head_object(Bucket=self.bucket, Key=self.cumulative_file_key)
            except ClientError:
                cumulative_exists = False

            if cumulative_exists:
                # Download existing file
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                    tmp_path = tmp.name
                    self.s3_client.download_file(self.bucket, self.cumulative_file_key, tmp_path)

                    # Load existing data
                    existing_df = pd.read_parquet(tmp_path)

                    # Clean up temp file
                    os.unlink(tmp_path)

                    # Merge with new data
                    logger.info(f"Updating cumulative file: {len(existing_df)} existing + {len(new_data)} new records")

                    # Concatenate and drop duplicates
                    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                    updated_df = combined_df.drop_duplicates(subset="id", keep="last")

                    logger.info(f"Cumulative file will have {len(updated_df)} records after deduplication")
            else:
                # First time creating cumulative file
                logger.info(f"Creating new cumulative file with {len(new_data)} records")
                updated_df = new_data

            # Upload updated file
            upload_obj(updated_df, self.bucket, self.cumulative_file_key)
            logger.info(f"Successfully updated cumulative file in S3: s3://{self.bucket}/{self.cumulative_file_key}")
            return True

        except Exception as e:
            logger.error(f"Error updating cumulative file: {e}")
            return False


class CommitteesAPI:
    """Client for interacting with the Parliament Committees API."""

    def __init__(self, base_url="https://committees-api.parliament.uk/api"):
        """Initialise the API client with the base URL."""
        self.base_url = base_url
        self.session = requests.Session()
        # Set a user agent
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "application/json",
            }
        )
        # Set default timeout
        self.timeout = 30

    def get_select_committees(self):
        """Get a list of all select committees."""
        url = f"{self.base_url}/Committees"
        params = {
            "CommitteeCategory": "Select",
            "CommitteeStatus": "Current",
            "Take": 100,  # Assuming we won't have more than 100 current select committees
        }

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving committees list: {e}")
            raise

    def get_oral_evidence_list(self, committee_id=None, start_date=None, end_date=None, skip=0, take=30):
        """Get a paginated list of oral evidence sessions matching criteria."""
        url = f"{self.base_url}/OralEvidence"
        params = {"Skip": skip, "Take": take}

        if committee_id:
            params["CommitteeId"] = committee_id

        if start_date:
            params["StartDate"] = start_date.isoformat()

        if end_date:
            params["EndDate"] = end_date.isoformat()

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving oral evidence list: {e}")
            raise

    def get_all_oral_evidence(self, committee_id=None, start_date=None, end_date=None):
        """Get all oral evidence sessions, handling pagination."""
        all_evidence = []
        skip = 0
        take = 30

        while True:
            try:
                batch = self.get_oral_evidence_list(
                    committee_id=committee_id, start_date=start_date, end_date=end_date, skip=skip, take=take
                )

                if not batch["items"]:
                    break

                all_evidence.extend(batch["items"])
                logger.debug(f"Downloaded {len(batch['items'])} evidence items (total: {len(all_evidence)})")

                # If we got fewer items than requested, we've reached the end
                if len(batch["items"]) < take:
                    break

                skip += take
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error retrieving oral evidence batch at skip={skip}: {e}")
                break

        logger.info(f"Retrieved {len(all_evidence)} total oral evidence sessions")
        return all_evidence

    def get_oral_evidence_document(self, evidence_id, file_format="Html"):
        """Get the document data for an oral evidence session."""
        url = f"{self.base_url}/OralEvidence/{evidence_id}/Document/{file_format}"

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error retrieving document for evidence {evidence_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving document for evidence {evidence_id}: {e}")
            return None


def clean_filename(filename):
    """Make a string safe for use as a filename."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Remove any other non-alphanumeric characters except underscores, hyphens and dots
    filename = "".join(c if c.isalnum() or c in ["_", "-", "."] else "_" for c in filename)

    # Truncate if too long
    max_length = 200
    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename


def get_existing_evidence_ids_local(output_dir: str) -> Set[str]:
    """
    Check local files to determine which evidence IDs already exist.

    Args:
        output_dir: Directory to check for existing files

    Returns:
        Set of evidence IDs that have already been processed
    """
    existing_ids = set()

    # Look for metadata parquet or CSV files
    metadata_files = [list(Path(output_dir).glob(f"select_committees*.csv"))]

    if metadata_files:
        # Use the most recent file
        latest_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Found existing metadata file: {latest_file}")

        try:
            # Read the file
            if latest_file.suffix == ".csv":
                df = pd.read_csv(latest_file)
            else:  # parquet
                df = pd.read_parquet(latest_file)

            # Extract IDs
            if "id" in df.columns:
                existing_ids = set(df["id"].dropna().astype(str))
                logger.info(f"Found {len(existing_ids)} existing evidence IDs in {latest_file}")
        except Exception as e:
            logger.warning(f"Error processing existing file {latest_file}: {e}")

    # Also check HTML directory to find additional IDs
    html_dir = os.path.join(output_dir, "htmls")
    if os.path.exists(html_dir):
        html_files = list(Path(html_dir).glob("*.html"))
        logger.info(f"Found {len(html_files)} HTML files in {html_dir}")

        # Try to extract IDs from filenames
        for html_file in html_files:
            filename = html_file.name
            parts = filename.split("_")

            # Look for a part that might be the evidence ID
            for part in parts:
                if part.isdigit():
                    existing_ids.add(part)
                    break

    logger.info(f"Total {len(existing_ids)} existing evidence IDs found locally")
    return existing_ids


def download_evidence_document(
    api: CommitteesAPI, evidence_id: int, output_path: str, file_format: str, max_retries: int = 3
) -> bool:
    """
    Download and save an evidence document with retries.

    Args:
        api: The API client
        evidence_id: ID of the evidence to download
        output_path: Path where to save the document
        file_format: Format of the document (Html, Pdf)
        max_retries: Maximum number of download attempts

    Returns:
        True if download was successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            # Get document details
            doc_details = api.get_oral_evidence_document(evidence_id, file_format)

            if not doc_details:
                logger.warning(f"No document details returned for evidence ID {evidence_id}")
                return False

            # Check for base64 encoded data - html is base64 encoded
            if "data" in doc_details and doc_details["data"]:
                logger.info(f"Saving document data for evidence ID {evidence_id}")

                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Decode base64 data and save
                try:
                    decoded_data = base64.b64decode(doc_details["data"])

                    with open(output_path, "wb") as f:
                        f.write(decoded_data)

                    logger.info(f"Successfully saved document to {output_path}")
                    return True
                except Exception as e:
                    logger.error(f"Error decoding or saving document data: {e}")
                    # If we're not at the last attempt, try again
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying download (attempt {attempt+1} of {max_retries})...")
                        time.sleep(2**attempt)  # Exponential backoff
                    continue
            else:
                logger.warning(f"No data field in document response for evidence ID {evidence_id}")
                return False

        except Exception as e:
            logger.error(f"Error downloading document for evidence ID {evidence_id}: {e}")
            # If we're not at the last attempt, try again
            if attempt < max_retries - 1:
                logger.info(f"Retrying download (attempt {attempt+1} of {max_retries})...")
                time.sleep(2**attempt)  # Exponential backoff
            continue

    # If we get here, all attempts failed
    return False


def process_evidence_session(
    api: CommitteesAPI,
    evidence: Dict[str, Any],
    evidence_ids: Set[str],
    output_dir: str,
    html_dir: str,
    file_format: str = "Html",
    use_s3: bool = False,
    s3_handler: Optional[SelectCommitteesToS3] = None,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Process a single evidence session: download document and extract metadata.

    Args:
        api: The API client
        evidence: Evidence session data from the API
        evidence_ids: Set of already downloaded evidence IDs
        output_dir: Base output directory
        html_dir: Directory to save HTML files
        file_format: Format to download (Html, Pdf)
        use_s3: Whether to use S3 storage
        s3_handler: S3 handler if use_s3 is True

    Returns:
        Tuple of (metadata dictionary or None, success flag)
    """
    # Extract evidence ID
    evidence_id = evidence.get("id")

    if not evidence_id:
        logger.warning("Evidence session has no ID, skipping")
        return None, False

    # Skip if already downloaded
    if str(evidence_id) in evidence_ids:
        logger.info(f"Evidence ID {evidence_id} already processed, skipping")
        return None, True

    # Get publication date
    pub_date = evidence.get("publicationDate")
    pub_date_str = pub_date.split("T")[0] if pub_date else "unknown_date"

    # Extract timestamp for unique filename
    timestamp = ""
    if pub_date and "T" in pub_date:
        try:
            # Extract time component from ISO format (2023-04-01T14:30:00Z)
            time_part = pub_date.split("T")[1].split(".")[0].replace(":", "-")
            timestamp = f"_{time_part}"
        except Exception:
            # Fallback to using the evidence_id as a unique identifier
            timestamp = f"_{evidence_id}"
    else:
        # Always add evidence_id as a fallback unique identifier
        timestamp = f"_{evidence_id}"

    # Get committee information
    committee_id = None
    committee_name = "unknown_committee"
    if evidence.get("committees") and len(evidence["committees"]) > 0:
        committee = evidence["committees"][0]
        committee_id = committee.get("id")
        committee_name = committee.get("name", committee_name)

    business_title = "unknown_inquiry"
    if evidence.get("committeeBusinesses") and len(evidence["committeeBusinesses"]) > 0:
        business_title = evidence["committeeBusinesses"][0].get("title", business_title)

    # Build filename with committee ID, date, timestamp, and topic
    extension = "html" if file_format.lower() == "html" else "pdf" if file_format.lower() == "pdf" else "txt"
    filename = f"committee{committee_id}_{pub_date_str}{timestamp}_{clean_filename(business_title)}.{extension}"

    # Set up paths
    local_path = os.path.join(html_dir, clean_filename(filename))
    s3_path = ""

    # Check if file already exists locally or in S3
    file_exists_local = os.path.exists(local_path)
    file_exists_s3 = False

    if use_s3 and s3_handler:
        file_exists_s3 = s3_handler.check_html_exists(clean_filename(filename))

    if file_exists_local:
        logger.info(f"Evidence document already exists locally: {filename}")
        # Store local path
        html_file = local_path
        if use_s3:
            # Upload to S3 if it exists locally but not in S3
            if not file_exists_s3:
                s3_path = s3_handler.upload_html(local_path, clean_filename(filename))
            else:
                s3_path = f"s3://{s3_handler.bucket}/{s3_handler.htmls_prefix}/{clean_filename(filename)}"
    elif use_s3 and file_exists_s3:
        logger.info(f"Evidence document exists in S3 but not locally: {filename}")
        # Download from S3 to local
        s3_key = f"{s3_handler.htmls_prefix}/{clean_filename(filename)}"
        s3_path = f"s3://{s3_handler.bucket}/{s3_key}"

        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            # Download file
            s3_handler.s3_client.download_file(s3_handler.bucket, s3_key, local_path)
            logger.info(f"Downloaded document from S3 to {local_path}")
            html_file = local_path
        except Exception as e:
            logger.error(f"Failed to download document from S3: {e}")
            # Try downloading from API
            success = download_evidence_document(api, evidence_id, local_path, file_format)
            if success:
                html_file = local_path
                # Upload to S3
                if use_s3:
                    s3_path = s3_handler.upload_html(local_path, clean_filename(filename))
            else:
                logger.error(f"Failed to download evidence document from API: {evidence_id}")
                return None, False
    else:
        # Download from API
        logger.info(f"Downloading evidence document for ID {evidence_id}")
        success = download_evidence_document(api, evidence_id, local_path, file_format)

        if success:
            html_file = local_path
            # Upload to S3 if enabled
            if use_s3:
                s3_path = s3_handler.upload_html(local_path, clean_filename(filename))
        else:
            logger.error(f"Failed to download evidence document from API: {evidence_id}")
            return None, False

    # Create metadata
    metadata = {
        "id": str(evidence_id),
        "committee_id": committee_id,
        "committee_name": committee_name,
        "business_title": business_title,
        "publication_date": pub_date,
        "date_str": pub_date_str,
        "html_file": local_path,
    }

    # Add S3 path if available
    if s3_path:
        metadata["s3_html_file"] = s3_path

    # Extract additional metadata fields if available
    if "meetingDate" in evidence:
        metadata["meeting_date"] = evidence["meetingDate"]

    if "witnesses" in evidence:
        witness_names = []
        for witness in evidence.get("witnesses", []):
            name = witness.get("name", "")
            if name:
                witness_names.append(name)
        metadata["witnesses"] = "; ".join(witness_names)

    # Add evidence to processed IDs
    evidence_ids.add(str(evidence_id))

    return metadata, True


def generate_pipeline_artifact(
    metadata: Dict[str, Any],
    output_dir: str,
    run_date: datetime,
    artifact_filename: str = "pipeline_artifact.json",
    use_s3: bool = False,
    s3_handler: Optional[SelectCommitteesToS3] = None,
    run_dir: Optional[str] = None,
) -> str:
    """
    Generate a pipeline artifact file from the metadata.

    Args:
        metadata: Dictionary with download metadata
        output_dir: Directory to save the artifact locally
        run_date: Date of the run
        artifact_filename: Name of the artifact file
        use_s3: Whether to use S3 storage
        s3_handler: S3 handler if use_s3 is True
        run_dir: Run directory in S3 if use_s3 is True

    Returns:
        Path to the created artifact file
    """
    # Add timestamp to artifact
    artifact_data = {
        "run_date": run_date.isoformat(),
    }

    # Add formatted summary
    artifact_data["summary"] = {
        "total_evidence_sessions_found": metadata.get("total_evidence_sessions", 0),
        "evidence_sessions_processed": metadata.get("evidence_sessions_processed", 0),
        "htmls_downloaded": metadata.get("htmls_downloaded", 0),
        "date_range": {
            "start": metadata.get("date_range", {}).get("start", ""),
            "end": metadata.get("date_range", {}).get("end", ""),
        },
        "output_files": {
            "metadata_file": os.path.basename(metadata.get("metadata_file", "")),
            "html_count": metadata.get("htmls_downloaded", 0),
        },
        "status": "success" if not metadata.get("error") else "error",
        "error_message": metadata.get("error", ""),
        "storage_mode": "s3" if use_s3 else "local",
    }

    if use_s3 and s3_handler and run_dir:
        # Upload to S3
        s3_path = s3_handler.upload_artifact(artifact_data, run_dir)
        artifact_path = s3_path
    else:
        # Save locally
        artifact_path = os.path.join(output_dir, artifact_filename)
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact_data, f, indent=2)

        logger.info(f"Generated pipeline artifact at {artifact_path}")

    return artifact_path


def download_select_committees_evidence(
    output_dir: str = "select_committees",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    file_format: str = "Html",
    batch_size: int = 50,
    use_s3: bool = False,
    s3_prefix: str = None,
    update_cumulative: bool = True,
) -> Dict[str, Any]:
    """
    Download oral evidence sessions from select committees.

    Args:
        output_dir: Directory to save the evidence data
        start_date: Minimum date for filtering (defaults to 7 days ago)
        end_date: Maximum date for filtering (defaults to now)
        file_format: Format to download (Html, Pdf)
        batch_size: Number of evidence sessions to process in each batch
        use_s3: Whether to use S3 storage
        s3_prefix: S3 prefix (folder path) if use_s3 is True
        update_cumulative: Whether to update the cumulative file in S3

    Returns:
        Dictionary with metadata about the download
    """
    run_date = datetime.now()

    # Create S3 handler if using S3
    s3_handler = None
    run_dir = None
    if use_s3:
        s3_handler = SelectCommitteesToS3(s3_prefix)
        run_dir = s3_handler.create_run_directory(run_date)
        logger.info(f"Created run directory: {run_dir}")

    os.makedirs(output_dir, exist_ok=True)
    html_dir = os.path.join(output_dir, "htmls")
    os.makedirs(html_dir, exist_ok=True)

    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now()

    if not start_date:
        # Default to 7 days before end date
        start_date = end_date - timedelta(days=7)

    logger.info(f"Downloading evidence sessions from {start_date.date()} to {end_date.date()}")
    if use_s3:
        logger.info(f"Using S3 storage: s3://{BUCKET_NAME_RAW}/{s3_prefix}")
    else:
        logger.info(f"Using local storage: {output_dir}")

    # Initialize API client
    api = CommitteesAPI()

    # Get existing evidence IDs to avoid re-downloading
    if use_s3:
        evidence_ids = s3_handler.get_existing_evidence_ids()
    else:
        evidence_ids = get_existing_evidence_ids_local(output_dir)

    # Get all oral evidence sessions in the date range
    try:
        all_evidence = api.get_all_oral_evidence(start_date=start_date, end_date=end_date)
    except Exception as e:
        logger.error(f"Error retrieving evidence sessions: {e}")
        error_metadata = {
            "download_date": run_date.isoformat(),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_evidence_sessions": 0,
            "evidence_sessions_processed": 0,
            "htmls_downloaded": 0,
            "error": str(e),
        }
        # Generate pipeline artifact for the error case
        generate_pipeline_artifact(
            error_metadata, output_dir, run_date, use_s3=use_s3, s3_handler=s3_handler, run_dir=run_dir
        )
        return error_metadata

    logger.info(f"Found {len(all_evidence)} evidence sessions in total")

    if not all_evidence:
        logger.warning("No evidence sessions found matching the criteria")
        no_data_metadata = {
            "download_date": run_date.isoformat(),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_evidence_sessions": 0,
            "evidence_sessions_processed": 0,
            "htmls_downloaded": 0,
            "status": "success",
        }
        # Generate pipeline artifact for the no data case
        generate_pipeline_artifact(
            no_data_metadata, output_dir, run_date, use_s3=use_s3, s3_handler=s3_handler, run_dir=run_dir
        )
        return no_data_metadata

    # Split evidence sessions into batches
    evidence_batches = [all_evidence[i : i + batch_size] for i in range(0, len(all_evidence), batch_size)]
    logger.info(f"Processing {len(all_evidence)} evidence sessions in {len(evidence_batches)} batches")

    # Process evidence sessions
    all_metadata = []
    html_downloaded_count = 0
    already_existing_count = 0

    for batch_index, batch in enumerate(evidence_batches):
        logger.info(f"Processing batch {batch_index+1}/{len(evidence_batches)} with {len(batch)} evidence sessions")

        batch_metadata = []
        for evidence in batch:
            metadata, success = process_evidence_session(
                api=api,
                evidence=evidence,
                evidence_ids=evidence_ids,
                output_dir=output_dir,
                html_dir=html_dir,
                file_format=file_format,
                use_s3=use_s3,
                s3_handler=s3_handler,
            )

            if metadata:
                batch_metadata.append(metadata)
                if "html_file" in metadata:
                    html_downloaded_count += 1
            elif success:
                # It was successful but evidence already existed
                already_existing_count += 1

        all_metadata.extend(batch_metadata)
        logger.info(f"Batch {batch_index+1} processed: {len(batch_metadata)} evidence sessions downloaded")

        if batch_index < len(evidence_batches) - 1:
            time.sleep(1)

    # Create DataFrame from metadata
    if all_metadata:
        metadata_df = pd.DataFrame(all_metadata)
        logger.info(f"Created metadata DataFrame with {len(metadata_df)} rows")

        # Save DataFrame
        metadata_file = os.path.join(output_dir, f"select_committees.parquet")

        try:
            metadata_df.to_parquet(metadata_file, index=False)
            logger.info(f"Saved metadata to {metadata_file}")

            # Upload to S3 if enabled
            if use_s3 and s3_handler:
                s3_metadata_file = s3_handler.upload_run_data(metadata_df, run_dir)

                # Update cumulative file if requested
                if update_cumulative and not metadata_df.empty:
                    s3_handler.update_cumulative_file(metadata_df)
        except Exception as e:
            logger.error(f"Error saving metadata DataFrame: {e}")
            metadata_file = None
    else:
        logger.warning("No metadata collected for evidence sessions")
        metadata_df = pd.DataFrame()
        metadata_file = None

    metadata = {
        "download_date": run_date.isoformat(),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "total_evidence_sessions": len(all_evidence),
        "evidence_sessions_processed": len(all_metadata),
        "htmls_downloaded": html_downloaded_count,
        "htmls_already_existing": already_existing_count,
        "metadata_file": metadata_file,
        "status": "success",
        "storage_mode": "s3" if use_s3 else "local",
    }

    generate_pipeline_artifact(metadata, output_dir, run_date, use_s3=use_s3, s3_handler=s3_handler, run_dir=run_dir)

    logger.info(f"Download complete.")
    logger.info(f"Found {len(all_evidence)} evidence sessions")
    logger.info(f"Processed {len(all_metadata)} new evidence sessions")
    logger.info(f"Downloaded {html_downloaded_count} HTML files")
    logger.info(f"Found {already_existing_count} already existing evidence sessions")

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Parliament Select Committee Oral Evidence")
    parser.add_argument(
        "--output", "-o", default="outputs/select_committees", help="Output directory for downloaded evidence"
    )
    parser.add_argument("--days", "-d", type=int, default=7, help="Number of days to look back for evidence")
    parser.add_argument("--start-date", "-s", help="Start date in YYYY-MM-DD format (overrides days parameter)")
    parser.add_argument("--end-date", "-e", help="End date in YYYY-MM-DD format (defaults to today)")
    parser.add_argument(
        "--format", "-f", choices=["Html", "Pdf"], default="Html", help="Format to download (Html, Pdf)"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=50, help="Number of evidence sessions to process in each batch"
    )
    parser.add_argument("--use-s3", action="store_true", help="Store files in S3 instead of locally")
    parser.add_argument(
        "--s3-prefix",
        default="data/policy/select_committees",
        help="S3 prefix (folder path) (required if --use-s3 is specified)",
    )
    parser.add_argument(
        "--no-update-cumulative",
        action="store_true",
        help="Don't update the cumulative file with new data (only applies with --use-s3)",
    )

    args = parser.parse_args()

    # Parse dates if provided
    end_date = None
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid end date format. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        end_date = datetime.now()

    start_date = None
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid start date format. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        start_date = end_date - timedelta(days=args.days)

    # Download evidence
    download_select_committees_evidence(
        output_dir=args.output,
        start_date=start_date,
        end_date=end_date,
        file_format=args.format,
        batch_size=args.batch_size,
        use_s3=args.use_s3,
        s3_prefix=args.s3_prefix,
        update_cumulative=not args.no_update_cumulative,
    )
