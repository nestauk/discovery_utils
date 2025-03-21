"""
Research Briefings Getter with S3 Integration

This script downloads research briefings data from the Parliament Open Data API,
including both the JSON metadata and PDF documents. It stores the metadata in
a DataFrame and downloads PDFs as separate files, with options to store results
locally or in S3.
"""

import glob
import json
import logging
import os
import re
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
import cloudscraper
import pandas as pd
import requests

from botocore.exceptions import ClientError

# Import S3 utilities
from discovery_utils.utils.s3 import BUCKET_NAME_RAW
from discovery_utils.utils.s3 import s3_client
from discovery_utils.utils.s3 import upload_obj


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("research_briefings_download.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ResearchBriefingsAPI:
    """Client for interacting with the UK Parliament Research Briefings API."""

    def __init__(self, base_url: str = "https://lda.data.parliament.uk") -> None:
        """Initialise the API client with the base URL."""
        self.base_url = base_url
        self.session = requests.Session()
        # Set a user agent - this is required for PDF downloads
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "application/json",
            }
        )
        # Set default timeout for all requests
        self.timeout = 30

    def get_research_briefings(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 0,
        page_size: int = 500,
    ) -> Dict[str, Any]:
        """
        Get a list of research briefings matching the specified criteria.

        Args:
            start_date: Minimum date for the 'date' field
            end_date: Maximum date for the 'date' field
            page: Page number for pagination
            page_size: Number of results per page

        Returns:
            Dictionary containing the API response
        """
        url = f"{self.base_url}/researchbriefings"
        params = {"_page": page, "_pageSize": min(page_size, 500)}  # API has a 500 result limit per page

        # Add date range filters if provided
        if start_date:
            params["min-date"] = start_date.isoformat()
        if end_date:
            params["max-date"] = end_date.isoformat()

        logger.debug(f"Requesting research briefings with params: {params}")

        # Request the data in JSON format with timeout
        try:
            response = self.session.get(f"{url}.json", params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Timeout error retrieving briefings list")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving briefings list: {e}")
            raise

    def get_all_research_briefings(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all research briefings matching criteria, handling pagination.

        Args:
            start_date: Minimum date for the 'date' field
            end_date: Maximum date for the 'date' field

        Returns:
            List of all briefing items
        """
        all_briefings = []
        page = 0
        page_size = 100
        total_pages = 1  # Initial value, will be updated from the response

        logger.info(f"Fetching research briefings from {start_date} to {end_date}")

        while page < total_pages:
            response = self.get_research_briefings(
                start_date=start_date, end_date=end_date, page=page, page_size=page_size
            )

            # Extract items from the response
            if "result" in response and "items" in response["result"]:
                items = response["result"]["items"]
                all_briefings.extend(items)

                # Update pagination info
                if page == 0:  # First page
                    if "itemsPerPage" in response["result"]:
                        page_size = response["result"]["itemsPerPage"]
                    if "totalResults" in response["result"]:
                        total_results = response["result"]["totalResults"]
                        total_pages = (total_results + page_size - 1) // page_size
                        logger.info(f"Found {total_results} briefings across {total_pages} pages")
            else:
                logger.warning("Unexpected response format")
                break

            page += 1
            logger.info(f"Fetched page {page} of {total_pages}")

            if page < total_pages:
                time.sleep(0.5)

        return all_briefings

    def get_research_briefing_by_id(self, briefing_id: str) -> Dict[str, Any]:
        """
        Get a specific research briefing by its ID.

        Args:
            briefing_id: The ID of the briefing to retrieve

        Returns:
            Dictionary containing the briefing data
        """
        url = f"{self.base_url}/researchbriefings/{briefing_id}"
        try:
            # Set a timeout to prevent indefinite waiting
            response = self.session.get(f"{url}.json", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # Log the URL from the result field
            if "result" in data and "_about" in data["result"]:
                logger.debug(f"Found briefing URL: {data['result']['_about']}")

            return data
        except requests.exceptions.Timeout:
            logger.error(f"Timeout error retrieving briefing {briefing_id}")
            return {"error": "Timeout"}
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error retrieving briefing {briefing_id}")
            return {"error": "ConnectionError"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving briefing {briefing_id}: {e}")
            return {"error": str(e)}


class ResearchBriefingsToS3:
    """Class to handle S3 interactions for research briefings data."""

    def __init__(self, prefix: str):
        """
        Initialize the S3 handler.

        Args:
            bucket: S3 bucket name
            prefix: S3 prefix (folder path)
        """
        self.bucket = BUCKET_NAME_RAW
        self.prefix = prefix
        self.s3_client = s3_client()

        # Define standard paths
        self.cumulative_file_key = f"{prefix}/research_briefings.parquet"
        self.runs_prefix = f"{prefix}/runs"
        self.pdfs_prefix = f"{prefix}/pdfs"

        logger.info(f"Initialized S3 handler for bucket: {self.bucket}, prefix: {prefix}")

    def create_run_directory(self, run_date: datetime) -> str:
        """
        Create a run directory for the current run.

        Args:
            run_date: Date of the run

        Returns:
            Run directory prefix
        """
        date_str = run_date.strftime("%Y%m%d_%H%M%S")
        return f"{self.prefix}/runs/{date_str}"

    def get_existing_briefing_ids(self) -> Set[str]:
        """
        Get set of IDs for briefings that already exist in S3.

        Returns:
            Set of existing briefing IDs
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

                # Read the parquet file
                df = pd.read_parquet(tmp_path)

                # Clean up
                os.unlink(tmp_path)

                # Extract IDs
                if "id" in df.columns:
                    existing_ids = set(df["id"].dropna().astype(str))
                    logger.info(f"Found {len(existing_ids)} existing briefing IDs in cumulative file")

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.info(f"No cumulative file found in S3")
            else:
                logger.warning(f"Error checking cumulative file: {e}")

        # Also check for any PDFs
        try:
            # List objects to find PDFs
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pdf_count = 0

            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.pdfs_prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if key.endswith(".pdf"):
                            pdf_count += 1
                            # Extract ID from filename
                            pdf_basename = os.path.basename(key)
                            pdf_id = os.path.splitext(pdf_basename)[0]
                            existing_ids.add(pdf_id)

            if pdf_count > 0:
                logger.info(f"Found {pdf_count} PDF files in S3")

        except Exception as e:
            logger.warning(f"Error checking for PDFs in S3: {e}")

        return existing_ids

    def upload_pdf(self, local_path: str, filename: str) -> str:
        """
        Upload a PDF file to S3.

        Args:
            local_path: Local path to the PDF file
            filename: Desired filename in S3

        Returns:
            S3 URI for the uploaded file
        """
        s3_key = f"{self.pdfs_prefix}/{filename}"

        try:
            self.s3_client.upload_file(local_path, self.bucket, s3_key)
            logger.info(f"Uploaded PDF to S3: s3://{self.bucket}/{s3_key}")
            return f"s3://{self.bucket}/{s3_key}"
        except Exception as e:
            logger.error(f"Failed to upload PDF to S3: {e}")
            return ""

    def check_pdf_exists(self, filename: str) -> bool:
        """
        Check if a PDF already exists in S3.

        Args:
            filename: PDF filename to check

        Returns:
            True if the PDF exists, False otherwise
        """
        s3_key = f"{self.pdfs_prefix}/{filename}"

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

        filename = f"research_briefings.parquet"
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
        filename = "artifact.json"
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
        Update the cumulative research briefings file with new data.

        Args:
            new_data: DataFrame with new briefings data

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


def extract_nested_value(data: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    """
    Extract a value from a nested dictionary using a path.

    Args:
        data: The dictionary to extract from
        path: List of keys forming a path to the desired value
        default: Value to return if the path doesn't exist

    Returns:
        The extracted value or default
    """
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    # If the value is a dictionary with "_value" key, extract that value
    if isinstance(current, dict) and "_value" in current:
        return current["_value"]

    return current


def clean_filename(filename: str) -> str:
    """
    Make a string safe for use as a filename.

    Args:
        filename: String to clean

    Returns:
        A cleaned string safe for use as a filename
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Remove any other non-alphanumeric characters except underscores, hyphens and dots
    filename = re.sub(r"[^\w\-\.]", "_", filename)

    # Truncate if too long
    max_length = 200
    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename


def get_existing_briefing_ids_local(output_dir: str) -> Set[str]:
    """
    Check if a metadata CSV exists locally and extract briefing IDs from it.

    Args:
        output_dir: Directory to check for existing metadata CSV

    Returns:
        Set of briefing IDs that have already been downloaded
    """
    existing_ids = set()

    # Look for metadata CSV files
    csv_files = glob.glob(os.path.join(output_dir, "research_briefings_*.csv"))
    parquet_files = glob.glob(os.path.join(output_dir, "research_briefings_*.parquet"))

    metadata_files = csv_files + parquet_files

    if not metadata_files:
        logger.info(f"No existing metadata files found in {output_dir}")
        return existing_ids

    # Use the most recent file
    latest_file = max(metadata_files, key=os.path.getmtime)
    logger.info(f"Found existing metadata file: {latest_file}")

    try:
        # Read the file
        if latest_file.endswith(".csv"):
            df = pd.read_csv(latest_file)
        else:  # parquet
            df = pd.read_parquet(latest_file)

        # Extract IDs
        if "id" in df.columns:
            existing_ids = set(df["id"].dropna().astype(str))
            logger.info(f"Found {len(existing_ids)} existing briefing IDs in {latest_file}")
    except Exception as e:
        logger.warning(f"Error processing existing file {latest_file}: {e}")

    # Also check PDFs directory to find additional IDs
    pdf_dir = os.path.join(output_dir, "pdfs")
    if os.path.exists(pdf_dir):
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

        for pdf_file in pdf_files:
            # Extract ID from filename
            pdf_basename = os.path.basename(pdf_file)
            pdf_id = os.path.splitext(pdf_basename)[0]
            existing_ids.add(pdf_id)

    logger.info(f"Total {len(existing_ids)} existing briefing IDs found locally")
    return existing_ids


def download_pdf_for_briefing(pdf_url: str, pdf_path: str, max_attempts: int = 3) -> bool:
    """
    Download a PDF document with retry mechanism.

    Args:
        pdf_url: URL of the PDF to download
        pdf_path: Path where the PDF should be saved
        max_attempts: Maximum number of download attempts

    Returns:
        True if download was successful, False otherwise
    """
    success = False
    # requests runs into issues when accessing the pdfs, but cloudscraper works
    scraper = cloudscraper.create_scraper()

    # Check there's somewhere to save the pdfs
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    for attempt in range(max_attempts):
        try:
            logger.info(f"Downloading PDF from {pdf_url}")

            # Set headers to mimic a browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Referer": "https://researchbriefings.parliament.uk/",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }

            response = scraper.get(pdf_url, headers=headers, timeout=30)
            response.raise_for_status()

            # Check if the response is actually a PDF
            content_type = response.headers.get("Content-Type", "")
            if "application/pdf" not in content_type:
                logger.warning(f"Response may not be a PDF (Content-Type: {content_type}). Checking file...")
                # Check first few bytes for PDF signature
                if not response.content.startswith(b"%PDF"):
                    logger.warning(f"Downloaded file does not appear to be a valid PDF")

            # Save the PDF
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Successfully downloaded PDF to {pdf_path}")
            success = True
            break

        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/{max_attempts} failed: {str(e)}")
            if attempt < max_attempts - 1:
                # Exponential backoff
                wait_time = 2**attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    return success


def extract_briefing_metadata(full_briefing: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata fields from a briefing JSON response.

    Args:
        full_briefing: The complete briefing data from the API

    Returns:
        Dictionary of extracted metadata fields
    """
    # get briefing data
    briefing_data = full_briefing.get("result", {}).get("primaryTopic", {})

    # Extract fields
    metadata = {
        # extract id from last segment of the URL e.g. 'http://data.parliament.uk/resources/12345'
        "id": briefing_data.get("_about", "").rstrip("/").split("/")[-1],
        "url": briefing_data.get("_about", ""),
        "title": extract_nested_value(briefing_data, ["title"], ""),
        "identifier": extract_nested_value(briefing_data, ["identifier"], ""),
        "abstract": extract_nested_value(briefing_data, ["abstract"], ""),
        "date": extract_nested_value(briefing_data, ["date"], ""),
        "modified": extract_nested_value(briefing_data, ["modified"], ""),
        "type": extract_nested_value(briefing_data, ["type"], "").split("#")[-1],
        "subType": extract_nested_value(briefing_data, ["subType", "prefLabel"], ""),
        "status": briefing_data.get("status", ""),
        "published": extract_nested_value(briefing_data, ["published"], ""),
        "description": extract_nested_value(briefing_data, ["description"], ""),
        "htmlsummary": briefing_data.get("htmlsummary", ""),
        "pdf_url": extract_nested_value(briefing_data, ["contentLocation"], ""),
    }

    # Extract topics
    topics = []
    if "topic" in briefing_data:
        topic_data = briefing_data["topic"]
        if isinstance(topic_data, list):
            for topic in topic_data:
                topic_name = extract_nested_value(topic, ["prefLabel"], "")
                if topic_name:
                    topics.append(topic_name)
        else:
            topic_name = extract_nested_value(topic_data, ["prefLabel"], "")
            if topic_name:
                topics.append(topic_name)
    metadata["topics"] = ", ".join(topics)

    # Extract creator information
    if "creator" in briefing_data:
        creator = briefing_data["creator"]
        if isinstance(creator, list) and creator:
            creator_item = creator[0]
            metadata["creator_given_name"] = extract_nested_value(creator_item, ["givenName"], "")
            metadata["creator_family_name"] = extract_nested_value(creator_item, ["familyName"], "")
        elif isinstance(creator, dict):
            metadata["creator_given_name"] = extract_nested_value(creator, ["givenName"], "")
            metadata["creator_family_name"] = extract_nested_value(creator, ["familyName"], "")

    return metadata


def process_briefings_batch(
    api: ResearchBriefingsAPI,
    briefings_batch: List[Dict[str, Any]],
    output_dir: str,
    pdf_dir: str,
    existing_ids: Set[str],
    use_s3: bool = False,
    s3_handler: Optional[ResearchBriefingsToS3] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Process a batch of briefings: extract metadata to DataFrame and download PDFs.

    Args:
        api: The API client
        briefings_batch: List of briefings metadata in the current batch
        output_dir: Directory to save the PDF files locally
        pdf_dir: Directory to save the PDF files
        existing_ids: Set of already downloaded briefing IDs
        use_s3: Whether to use S3 storage
        s3_handler: S3 handler if use_s3 is True

    Returns:
        Tuple of (DataFrame with briefing metadata, statistics dictionary)
    """
    # List to collect row data for the DataFrame
    rows_data = []

    logger.info(f"Processing {len(briefings_batch)} briefings")

    # Track statistics
    stats = {
        "total_in_batch": len(briefings_batch),
        "metadata_extracted": 0,
        "already_exist": 0,
        "api_errors": 0,
        "pdfs_downloaded": 0,
        "pdfs_already_exist": 0,
        "pdfs_failed": 0,
        "pdfs_no_url": 0,
    }

    for i, briefing in enumerate(briefings_batch):
        logger.info(f"Processing briefing {i+1}/{len(briefings_batch)}")

        # Extract briefing ID
        briefing_id = None

        # Check for _about URL in various formats
        if "_about" in briefing:
            about_url = briefing["_about"]
            briefing_id = about_url.rstrip("/").split("/")[-1]

        if not briefing_id:
            logger.warning(f"Couldn't extract briefing ID from item {i+1}")
            stats["api_errors"] += 1
            continue

        # Check if this briefing has already been downloaded
        if briefing_id in existing_ids:
            logger.info(f"Briefing {briefing_id} already processed, skipping")
            stats["already_exist"] += 1
            continue

        # Get the full briefing data
        full_briefing = api.get_research_briefing_by_id(briefing_id)

        if "error" in full_briefing:
            logger.error(f"Error retrieving briefing {briefing_id}: {full_briefing['error']}")
            stats["api_errors"] += 1
            continue

        # Extract metadata to dictionary
        metadata = extract_briefing_metadata(full_briefing)

        # Add to existing IDs so we don't try to download it again
        existing_ids.add(briefing_id)

        # Download the PDF
        pdf_url = metadata.get("pdf_url")

        if pdf_url:
            identifier = metadata.get("identifier")
            if identifier:
                pdf_filename = f"{identifier}.pdf"
            else:
                pdf_filename = f"{briefing_id}.pdf"

            pdf_filename = clean_filename(pdf_filename)

            # Always create the local PDF directory
            if not os.path.exists(pdf_dir):
                os.makedirs(pdf_dir, exist_ok=True)

            # Local PDF path will be the same regardless of storage mode
            local_pdf_path = os.path.join(pdf_dir, pdf_filename)

            if use_s3:
                # First check if the PDF already exists locally
                if os.path.exists(local_pdf_path):
                    logger.info(f"PDF already exists locally: {pdf_filename}")
                    stats["pdfs_already_exist"] += 1
                    # Set both S3 and local paths in metadata
                    metadata["pdf_file"] = f"s3://{s3_handler.bucket}/{s3_handler.pdfs_prefix}/{pdf_filename}"
                    metadata["local_pdf_file"] = local_pdf_path
                # If not local, check if it exists in S3
                elif s3_handler.check_pdf_exists(pdf_filename):
                    logger.info(f"PDF exists in S3 but not locally, downloading: {pdf_filename}")

                    # Download from S3 to local path
                    try:
                        s3_handler.s3_client.download_file(
                            s3_handler.bucket, f"{s3_handler.pdfs_prefix}/{pdf_filename}", local_pdf_path
                        )
                        logger.info(f"Downloaded PDF from S3 to local path: {local_pdf_path}")
                        stats["pdfs_already_exist"] += 1

                        # Set both paths in metadata
                        metadata["pdf_file"] = f"s3://{s3_handler.bucket}/{s3_handler.pdfs_prefix}/{pdf_filename}"
                        metadata["local_pdf_file"] = local_pdf_path
                    except Exception as e:
                        logger.error(f"Failed to download PDF from S3: {e}")
                        # Try downloading from original source
                        if download_pdf_for_briefing(pdf_url, local_pdf_path):
                            # Upload to S3
                            s3_path = s3_handler.upload_pdf(local_pdf_path, pdf_filename)
                            if s3_path:
                                stats["pdfs_downloaded"] += 1
                                metadata["pdf_file"] = s3_path
                                metadata["local_pdf_file"] = local_pdf_path
                            else:
                                stats["pdfs_failed"] += 1
                        else:
                            stats["pdfs_failed"] += 1
                else:
                    # Neither local nor in S3, download from source
                    if download_pdf_for_briefing(pdf_url, local_pdf_path):
                        # Upload to S3
                        s3_path = s3_handler.upload_pdf(local_pdf_path, pdf_filename)
                        if s3_path:
                            stats["pdfs_downloaded"] += 1
                            metadata["pdf_file"] = s3_path
                            metadata["local_pdf_file"] = local_pdf_path
                        else:
                            stats["pdfs_failed"] += 1
                    else:
                        stats["pdfs_failed"] += 1
            else:
                # Local storage mode (simpler case - just download if needed)
                # Check if PDF already exists locally
                if os.path.exists(local_pdf_path):
                    logger.info(f"PDF already exists locally: {pdf_filename}")
                    stats["pdfs_already_exist"] += 1
                    metadata["pdf_file"] = local_pdf_path
                    # For consistency, set both fields the same in local mode
                    metadata["local_pdf_file"] = local_pdf_path
                else:
                    # Download the PDF
                    if download_pdf_for_briefing(pdf_url, local_pdf_path):
                        stats["pdfs_downloaded"] += 1
                        metadata["pdf_file"] = local_pdf_path
                        metadata["local_pdf_file"] = local_pdf_path
                    else:
                        stats["pdfs_failed"] += 1
        else:
            logger.warning(f"No PDF URL found for briefing {briefing_id}")
            stats["pdfs_no_url"] += 1

        # Add the metadata row to our collection
        rows_data.append(metadata)
        stats["metadata_extracted"] += 1

        # Add a delay between briefings to be nice to the API
        time.sleep(0.5)

    # Log batch statistics
    logger.info(f"Batch processing complete:")
    logger.info(f"  Total briefings in batch: {stats['total_in_batch']}")
    logger.info(f"  Metadata successfully extracted: {stats['metadata_extracted']}")
    logger.info(f"  Briefings already existed: {stats['already_exist']}")
    logger.info(f"  API errors: {stats['api_errors']}")
    logger.info(f"  PDFs successfully downloaded: {stats['pdfs_downloaded']}")
    logger.info(f"  PDFs already existed: {stats['pdfs_already_exist']}")
    logger.info(f"  PDFs failed: {stats['pdfs_failed']}")
    logger.info(f"  Briefings with no PDF URL: {stats['pdfs_no_url']}")

    # Return a DataFrame of the batch results and stats
    return pd.DataFrame(rows_data), stats


def generate_pipeline_artifact(
    metadata: Dict[str, Any],
    output_dir: str,
    run_date: datetime,
    artifact_filename: str = "pipeline_artifact.json",
    use_s3: bool = False,
    s3_handler: Optional[ResearchBriefingsToS3] = None,
    run_dir: Optional[str] = None,
) -> str:
    """
    Generate a pipeline artifact file from the metadata.

    Args:
        metadata: Dictionary with download metadata
        output_dir: Directory to save the artifact
        run_date: Date of the run
        artifact_filename: Name of the artifact file
        use_s3: Whether to use S3 storage
        s3_handler: S3 handler if use_s3 is True
        run_dir: Run directory in S3

    Returns:
        Path to the created artifact file
    """
    # Add timestamp to artifact
    artifact_data = {
        "timestamp": datetime.now().isoformat(),
        "run_date": run_date.isoformat(),
    }

    # Add formatted summary
    artifact_data["summary"] = {
        "total_briefings_found": metadata.get("total_briefings", 0),
        "briefings_processed": metadata.get("metadata_rows", 0),
        "pdfs_downloaded": metadata.get("pdfs_downloaded", 0),
        "date_range": {
            "start": metadata.get("date_range", {}).get("start", ""),
            "end": metadata.get("date_range", {}).get("end", ""),
        },
        "output_files": {
            "metadata_file": os.path.basename(metadata.get("metadata_file", "")),
            "pdf_count": metadata.get("pdfs_downloaded", 0),
        },
        "status": "success" if not metadata.get("error") else "error",
        "error_message": metadata.get("error", ""),
        "storage_mode": "s3" if use_s3 else "local",
    }

    if use_s3:
        # Upload artifact to S3 runs directory
        artifact_path = s3_handler.upload_artifact(artifact_data, run_dir)
    else:
        # Save artifact locally
        artifact_path = os.path.join(output_dir, artifact_filename)

        # Create directory if needed
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact_data, f, indent=2)

        logger.info(f"Generated pipeline artifact at {artifact_path}")

    return artifact_path


def download_research_briefings(
    output_dir: str = "research_briefings",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    batch_size: int = 50,
    use_s3: bool = False,
    s3_prefix: str = None,
    update_cumulative: bool = True,
) -> Dict[str, Any]:
    """
    Download research briefings matching criteria, including PDFs.

    Args:
        output_dir: Directory to save the briefings data
        start_date: Minimum date for filtering (defaults to 30 days ago)
        end_date: Maximum date for filtering (defaults to now)
        batch_size: Number of briefings to process in each batch
        use_s3: Whether to use S3 storage
        s3_prefix: S3 prefix (folder path) if use_s3 is True
        update_cumulative: Whether to update the cumulative file with new data

    Returns:
        Dictionary with metadata about the download
    """
    # Set the run date (used for file naming)
    run_date = datetime.now()

    # Create S3 handler if using S3
    s3_handler = None
    run_dir = None
    if use_s3:
        s3_handler = ResearchBriefingsToS3(s3_prefix)
        run_dir = s3_handler.create_run_directory(run_date)
        logger.info(f"Created run directory: {run_dir}")

    # Always create local output directory, regardless of storage mode
    os.makedirs(output_dir, exist_ok=True)

    # Always create PDF directory for local copies, regardless of storage mode
    pdf_dir = os.path.join(output_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    logger.info(f"Local PDF storage directory: {pdf_dir}")

    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now()

    if not start_date:
        # Default to 30 days before end date
        start_date = end_date - timedelta(days=30)

    logger.info(f"Downloading research briefings from {start_date.date()} to {end_date.date()}")
    if use_s3:
        logger.info(f"Using S3 storage: s3://{BUCKET_NAME_RAW}/{s3_prefix}")
    else:
        logger.info(f"Using local storage: {output_dir}")

    # Initialise API client
    api = ResearchBriefingsAPI()

    # Get all briefings matching criteria
    try:
        briefings = api.get_all_research_briefings(start_date=start_date, end_date=end_date)
    except Exception as e:
        logger.error(f"Error retrieving briefings: {e}")
        error_metadata = {
            "download_date": run_date.isoformat(),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_briefings": 0,
            "metadata_rows": 0,
            "pdfs_downloaded": 0,
            "error": str(e),
        }
        # Generate pipeline artifact for the error case
        generate_pipeline_artifact(
            error_metadata, output_dir, run_date, use_s3=use_s3, s3_handler=s3_handler, run_dir=run_dir
        )
        return error_metadata

    logger.info(f"Found {len(briefings)} research briefings in total")

    if not briefings:
        logger.warning("No briefings found matching the criteria")
        no_data_metadata = {
            "download_date": run_date.isoformat(),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_briefings": 0,
            "metadata_rows": 0,
            "pdfs_downloaded": 0,
            "status": "success",
        }
        # Generate pipeline artifact for the no data case
        generate_pipeline_artifact(
            no_data_metadata, output_dir, run_date, use_s3=use_s3, s3_handler=s3_handler, run_dir=run_dir
        )
        return no_data_metadata

    # Get existing briefing IDs to avoid re-downloading
    if use_s3:
        existing_ids = s3_handler.get_existing_briefing_ids()
    else:
        existing_ids = get_existing_briefing_ids_local(output_dir)

    # Split briefings into batches
    briefings_batches = [briefings[i : i + batch_size] for i in range(0, len(briefings), batch_size)]

    # Process all batches
    all_dfs = []
    all_stats = []

    for i, batch in enumerate(briefings_batches):
        logger.info(f"Processing batch {i+1}/{len(briefings_batches)}")
        batch_df, batch_stats = process_briefings_batch(
            api=api,
            briefings_batch=batch,
            output_dir=output_dir,
            pdf_dir=pdf_dir,
            existing_ids=existing_ids,
            use_s3=use_s3,
            s3_handler=s3_handler,
        )

        if not batch_df.empty:
            all_dfs.append(batch_df)
            all_stats.append(batch_stats)

    # Combine all batches into a single DataFrame
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined {len(all_dfs)} batches into DataFrame with {len(combined_df)} rows")

        # Calculate aggregate statistics
        pdfs_downloaded = sum(stats.get("pdfs_downloaded", 0) for stats in all_stats)
        pdfs_already_exist = sum(stats.get("pdfs_already_exist", 0) for stats in all_stats)
        total_pdfs = pdfs_downloaded + pdfs_already_exist

        # Save the DataFrame both locally and to S3 if applicable
        local_metadata_file = os.path.join(output_dir, f"research_briefings.parquet")

        try:
            # Always save locally first
            combined_df.to_parquet(local_metadata_file, index=False)
            logger.info(f"Saved DataFrame locally to {local_metadata_file}")

            if use_s3:
                # Also upload to S3
                s3_metadata_file = s3_handler.upload_run_data(combined_df, run_dir)

                # For tracking, use the S3 path in metadata
                metadata_file = s3_metadata_file

                # Update cumulative file if requested
                if update_cumulative and s3_metadata_file:
                    s3_handler.update_cumulative_file(combined_df)
            else:
                # For local-only mode, use the local path
                metadata_file = local_metadata_file
        except Exception as e:
            logger.error(f"Error saving DataFrame file: {str(e)}")
            metadata_file = None

        # Save summary metadata about the download
        metadata = {
            "download_date": run_date.isoformat(),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_briefings": len(briefings),
            "metadata_rows": len(combined_df),
            "pdfs_downloaded": int(pdfs_downloaded),
            "pdfs_already_exist": int(pdfs_already_exist),
            "total_pdfs": int(total_pdfs),
            "metadata_file": metadata_file,
            "status": "success",
            "storage_mode": "s3" if use_s3 else "local",
        }

        # Generate pipeline artifact
        generate_pipeline_artifact(
            metadata, output_dir, run_date, use_s3=use_s3, s3_handler=s3_handler, run_dir=run_dir
        )

        logger.info(f"Download complete. Found {len(briefings)} briefings")
        logger.info(f"Processed {len(combined_df)} briefings metadata")
        logger.info(f"Downloaded {pdfs_downloaded} new PDF files")
        logger.info(f"Found {pdfs_already_exist} existing PDF files")
        logger.info(f"Total PDFs: {total_pdfs}")

        return metadata
    else:
        logger.warning("No briefings were processed successfully")
        no_success_metadata = {
            "download_date": run_date.isoformat(),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_briefings": len(briefings),
            "metadata_rows": 0,
            "pdfs_downloaded": 0,
            "error": "No briefings processed successfully",
        }

        # Generate pipeline artifact for the no success case
        generate_pipeline_artifact(
            no_success_metadata, output_dir, run_date, use_s3=use_s3, s3_handler=s3_handler, run_dir=run_dir
        )

        return no_success_metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download UK Parliament Research Briefings")
    parser.add_argument(
        "--output", "-o", default="outputs/research_briefings", help="Output directory for downloaded briefings"
    )
    parser.add_argument("--days", "-d", type=int, default=7, help="Number of days to look back for briefings")
    parser.add_argument("--start-date", "-s", help="Start date in YYYY-MM-DD format (overrides days parameter)")
    parser.add_argument("--end-date", "-e", help="End date in YYYY-MM-DD format (defaults to today)")
    parser.add_argument(
        "--batch-size", "-b", type=int, default=50, help="Number of briefings to process in each batch"
    )
    parser.add_argument("--use-s3", action="store_true", help="Store files in S3 instead of locally")
    parser.add_argument(
        "--s3-prefix",
        default="data/policy/research_briefings",
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

    # Download briefings
    download_research_briefings(
        output_dir=args.output,
        start_date=start_date,
        end_date=end_date,
        batch_size=args.batch_size,
        use_s3=args.use_s3,
        s3_prefix=args.s3_prefix,
        update_cumulative=not args.no_update_cumulative,
    )
