"""
Research Briefings Getter

This script downloads research briefings data from the Parliament Open Data API,
including both the JSON metadata and PDF documents. It stores the metadata in
a DataFrame and downloads PDFs as separate files.
"""

import glob
import json
import logging
import os
import re
import sys
import time

from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import pandas as pd
import requests


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
                logger.info(f"Found briefing URL: {data['result']['_about']}")

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


def get_existing_briefing_ids(output_dir: str) -> Set[str]:
    """
    Check if a metadata CSV exists and extract briefing IDs from it.

    Args:
        output_dir: Directory to check for existing metadata CSV

    Returns:
        Set of briefing IDs that have already been downloaded
    """
    existing_ids = set()

    # Look for metadata CSV files
    csv_files = glob.glob(os.path.join(output_dir, "research_briefings_*.csv"))

    if not csv_files:
        logger.info(f"No existing metadata CSV found in {output_dir}")
        return existing_ids

    # Use the most recent CSV file
    latest_csv = max(csv_files, key=os.path.getmtime)
    logger.info(f"Found existing metadata CSV: {latest_csv}")

    try:
        # Read the CSV file
        df = pd.read_csv(latest_csv)

        # Extract IDs
        if "id" in df.columns:
            existing_ids = set(df["id"].dropna().astype(str))
            logger.info(f"Found {len(existing_ids)} existing briefing IDs in {latest_csv}")
    except Exception as e:
        logger.warning(f"Error processing existing CSV file {latest_csv}: {e}")

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

    logger.info(f"Total {len(existing_ids)} existing briefing IDs found")
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

    for attempt in range(max_attempts):
        try:
            logger.info(f"Downloading PDF from {pdf_url}")

            # Set headers to mimic a browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }

            # Set a timeout and use headers to prevent 403 errors
            response = requests.get(pdf_url, headers=headers, timeout=30)
            response.raise_for_status()

            # Check if the response is actually a PDF
            content_type = response.headers.get("Content-Type", "")
            if "application/pdf" not in content_type and not pdf_url.endswith(".pdf"):
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
    pdf_dir: str,
    existing_ids: Set[str],
    checkpoint_file: str,
    batch_index: int = 0,
    total_batches: int = 1,
) -> pd.DataFrame:
    """
    Process a batch of briefings: extract metadata to DataFrame and download PDFs.

    Args:
        api: The API client
        briefings_batch: List of briefings metadata in the current batch
        pdf_dir: Directory to save the PDF files
        existing_ids: Set of already downloaded briefing IDs
        checkpoint_file: Path to save checkpoint data
        batch_index: Index of the current batch
        total_batches: Total number of batches

    Returns:
        DataFrame with briefing metadata
    """
    # List to collect row data for the DataFrame
    rows_data = []

    logger.info(f"Processing batch {batch_index+1}/{total_batches} with {len(briefings_batch)} briefings")

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
        logger.info(f"Processing briefing {i+1}/{len(briefings_batch)} in batch {batch_index+1}/{total_batches}")

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
            pdf_path = os.path.join(pdf_dir, pdf_filename)

            # Check if PDF already exists
            if os.path.exists(pdf_path):
                logger.info(f"PDF already exists: {pdf_filename}")
                stats["pdfs_already_exist"] += 1
                metadata["pdf_file"] = pdf_path  # Add path to existing PDF
            else:
                # Download the PDF
                if download_pdf_for_briefing(pdf_url, pdf_path):
                    stats["pdfs_downloaded"] += 1
                    metadata["pdf_file"] = pdf_path  # Add path to new PDF
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

        # Save checkpoint after each briefing
        if checkpoint_file and rows_data:
            try:
                # Create a DataFrame from the rows processed so far
                checkpoint_df = pd.DataFrame(rows_data)

                # Save directly to checkpoint CSV file
                checkpoint_df.to_csv(checkpoint_file, index=False)

                # Also save a stats file alongside it
                stats_file = f"{checkpoint_file}.stats.json"
                stats_data = {
                    "stats": stats,
                    "last_processed_index": i,
                    "batch_index": batch_index,
                    "timestamp": datetime.now().isoformat(),
                }

                with open(stats_file, "w", encoding="utf-8") as f:
                    json.dump(stats_data, f, indent=2)

                logger.debug(f"Updated checkpoint at {checkpoint_file}")
            except Exception as e:
                logger.warning(f"Error saving checkpoint: {str(e)}")

    # Log batch statistics
    logger.info(f"Batch {batch_index+1}/{total_batches} processing complete:")
    logger.info(f"  Total briefings in batch: {stats['total_in_batch']}")
    logger.info(f"  Metadata successfully extracted: {stats['metadata_extracted']}")
    logger.info(f"  Briefings already existed: {stats['already_exist']}")
    logger.info(f"  API errors: {stats['api_errors']}")
    logger.info(f"  PDFs successfully downloaded: {stats['pdfs_downloaded']}")
    logger.info(f"  PDFs already existed: {stats['pdfs_already_exist']}")
    logger.info(f"  PDFs failed: {stats['pdfs_failed']}")
    logger.info(f"  Briefings with no PDF URL: {stats['pdfs_no_url']}")

    # Return a DataFrame of the batch results
    return pd.DataFrame(rows_data)


def generate_pipeline_artifact(
    metadata: Dict[str, Any], output_dir: str, artifact_filename: str = "pipeline_artifact.json"
) -> str:
    """
    Generate a pipeline artifact file from the metadata.

    Args:
        metadata: Dictionary with download metadata
        output_dir: Directory to save the artifact
        artifact_filename: Name of the artifact file

    Returns:
        Path to the created artifact file
    """
    artifact_path = os.path.join(output_dir, artifact_filename)

    # Add timestamp to artifact
    artifact_data = {
        "timestamp": datetime.now().isoformat(),
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
            "metadata_csv": os.path.basename(metadata.get("metadata_file", "")),
            "pdf_count": metadata.get("pdfs_downloaded", 0),
        },
        "status": "success" if not metadata.get("error") else "error",
        "error_message": metadata.get("error", ""),
    }

    # Write to file
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact_data, f, indent=2)

    logger.info(f"Generated pipeline artifact at {artifact_path}")
    return artifact_path


def download_research_briefings(
    output_dir: str = "research_briefings",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    batch_size: int = 50,
    resume_from_checkpoint: bool = False,
) -> Dict[str, Any]:
    """
    Download research briefings matching criteria, including PDFs.

    Args:
        output_dir: Directory to save the briefings data
        start_date: Minimum date for filtering (defaults to 30 days ago)
        end_date: Maximum date for filtering (defaults to now)
        batch_size: Number of briefings to process in each batch
        resume_from_checkpoint: Whether to try resuming from a checkpoint

    Returns:
        Dictionary with metadata about the download
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create PDF directory
    pdf_dir = os.path.join(output_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now()

    if not start_date:
        # Default to 30 days before end date
        start_date = end_date - timedelta(days=30)

    logger.info(f"Downloading research briefings from {start_date.date()} to {end_date.date()}")

    # Initialise API client
    api = ResearchBriefingsAPI()

    # Create checkpoint file path
    checkpoint_file = os.path.join(
        output_dir, f"checkpoint_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    )

    # Try to load checkpoint if requested
    start_batch = 0
    batch_dataframes = []

    if resume_from_checkpoint and os.path.exists(checkpoint_file):
        try:
            # Load the checkpoint CSV directly
            df = pd.read_csv(checkpoint_file)
            batch_dataframes.append(df)
            logger.info(f"Loaded {len(df)} processed briefings from checkpoint CSV")

            # Look for the stats file to get batch index
            stats_file = f"{checkpoint_file}.stats.json"
            if os.path.exists(stats_file):
                with open(stats_file, "r", encoding="utf-8") as f:
                    stats_data = json.load(f)
                    if "batch_index" in stats_data:
                        start_batch = stats_data["batch_index"] + 1
                        logger.info(f"Resuming from batch {start_batch}")
        except Exception as e:
            logger.warning(f"Error loading checkpoint, starting from beginning: {str(e)}")
            start_batch = 0
            batch_dataframes = []

    # Get all briefings matching criteria
    try:
        briefings = api.get_all_research_briefings(start_date=start_date, end_date=end_date)
    except Exception as e:
        logger.error(f"Error retrieving briefings: {e}")
        error_metadata = {
            "download_date": datetime.now().isoformat(),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_briefings": 0,
            "metadata_rows": 0,
            "pdfs_downloaded": 0,
            "error": str(e),
        }
        # Generate pipeline artifact for the error case
        generate_pipeline_artifact(error_metadata, output_dir)
        return error_metadata

    logger.info(f"Found {len(briefings)} research briefings in total")

    if not briefings:
        logger.warning("No briefings found matching the criteria")
        no_data_metadata = {
            "download_date": datetime.now().isoformat(),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_briefings": 0,
            "metadata_rows": 0,
            "pdfs_downloaded": 0,
            "status": "success",
        }
        # Generate pipeline artifact for the no data case
        generate_pipeline_artifact(no_data_metadata, output_dir)
        return no_data_metadata

    # Get existing briefing IDs to avoid re-downloading
    existing_ids = get_existing_briefing_ids(output_dir)

    # Split briefings into batches
    briefings_batches = [briefings[i : i + batch_size] for i in range(0, len(briefings), batch_size)]
    total_batches = len(briefings_batches)

    logger.info(f"Processing {len(briefings)} briefings in {total_batches} batches of {batch_size}")

    # Process each batch, starting from the checkpoint if available
    for batch_index in range(start_batch, total_batches):
        batch = briefings_batches[batch_index]

        batch_df = process_briefings_batch(
            api=api,
            briefings_batch=batch,
            pdf_dir=pdf_dir,
            existing_ids=existing_ids,
            checkpoint_file=checkpoint_file,
            batch_index=batch_index,
            total_batches=total_batches,
        )

        # Add batch DataFrame to our collection
        if not batch_df.empty:
            batch_dataframes.append(batch_df)

        # Save a batch-specific checkpoint
        batch_csv_file = os.path.join(
            output_dir, f"batch_{batch_index}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        )
        try:
            batch_df.to_csv(batch_csv_file, index=False)
            logger.info(f"Saved batch DataFrame to {batch_csv_file}")
        except Exception as e:
            logger.warning(f"Error saving batch DataFrame: {str(e)}")

    # Combine all batches into a single DataFrame
    if batch_dataframes:
        combined_df = pd.concat(batch_dataframes, ignore_index=True)
        logger.info(f"Combined {len(batch_dataframes)} batches into DataFrame with {len(combined_df)} rows")

        # Save the complete DataFrame
        metadata_file = os.path.join(
            output_dir, f"research_briefings_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        )

        try:
            combined_df.to_parquet(metadata_file, index=False)
            logger.info(f"Saved complete DataFrame to {metadata_file}")
        except Exception as e:
            logger.error(f"Error saving DataFrame file: {str(e)}")

        # Clean up checkpoint and batch files
        logger.info("Cleaning up temporary files...")
        files_to_clean = []

        # Find checkpoint files
        checkpoint_pattern = os.path.join(output_dir, f"checkpoint_*.csv")
        files_to_clean.extend(glob.glob(checkpoint_pattern))

        # Find stats files
        stats_pattern = os.path.join(output_dir, f"*.stats.json")
        files_to_clean.extend(glob.glob(stats_pattern))

        # Find batch files
        batch_pattern = os.path.join(
            output_dir, f"batch_*_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        )
        files_to_clean.extend(glob.glob(batch_pattern))

        # Delete the files
        for file_path in files_to_clean:
            try:
                os.remove(file_path)
                logger.debug(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Error removing temporary file {file_path}: {str(e)}")

        logger.info(f"Cleanup complete. Removed {len(files_to_clean)} temporary files")

        # Count successful downloads
        metadata_rows = len(combined_df)
        pdfs_downloaded = combined_df["pdf_file"].notna().sum()

        # Save summary metadata about the download
        metadata = {
            "download_date": datetime.now().isoformat(),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_briefings": len(briefings),
            "metadata_rows": metadata_rows,
            "pdfs_downloaded": int(pdfs_downloaded),
            "metadata_file": metadata_file,
            "status": "success",
        }

        # Generate pipeline artifact
        generate_pipeline_artifact(metadata, output_dir)

        logger.info(f"Download complete. Found {len(briefings)} briefings")
        logger.info(f"Processed {metadata_rows} briefings metadata")
        logger.info(f"Downloaded {pdfs_downloaded} PDF files")

        return metadata
    else:
        logger.warning("No briefings were processed successfully")
        no_success_metadata = {
            "download_date": datetime.now().isoformat(),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_briefings": len(briefings),
            "metadata_rows": 0,
            "pdfs_downloaded": 0,
            "error": "No briefings processed successfully",
        }

        # Generate pipeline artifact for the no success case
        generate_pipeline_artifact(no_success_metadata, output_dir)

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
    parser.add_argument(
        "--resume", "-r", action="store_true", help="Try to resume a previous download using checkpoint files"
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
        resume_from_checkpoint=args.resume,
    )
