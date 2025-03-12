"""
UK Parliament Research Briefings Getter

This script downloads research briefings data from the UK Parliament API.
It provides functionality to retrieve briefings, filter by date range, and
save the JSON response data.
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
from typing import Union

import requests

from discovery_utils.getters.research_briefings_pdf import download_pdfs_for_briefings


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("research_briefings_download.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ResearchBriefingsAPI:
    """Client for interacting with the UK Parliament Research Briefings API."""

    def __init__(self, base_url="https://lda.data.parliament.uk"):
        """Initialize the API client with the base URL."""
        self.base_url = base_url
        self.session = requests.Session()
        # Set a user agent to be polite
        self.session.headers.update({"User-Agent": "ResearchBriefingsDownloader/1.0", "Accept": "application/json"})
        # Set default timeout for all requests
        self.timeout = 30  # 30 second timeout

    def get_research_briefings(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        topic: Optional[str] = None,
        type_id: Optional[str] = None,
        search_term: Optional[str] = None,
        page: int = 0,
        page_size: int = 500,
        format: str = "json",
    ) -> Dict:
        """
        Get a list of research briefings matching the specified criteria.

        Args:
            start_date: Minimum date for the 'date' field
            end_date: Maximum date for the 'date' field
            topic: Filter by specific topic
            type_id: Filter by specific briefing type
            search_term: Text search across briefings
            page: Page number for pagination
            page_size: Number of results per page
            format: Response format (json)

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

        # Add other filters if provided
        if topic:
            params["topic"] = topic
        if type_id:
            params["type"] = type_id
        if search_term:
            params["_search"] = search_term

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
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        topic: Optional[str] = None,
        type_id: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get all research briefings matching criteria, handling pagination.

        Args:
            start_date: Minimum date for the 'date' field
            end_date: Maximum date for the 'date' field
            topic: Filter by specific topic
            type_id: Filter by specific briefing type
            search_term: Text search across briefings

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
                start_date=start_date,
                end_date=end_date,
                topic=topic,
                type_id=type_id,
                search_term=search_term,
                page=page,
                page_size=page_size,
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

            # Move to next page
            page += 1
            logger.info(f"Fetched page {page} of {total_pages}")

            # Be nice to the API
            if page < total_pages:
                time.sleep(0.5)

        return all_briefings

    def get_research_briefing_by_id(self, briefing_id: str) -> Dict:
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

    def get_topics(self) -> Dict:
        """
        Get all available research briefing topics.

        Returns:
            Dictionary containing topic data
        """
        url = f"{self.base_url}/researchbriefingtopics"
        try:
            response = self.session.get(f"{url}.json", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving topics: {e}")
            return {"error": str(e)}

    def get_types(self) -> Dict:
        """
        Get all available research briefing types.

        Returns:
            Dictionary containing briefing type data
        """
        url = f"{self.base_url}/researchbriefingtypes"
        try:
            response = self.session.get(f"{url}.json", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving briefing types: {e}")
            return {"error": str(e)}


def extract_briefing_id_from_url(url: str) -> str:
    """
    Extract the briefing ID from a URL.

    Args:
        url: The URL to extract from

    Returns:
        The extracted briefing ID
    """
    # Try to extract the ID from the URL
    if not url:
        return ""

    # Different formats to try
    patterns = [
        r"resources/(\d+)",  # For URLs like 'http://data.parliament.uk/resources/12345'
        r"researchbriefings/(\d+)",  # For URLs like 'http://eldaddp.azurewebsites.net/researchbriefings/12345'
        r"/([^/]+)$",  # Last segment of the URL as a fallback
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # If no pattern matches, use the last segment of the URL
    return url.rstrip("/").split("/")[-1]


def extract_nested_value(data: Dict, path: List[str], default=None) -> Any:
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
    return current


def extract_field_value(data: Any) -> Any:
    """
    Extract the actual value from a field which might be wrapped in API-specific structures.

    Args:
        data: The field value to extract from

    Returns:
        The extracted actual value
    """
    # Handle different data types and structures
    if data is None:
        return None

    # If it's a list, process the first item
    if isinstance(data, list):
        if not data:
            return None
        return extract_field_value(data[0])

    # If it's a dictionary with _value key, extract that
    if isinstance(data, dict):
        if "_value" in data:
            return data["_value"]
        # For resource objects, try to get label or prefLabel
        if "label" in data:
            return extract_field_value(data["label"])
        if "prefLabel" in data:
            return extract_field_value(data["prefLabel"])

    # Otherwise return the data as is
    return data


def extract_briefing_metadata(briefing_data: Dict) -> Dict:
    """
    Extract metadata from a briefing response.

    Args:
        briefing_data: The raw API response for a briefing

    Returns:
        Dictionary with extracted metadata
    """
    # Navigate to the primary topic
    primary_topic = extract_nested_value(briefing_data, ["result", "primaryTopic"], {})

    # Extract the about URL
    about_url = extract_nested_value(briefing_data, ["result", "_about"], "")

    # Get the briefing ID from the URL
    briefing_id = extract_briefing_id_from_url(about_url)

    # Base metadata
    metadata = {
        "id": briefing_id,
        "url": about_url,
        "title": extract_field_value(primary_topic.get("title")),
        "identifier": extract_field_value(primary_topic.get("identifier")),
        "abstract": extract_field_value(primary_topic.get("abstract")),
        "description": extract_field_value(primary_topic.get("description")),
        "htmlsummary": primary_topic.get("htmlsummary"),
        "date": extract_field_value(primary_topic.get("date")),
        "modified": extract_field_value(primary_topic.get("modified")),
        "status": primary_topic.get("status"),
        "published": extract_field_value(primary_topic.get("published")),
    }

    # Extract topics
    topics = []
    if "topic" in primary_topic:
        topic_data = primary_topic["topic"]
        if isinstance(topic_data, list):
            for topic in topic_data:
                topic_name = extract_field_value(topic)
                if topic_name:
                    topics.append(topic_name)
        else:
            topic_name = extract_field_value(topic_data)
            if topic_name:
                topics.append(topic_name)

    metadata["topics"] = topics

    # Extract type
    if "type" in primary_topic:
        metadata["type"] = extract_field_value(primary_topic["type"])
    if "subType" in primary_topic:
        metadata["subType"] = extract_field_value(primary_topic["subType"])

    # Extract creator information
    if "creator" in primary_topic:
        creator = primary_topic["creator"]
        # Handle creator being a list or a dictionary
        if isinstance(creator, list):
            # Use the first creator if there are multiple
            if creator:
                creator_item = creator[0]
                creator_info = {
                    "name": extract_field_value(
                        creator_item.get("fullName") if isinstance(creator_item, dict) else None
                    ),
                    "givenName": extract_field_value(
                        creator_item.get("givenName") if isinstance(creator_item, dict) else None
                    ),
                    "familyName": extract_field_value(
                        creator_item.get("familyName") if isinstance(creator_item, dict) else None
                    ),
                }
                metadata["creator"] = creator_info
        elif isinstance(creator, dict):
            creator_info = {
                "name": extract_field_value(creator.get("fullName")),
                "givenName": extract_field_value(creator.get("givenName")),
                "familyName": extract_field_value(creator.get("familyName")),
            }
            metadata["creator"] = creator_info

    return metadata


def get_existing_briefing_ids(output_dir: str) -> Set[str]:
    """
    Scan the output directory for existing JSON files and extract briefing IDs.

    Args:
        output_dir: Directory to scan for existing briefings

    Returns:
        Set of briefing IDs that have already been downloaded
    """
    existing_ids = set()

    # Find all JSON files in the output directory
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {output_dir}")

    # Count for progress reporting
    processed_files = 0
    total_files = len(json_files)

    for json_file in json_files:
        processed_files += 1

        # Log progress periodically
        if processed_files % 100 == 0 or processed_files == total_files:
            logger.info(
                f"Scanning existing files: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%)"
            )

        # Skip metadata and checkpoint files
        if any(skip_term in os.path.basename(json_file) for skip_term in ["metadata", "checkpoint", "raw_briefings"]):
            continue

        try:
            # Read the JSON file
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract the _about URL from the primaryTopic
            about_url = extract_nested_value(data, ["result", "primaryTopic", "_about"])

            if about_url:
                # Extract the ID from the URL
                briefing_id = extract_briefing_id_from_url(about_url)
                if briefing_id:
                    existing_ids.add(briefing_id)

        except Exception as e:
            logger.warning(f"Error processing existing file {json_file}: {e}")

    logger.info(f"Found {len(existing_ids)} existing briefing IDs in {output_dir}")
    return existing_ids


def find_existing_file_for_briefing(briefing_id: str, output_dir: str) -> Optional[str]:
    """
    Find the existing JSON file for a briefing ID.

    Args:
        briefing_id: The briefing ID to find
        output_dir: Directory to search in

    Returns:
        Path to the existing file or None if not found
    """
    # Find all JSON files in the output directory
    json_files = glob.glob(os.path.join(output_dir, "*.json"))

    for json_file in json_files:
        # Skip metadata and checkpoint files
        if any(skip_term in os.path.basename(json_file) for skip_term in ["metadata", "checkpoint", "raw_briefings"]):
            continue

        try:
            # Read the JSON file
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract the _about URL from the primaryTopic
            about_url = extract_nested_value(data, ["result", "primaryTopic", "_about"])

            if about_url:
                # Extract the ID from the URL
                file_briefing_id = extract_briefing_id_from_url(about_url)
                if file_briefing_id == briefing_id:
                    return json_file

        except Exception as e:
            logger.debug(f"Error checking file {json_file} for briefing ID {briefing_id}: {e}")

    return None


def process_briefings(api: ResearchBriefingsAPI, briefings: List[Dict], output_dir: str) -> List[Dict]:
    """
    Process each briefing and save the JSON data.

    Args:
        api: The API client
        briefings: List of briefings metadata
        output_dir: Directory to save the JSON files

    Returns:
        List of briefings with updated metadata
    """
    # Get existing briefing IDs
    existing_ids = get_existing_briefing_ids(output_dir)

    processed_briefings = []
    total_briefings = len(briefings)

    for i, briefing in enumerate(briefings):
        logger.info(f"Processing briefing {i+1}/{total_briefings} ({(i+1)/total_briefings*100:.1f}%)")

        # Extract metadata from the briefing data
        # First try if this is already a full briefing response
        if "result" in briefing and "primaryTopic" in briefing["result"]:
            metadata = extract_briefing_metadata(briefing)
            full_briefing = briefing

            # Get the ID from the _about URL
            about_url = extract_nested_value(briefing, ["result", "primaryTopic", "_about"], "")
            briefing_id = extract_briefing_id_from_url(about_url) if about_url else None
        else:
            # Otherwise try to extract the briefing ID
            briefing_id = None

            # Check for _about URL in various formats
            if "_about" in briefing:
                about_url = briefing["_about"]
                briefing_id = extract_briefing_id_from_url(about_url)

            # If we don't have an ID yet, check for an ID field
            if not briefing_id and "id" in briefing:
                briefing_id = briefing["id"]

            if not briefing_id:
                logger.warning(f"Couldn't extract briefing ID from item {i+1}")
                continue

            # Check if this briefing has already been downloaded
            if briefing_id in existing_ids:
                logger.info(f"Briefing {briefing_id} already downloaded, checking existing file...")

                # Try to find the existing file for this briefing
                existing_file = find_existing_file_for_briefing(briefing_id, output_dir)

                if existing_file:
                    logger.info(f"Using existing file for briefing {briefing_id}: {existing_file}")
                    # Add metadata for the existing file
                    try:
                        with open(existing_file, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)

                        metadata = extract_briefing_metadata(existing_data)
                        metadata["json_file"] = existing_file
                        processed_briefings.append(metadata)
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing existing file for {briefing_id}: {e}")
                        # If we can't process the existing file, we'll re-download it
                else:
                    logger.warning(f"Briefing ID {briefing_id} in existing_ids but no file found, will re-download")

            # Get the full briefing data with retry mechanism
            max_retries = 3
            retry_delay = 5  # seconds
            full_briefing = None

            for retry in range(max_retries):
                try:
                    full_briefing = api.get_research_briefing_by_id(briefing_id)
                    if "error" in full_briefing:
                        logger.warning(
                            f"Error retrieving briefing {briefing_id} on attempt {retry+1}: {full_briefing['error']}"
                        )
                        if retry < max_retries - 1:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        continue
                    break  # Success, exit retry loop
                except Exception as e:
                    logger.error(f"Unexpected error on attempt {retry+1}: {e}")
                    if retry < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff

            if not full_briefing or "error" in full_briefing:
                logger.error(f"Failed to retrieve briefing {briefing_id} after {max_retries} attempts")
                # Create a minimal metadata record with the error
                metadata = {
                    "id": briefing_id,
                    "error": full_briefing.get("error", "Unknown error") if full_briefing else "Failed to retrieve",
                }
                # Add this to processed briefings but continue to next briefing
                processed_briefings.append(metadata)
                continue

            metadata = extract_briefing_metadata(full_briefing)

        # Create a sanitized filename
        title = metadata.get("title", "Unknown")
        identifier = metadata.get("identifier", "")
        date = metadata.get("date", "")
        date_part = date.split("T")[0] if date and "T" in date else ""

        # Use identifier or id for the filename
        ref_part = identifier if identifier else metadata.get("id", "")

        # Create a base filename
        filename_base = clean_filename(f"{date_part}_{ref_part}_{title}")

        logger.info(f"Processing briefing: {title}")

        # Save the full JSON response
        json_path = os.path.join(output_dir, f"{filename_base}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(full_briefing, f, indent=2)

            # Add the file path to the metadata
            metadata["json_file"] = json_path

            # Add to existing IDs so we don't try to download it again in this session
            if briefing_id:
                existing_ids.add(briefing_id)
        except Exception as e:
            logger.error(f"Error saving JSON file for briefing {metadata.get('id')}: {e}")
            metadata["error_saving"] = str(e)

        processed_briefings.append(metadata)

        # Be nice to the API with an adaptive delay
        # If we're processing many briefings, use a longer delay
        if total_briefings > 1000:
            time.sleep(1.0)  # 1 second for very large batches
        elif total_briefings > 100:
            time.sleep(0.7)  # 0.7 seconds for large batches
        else:
            time.sleep(0.5)  # 0.5 seconds for small batches

    return processed_briefings


def download_research_briefings(
    output_dir: str = "research_briefings",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    topic: Optional[str] = None,
    search_term: Optional[str] = None,
) -> Dict:
    """
    Download research briefings matching criteria.

    Args:
        output_dir: Directory to save the briefings data
        start_date: Minimum date for filtering (defaults to 30 days ago)
        end_date: Maximum date for filtering (defaults to now)
        topic: Filter by specific topic
        search_term: Text search across briefings

    Returns:
        Dictionary with metadata about the download
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now()

    if not start_date:
        # Default to 30 days before end date
        start_date = end_date - timedelta(days=30)

    logger.info(f"Downloading research briefings from {start_date.date()} to {end_date.date()}")

    # Initialize API client
    api = ResearchBriefingsAPI()

    # For very large time windows, split into smaller chunks (e.g., monthly)
    max_chunk_days = 90  # 3 months at a time
    total_days = (end_date - start_date).days

    if total_days > max_chunk_days:
        logger.info(f"Large date range detected ({total_days} days). Processing in chunks of {max_chunk_days} days.")
        all_briefings = []

        # Process in chunks
        chunk_start = start_date
        while chunk_start < end_date:
            # Calculate chunk end date (either max_chunk_days ahead or end_date, whichever is earlier)
            chunk_end = min(chunk_start + timedelta(days=max_chunk_days), end_date)

            logger.info(f"Processing chunk from {chunk_start.date()} to {chunk_end.date()}")

            # Get briefings for this chunk
            try:
                chunk_briefings = api.get_all_research_briefings(
                    start_date=chunk_start, end_date=chunk_end, topic=topic, search_term=search_term
                )

                logger.info(f"Found {len(chunk_briefings)} briefings in this chunk")
                all_briefings.extend(chunk_briefings)

                # Check if we should save a checkpoint for this chunk
                if len(all_briefings) > 0:
                    checkpoint_file = os.path.join(
                        output_dir, f"checkpoint_{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}.json"
                    )
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(chunk_briefings, f, indent=2)
                    logger.info(f"Saved checkpoint to {checkpoint_file}")

                # Move to next chunk
                chunk_start = chunk_end + timedelta(days=1)

                # Brief delay between chunks
                time.sleep(2)

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_start.date()} to {chunk_end.date()}: {e}")
                # Try to recover by moving to the next chunk
                chunk_start = chunk_end + timedelta(days=1)

        briefings = all_briefings
    else:
        # Get all briefings matching criteria in one go for smaller time windows
        try:
            briefings = api.get_all_research_briefings(
                start_date=start_date, end_date=end_date, topic=topic, search_term=search_term
            )
        except Exception as e:
            logger.error(f"Error retrieving briefings: {e}")
            return {"error": str(e)}

    logger.info(f"Found {len(briefings)} research briefings in total")

    if not briefings:
        logger.warning("No briefings found matching the criteria")
        return {
            "download_date": datetime.now().isoformat(),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_briefings": 0,
            "successful_downloads": 0,
            "topic_filter": topic,
            "search_term": search_term,
        }

    # Save the raw briefings list for backup
    raw_briefings_file = os.path.join(
        output_dir, f"raw_briefings_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
    )
    with open(raw_briefings_file, "w", encoding="utf-8") as f:
        json.dump(briefings, f, indent=2)
    logger.info(f"Saved raw briefings list to {raw_briefings_file}")

    # Process all briefings and save JSON data
    try:
        processed_briefings = process_briefings(api, briefings, output_dir)
    except Exception as e:
        logger.error(f"Error processing briefings: {e}")
        return {
            "error": str(e),
            "download_date": datetime.now().isoformat(),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_briefings": len(briefings),
            "successful_downloads": 0,
        }

    # Count successful downloads
    successful_downloads = sum(1 for b in processed_briefings if "json_file" in b and "error" not in b)

    # Save the metadata
    metadata_file = os.path.join(
        output_dir, f"briefings_metadata_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
    )

    # Save metadata even if there are some errors
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(processed_briefings, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving metadata file: {e}")

    # Save summary metadata about the download
    metadata = {
        "download_date": datetime.now().isoformat(),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "total_briefings": len(processed_briefings),
        "successful_downloads": successful_downloads,
        "topic_filter": topic,
        "search_term": search_term,
    }

    try:
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving summary metadata file: {e}")

    logger.info(f"Downloaded {successful_downloads}/{len(processed_briefings)} research briefings to {output_dir}")

    return metadata


def clean_filename(filename: str) -> str:
    """Make a string safe for use as a filename."""
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download UK Parliament Research Briefings")
    # Existing arguments
    parser.add_argument(
        "--output", "-o", default="research_briefings", help="Output directory for downloaded briefings"
    )
    parser.add_argument("--days", "-d", type=int, default=30, help="Number of days to look back for briefings")
    parser.add_argument("--start-date", "-s", help="Start date in YYYY-MM-DD format (overrides days parameter)")
    parser.add_argument("--end-date", "-e", help="End date in YYYY-MM-DD format (defaults to today)")
    parser.add_argument("--topic", "-t", help="Filter by topic")
    parser.add_argument("--search", "-q", help="Search term")
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=90,
        help="Maximum chunk size in days for large date ranges (default: 90)",
    )
    parser.add_argument(
        "--resume", "-r", action="store_true", help="Try to resume a previous download using checkpoint files"
    )

    # Add new arguments for PDF download
    parser.add_argument("--download-pdfs", action="store_true", help="Download PDF documents for research briefings")
    parser.add_argument("--pdf-dir", default="pdfs", help="Subdirectory name for PDF storage (default: 'pdfs')")
    parser.add_argument("--overwrite-pdfs", action="store_true", help="Overwrite existing PDFs")

    # Add mode argument to allow running only PDF download without JSON retrieval
    parser.add_argument(
        "--mode",
        choices=["json", "pdf", "both"],
        default="json",
        help="Operation mode: 'json' to download JSON only, 'pdf' to download "
        + "PDFs only for existing JSONs, 'both' to do both (default: 'json')",
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

    # Execute based on mode
    if args.mode in ["json", "both"]:
        # Download JSONs
        download_research_briefings(
            output_dir=args.output, start_date=start_date, end_date=end_date, topic=args.topic, search_term=args.search
        )

    if args.mode in ["pdf", "both"] or args.download_pdfs:
        # Download PDFs for existing JSONs
        download_pdfs_for_briefings(json_dir=args.output, output_subdir=args.pdf_dir, overwrite=args.overwrite_pdfs)
