"""
UK Parliament Research Briefings PDF Downloader

This module extends the research_briefings_getter.py script to download
the PDF documents associated with research briefings.
"""

import glob
import json
import logging
import os
import time

from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import requests


# Setup logging to match the main script
logger = logging.getLogger(__name__)


def extract_nested_value(data: Dict, path: List[str], default=None):
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


def extract_briefing_id_from_url(url: str) -> str:
    """
    Extract the briefing ID from a URL.

    Args:
        url: The URL to extract from

    Returns:
        The extracted briefing ID
    """
    import re

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


def clean_filename(filename: str) -> str:
    """Make a string safe for use as a filename."""
    import re

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


def download_pdfs_for_briefings(
    json_dir: str,
    output_subdir: str = "pdfs",
    overwrite: bool = False,
    max_attempts: int = 3,
    delay_between_downloads: float = 0.5,
) -> Dict:
    """
    Download PDF documents for research briefings based on JSON files.

    Args:
        json_dir: Directory containing research briefing JSON files
        output_subdir: Subdirectory name where PDFs will be stored
        overwrite: Whether to overwrite existing PDFs
        max_attempts: Maximum number of download attempts per PDF
        delay_between_downloads: Delay between downloads to be nice to the server

    Returns:
        Dictionary with download statistics
    """
    # Create output directory
    pdf_dir = os.path.join(json_dir, output_subdir)
    os.makedirs(pdf_dir, exist_ok=True)

    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(json_dir, "*.json"))

    # Skip metadata files
    json_files = [
        f
        for f in json_files
        if not any(skip_term in os.path.basename(f) for skip_term in ["metadata", "checkpoint", "raw_briefings"])
    ]

    logger.info(f"Found {len(json_files)} JSON files to process for PDF downloads")

    # Statistics
    stats = {
        "total_jsons": len(json_files),
        "pdfs_downloaded": 0,
        "pdfs_already_exist": 0,
        "pdfs_failed": 0,
        "pdfs_no_url": 0,
    }

    # Process each JSON file
    for i, json_file in enumerate(json_files):
        logger.info(f"Processing file {i+1}/{len(json_files)}: {os.path.basename(json_file)}")

        try:
            # Load the JSON file
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract briefing ID and PDF URL
            pdf_url = None

            # Try different paths where the PDF URL might be stored
            pdf_url = extract_nested_value(data, ["result", "primaryTopic", "contentLocation"], None)

            # Also try the briefingDocument fileUrl as a backup
            if not pdf_url:
                pdf_url = extract_nested_value(data, ["result", "primaryTopic", "briefingDocument", "fileUrl"], None)

            # Try to get the briefing ID from the _about URL
            about_url = extract_nested_value(data, ["result", "primaryTopic", "_about"], "")
            briefing_id = extract_briefing_id_from_url(about_url)

            # Try to get the identifier as an alternative
            identifier = extract_nested_value(data, ["result", "primaryTopic", "identifier", "_value"], "")

            # Get the title for filename creation
            title = extract_nested_value(data, ["result", "primaryTopic", "title"], "")

            # If no URL found, log and continue
            if not pdf_url:
                logger.warning(f"No PDF URL found in {os.path.basename(json_file)}")
                stats["pdfs_no_url"] += 1
                continue

            # Create filename for the PDF
            if identifier:
                # Use identifier (e.g., SN05035) as it's more human-readable
                pdf_filename = f"{identifier}.pdf"
            elif briefing_id:
                # Use briefing ID as fallback
                pdf_filename = f"{briefing_id}.pdf"
            else:
                # Use the JSON filename as a base
                base_name = os.path.splitext(os.path.basename(json_file))[0]
                pdf_filename = f"{base_name}.pdf"

            # Clean the filename
            pdf_filename = clean_filename(pdf_filename)
            pdf_path = os.path.join(pdf_dir, pdf_filename)

            # Check if PDF already exists
            if os.path.exists(pdf_path) and not overwrite:
                logger.info(f"PDF already exists: {pdf_filename}")
                stats["pdfs_already_exist"] += 1
                continue

            # Download the PDF with retry mechanism
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
                    stats["pdfs_downloaded"] += 1
                    success = True
                    break

                except requests.exceptions.RequestException as e:
                    logger.warning(f"Attempt {attempt+1}/{max_attempts} failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        # Exponential backoff
                        wait_time = 2**attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)

            if not success:
                logger.error(f"Failed to download PDF after {max_attempts} attempts")
                stats["pdfs_failed"] += 1

            # Add delay between downloads to be nice to the server
            time.sleep(delay_between_downloads)

        except Exception as e:
            logger.error(f"Error processing {os.path.basename(json_file)}: {str(e)}")
            stats["pdfs_failed"] += 1

    # Log summary
    logger.info(f"PDF download complete. Summary:")
    logger.info(f"  Total JSON files processed: {stats['total_jsons']}")
    logger.info(f"  PDFs successfully downloaded: {stats['pdfs_downloaded']}")
    logger.info(f"  PDFs already existed: {stats['pdfs_already_exist']}")
    logger.info(f"  JSONs with no PDF URL: {stats['pdfs_no_url']}")
    logger.info(f"  Failed downloads: {stats['pdfs_failed']}")

    return stats
