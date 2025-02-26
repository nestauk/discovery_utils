"""
Parliament Oral Evidence Sessions Getter

This script downloads oral evidence transcripts from Parliament select committees
using the Committees API.
"""

import base64
import json
import logging
import os
import sys
import time

from datetime import datetime
from datetime import timedelta

import requests


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evidence_download.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class CommitteesAPI:
    """Client for interacting with the Parliament Committees API."""

    def __init__(self, base_url="https://committees-api.parliament.uk/api"):
        """Initialise the API client with the base URL."""
        self.base_url = base_url
        self.session = requests.Session()

    def get_select_committees(self):
        """Get a list of all select committees."""
        url = f"{self.base_url}/Committees"
        params = {
            "CommitteeCategory": "Select",
            "CommitteeStatus": "Current",
            "Take": 100,  # Assuming we won't have more than 100 current select committees
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

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

        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_all_oral_evidence(self, committee_id=None, start_date=None, end_date=None):
        """Get all oral evidence sessions, handling pagination."""
        all_evidence = []
        skip = 0
        take = 30

        while True:
            batch = self.get_oral_evidence_list(
                committee_id=committee_id, start_date=start_date, end_date=end_date, skip=skip, take=take
            )

            if not batch["items"]:
                break

            all_evidence.extend(batch["items"])

            # If we got fewer items than requested, we've reached the end
            if len(batch["items"]) < take:
                break

            skip += take
            time.sleep(0.5)  # Be nice to the API

        return all_evidence

    def get_oral_evidence_document(self, evidence_id, file_format="Html"):
        """Get the document data for an oral evidence session."""
        url = f"{self.base_url}/OralEvidence/{evidence_id}/Document/{file_format}"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error retrieving document for evidence {evidence_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving document for evidence {evidence_id}: {e}")
            return None


def download_committee_evidence(api, committee_id, committee_name, output_dir, start_date, end_date, file_format):
    """Download all evidence for a specific committee within the date range."""
    logger.info(f"Processing committee: {committee_name} (ID: {committee_id})")

    # Create committee directory
    committee_dir = os.path.join(output_dir, f"{committee_id}_{clean_filename(committee_name)}")
    os.makedirs(committee_dir, exist_ok=True)

    # Get evidence sessions
    evidence_sessions = api.get_all_oral_evidence(committee_id, start_date, end_date)
    logger.info(f"Found {len(evidence_sessions)} oral evidence sessions for {committee_name}")

    # Download each evidence session
    success_count = 0
    for evidence in evidence_sessions:
        evidence_id = evidence["id"]

        # Get publication date
        pub_date = evidence.get("publicationDate")
        pub_date_str = pub_date if pub_date else "unknown_date"

        # Get committee business title (if available)
        business_title = "unknown_inquiry"
        if evidence.get("committeeBusinesses") and len(evidence["committeeBusinesses"]) > 0:
            business_title = evidence["committeeBusinesses"][0].get("title", business_title)

        # Create filename
        extension = "html" if file_format.lower() == "html" else "pdf" if file_format.lower() == "pdf" else "txt"
        filename = f"{pub_date_str}_{clean_filename(business_title)}.{extension}"
        output_path = os.path.join(committee_dir, clean_filename(filename))

        # Skip if already downloaded
        if os.path.exists(output_path):
            logger.info(f"Document already exists: {output_path}")
            success_count += 1
        else:
            # Get document
            success = download_evidence_document(api, evidence_id, output_path, file_format)
            if success:
                success_count += 1

        # Be nice to the API
        time.sleep(0.5)

    return success_count


def download_evidence_document(api, evidence_id, output_path, file_format):
    """Download and save an evidence document."""
    try:
        # Get document details
        doc_details = api.get_oral_evidence_document(evidence_id, file_format)

        if not doc_details:
            logger.warning(f"No document details returned for evidence ID {evidence_id}")
            return False

        # Check for base64 encoded data
        if "data" in doc_details and doc_details["data"]:
            logger.info(f"Saving document data for evidence ID {evidence_id}")

            # Decode base64 data and save
            try:
                decoded_data = base64.b64decode(doc_details["data"])

                with open(output_path, "wb") as f:
                    f.write(decoded_data)

                logger.info(f"Successfully saved document to {output_path}")
                return True
            except Exception as e:
                logger.error(f"Error decoding or saving document data: {e}")
                return False
        else:
            logger.warning(f"No data field in document response for evidence ID {evidence_id}")
            return False

    except Exception as e:
        logger.error(f"Error downloading document for evidence ID {evidence_id}: {e}")
        return False


def clean_filename(filename):
    """Make a string safe for use as a filename."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Truncate if too long
    max_length = 200
    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename


def download_all_evidence(output_dir="parliament_evidence", start_date=None, end_date=None, file_format="Html"):
    """Download all oral evidence from select committees in the specified date range."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now()

    if not start_date:
        # Default to 6 months before end date
        start_date = end_date - timedelta(days=180)

    logger.info(f"Downloading oral evidence from {start_date.date()} to {end_date.date()}")

    # Initialise API client
    api = CommitteesAPI()

    # Get all select committees
    committees_response = api.get_select_committees()
    committees = committees_response.get("items", [])
    logger.info(f"Found {len(committees)} select committees")

    # Metadata to track progress
    metadata = {
        "download_date": datetime.now().isoformat(),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "committees": len(committees),
        "total_evidence_count": 0,
        "total_download_count": 0,
    }

    # Download evidence for each committee
    total_evidence_count = 0
    total_download_count = 0

    for committee in committees:
        committee_id = committee["id"]
        committee_name = committee["name"]

        success_count = download_committee_evidence(
            api=api,
            committee_id=committee_id,
            committee_name=committee_name,
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
            file_format=file_format,
        )

        evidence_count = len(api.get_all_oral_evidence(committee_id, start_date, end_date))
        total_evidence_count += evidence_count
        total_download_count += success_count

    # Update and save metadata
    metadata["total_evidence_count"] = total_evidence_count
    metadata["total_download_count"] = total_download_count

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Download complete. Found {total_evidence_count} oral evidence sessions, successfully downloaded {total_download_count}."
    )

    return metadata


if __name__ == "__main__":
    # Set specific date range (6-7 months ago)
    end_date = datetime.now()
    start_date = datetime.now() - timedelta(days=7)

    # Download evidence
    download_all_evidence(
        output_dir="uk_parliament_oral_evidence",
        start_date=start_date,
        end_date=end_date,
        file_format="Html",  # Can be "Html", "Pdf", or "OriginalFormat"
    )
