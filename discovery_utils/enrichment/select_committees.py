"""
Parliament Oral Evidence Sessions Enrichment with S3 Integration

This script processes downloaded oral evidence transcripts from Parliament select committees
and performs keyword analysis on them, with support for both local and S3 storage.
"""

import glob
import json
import logging
import os
import re
import sys
import tempfile

from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import pandas as pd

from botocore.exceptions import ClientError
from bs4 import BeautifulSoup

# Import the S3 handler from the getter module
from discovery_utils.getters.select_committees import SelectCommitteesToS3

# Import the keyword module functions
from discovery_utils.utils.keywords import enrich_keyword_labels
from discovery_utils.utils.keywords import get_keyword_hits
from discovery_utils.utils.keywords import get_keywords
from discovery_utils.utils.keywords import transform_labels_df

# Import S3 utilities
from discovery_utils.utils.s3 import BUCKET_NAME_RAW
from discovery_utils.utils.s3 import s3_client
from discovery_utils.utils.s3 import upload_obj


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evidence_keyword_search.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def find_evidence_files(base_dir="select_committees", htmls_dir="htmls"):
    """Find all HTML evidence files in the specified directory structure."""
    logger.info(f"Searching for evidence files in {base_dir}/{htmls_dir}")

    # Find all HTML files
    html_files = glob.glob(f"{base_dir}/{htmls_dir}/*.html")

    logger.info(f"Found {len(html_files)} HTML evidence files")
    return html_files


def extract_metadata_from_path(file_path):
    """
    Extract metadata from the file path and name.

    Args:
        file_path: Path to the HTML file

    Returns:
        Dictionary with extracted metadata
    """
    metadata = {}

    # Get the base filename
    basename = os.path.basename(file_path)

    # Try to extract committee ID from the filename
    # Format: committee{ID}_{date}_{timestamp}_{title}.html
    committee_match = re.match(r"committee(\d+)_(.+?)(?:_\d{2}-\d{2}-\d{2})?_(.+)\.html", basename)

    if committee_match:
        metadata["committee_id"] = committee_match.group(1)
        date_str = committee_match.group(2)
        metadata["topic"] = committee_match.group(3).replace("_", " ")

        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            metadata["date"] = date
            metadata["date_str"] = date_str
        except ValueError:
            logger.warning(f"Could not parse date from filename: {basename}")
            metadata["date_str"] = ""
    else:
        # If filename doesn't match expected pattern
        metadata["topic"] = os.path.splitext(basename)[0].replace("_", " ")
        metadata["date_str"] = ""

    return metadata


def extract_text_content(html_content):
    """
    Extract the main text content from the HTML.
    Specifically designed for parliamentary committee evidence sessions.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # First try to detect the document structure
    document_structure = detect_document_structure(soup)
    logger.debug(f"Detected document structure: {document_structure}")

    # Extract text based on document structure
    if document_structure == "committee_qa":
        return extract_committee_qa(soup)
    elif document_structure == "committee_transcript":
        return extract_committee_transcript(soup)
    else:
        # Fall back to a generic approach
        return extract_generic_html(soup)


def detect_document_structure(soup):
    """Detect the structure/type of the document based on content patterns."""
    # Check for Q&A pattern with question numbers (Q1, Q2, etc.)
    if soup.find(string=re.compile(r"Q\d+")) or soup.find(string=re.compile(r"The Chair:")):
        return "committee_qa"

    # Check for committee transcript patterns
    elif (
        soup.find(string=re.compile(r"Evidence Session", re.IGNORECASE))
        or soup.find(string=re.compile(r"Members present", re.IGNORECASE))
        or soup.find(string=re.compile(r"Witnesses", re.IGNORECASE))
    ):
        return "committee_transcript"

    # Default
    return "unknown"


def extract_committee_qa(soup):
    """
    Extract text from a Q&A format committee transcript.
    """
    content = []

    # Try to find all paragraphs
    paragraphs = soup.find_all("p")
    if not paragraphs:
        # If no paragraphs found, try to get text blocks by other means
        paragraphs = soup.find_all(["div", "span"])

    # Skip header section (everything before first question)
    start_idx = 0
    for i, p in enumerate(paragraphs):
        text = p.get_text(strip=True)
        if re.search(r"^Q\d+", text) or re.search(r"The Chair:", text):
            start_idx = i
            break

    # Process paragraphs, starting from the first question
    current_question = ""
    current_speaker = ""
    current_text = ""

    for i, p in enumerate(paragraphs):
        if i < start_idx:
            continue

        text = extract_text_with_whitespace(p)
        if not text:
            continue

        # Check for new question
        q_match = re.match(r"^Q(\d+)", text)
        if q_match:
            # Save previous question/answer if exists
            if current_text:
                full_text = f"{current_question} {current_speaker}: {current_text}".strip()
                content.append(full_text)

            # Start new question
            current_question = f"Q{q_match.group(1)}"
            current_speaker = ""
            current_text = re.sub(r"^Q\d+\s*", "", text).strip()
            continue

        # Check for new speaker
        speaker_match = re.match(r"^([^:]+):\s*(.*)", text)
        if speaker_match:
            # Save previous speaker's text if exists
            if current_speaker and current_text:
                full_text = f"{current_question} {current_speaker}: {current_text}".strip()
                content.append(full_text)

            # Start new speaker
            current_speaker = speaker_match.group(1).strip()
            current_text = speaker_match.group(2).strip()
        else:
            # Continue previous speaker's text
            current_text += " " + text

    # Add the last section if exists
    if current_speaker and current_text:
        full_text = f"{current_question} {current_speaker}: {current_text}".strip()
        content.append(full_text)

    return "\n\n".join(content)


def extract_committee_transcript(soup):
    """Extract text from HTML that has a committee transcript structure."""
    content = []

    # Find all paragraphs
    paragraphs = soup.find_all("p")

    # Identify and skip header section
    header_end_idx = 0
    for i, p in enumerate(paragraphs):
        # Common patterns that indicate the end of the header section
        if p.find(string=re.compile(r"Evidence Session", re.IGNORECASE)) or p.find(
            string=re.compile(r"Q\d+", re.IGNORECASE)
        ):
            header_end_idx = i
            break

    # Process paragraphs, skipping the header section
    for i, p in enumerate(paragraphs):
        if i <= header_end_idx and header_end_idx > 0:
            continue

        # Extract text preserving whitespace
        text = extract_text_with_whitespace(p)

        # Add non-empty paragraphs
        if text and len(text) > 5:  # Skip very short lines
            content.append(text)

    return "\n".join(content)


def extract_generic_html(soup):
    """Generic fallback method to extract text from any HTML structure."""
    # Remove script and style elements that would add noise
    for element in soup(["script", "style"]):
        element.extract()

    # Get text from body
    body_text = (
        soup.body.get_text(separator=" ", strip=True) if soup.body else soup.get_text(separator=" ", strip=True)
    )

    # Clean up whitespace
    body_text = re.sub(r"\s+", " ", body_text).strip()

    # Split into paragraphs at sensible boundaries
    paragraphs = re.split(r"(?<=\.)\s{2,}|(?<=\?)\s{2,}|(?<=!)\s{2,}|\n+", body_text)

    # Filter out empty paragraphs and very short ones
    filtered_paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 5]

    return "\n".join(filtered_paragraphs)


def extract_text_with_whitespace(element):
    """Extract text from an element while preserving whitespace between words."""
    text = ""
    for child in element.contents:
        if child.name:
            # For nested elements, get their text
            text += child.get_text()
        else:
            # For direct text nodes
            text += str(child)

    # Clean up the text - normalise whitespace but preserve it between words
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_evidence_files(html_files, df=None):
    """
    Process each HTML file to extract content and metadata.

    Args:
        html_files: List of HTML file paths
        df: Optional DataFrame with additional metadata

    Returns:
        DataFrame with evidence data
    """
    evidence_data = []

    for html_file in html_files:
        logger.info(f"Processing {html_file}")

        try:
            # Read the HTML file
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Extract metadata from file path
            metadata = extract_metadata_from_path(html_file)

            # Add file path
            metadata["file_path"] = html_file

            # Extract text content
            text_content = extract_text_content(html_content)

            # Combine metadata
            evidence_entry = {
                "id": os.path.basename(html_file),
                "file_path": html_file,
                "text": text_content,
                "word_count": len(text_content.split()),
                **metadata,
            }

            if df is not None:
                if "committee_id" in metadata and "committee_id" in df.columns:
                    matches = df[df["committee_id"] == metadata["committee_id"]]
                    if not matches.empty:
                        # Get first match
                        match = matches.iloc[0]
                        # Add additional metadata
                        for key, value in match.items():
                            if key not in evidence_entry and not pd.isna(value):
                                evidence_entry[key] = value

            evidence_data.append(evidence_entry)

        except Exception as e:
            logger.error(f"Error processing {html_file}: {e}")

    return pd.DataFrame(evidence_data)


def load_select_committees_data(
    input_file: str, use_s3: bool = False, s3_handler: Optional[SelectCommitteesToS3] = None
) -> pd.DataFrame:
    """
    Load select committees data from a parquet file.

    Args:
        input_file: Path to the select committees parquet file
        use_s3: Whether to use S3 storage
        s3_handler: S3 handler if use_s3 is True

    Returns:
        DataFrame with select committees data
    """
    if use_s3 and s3_handler:
        try:
            # Check if local file exists first
            if os.path.exists(input_file):
                logger.info(f"Loading select committees data from local file: {input_file}")
                df = pd.read_parquet(input_file)
            else:
                # Download from S3 to a temporary file
                logger.info(f"Local file not found, downloading from S3: {s3_handler.cumulative_file_key}")
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                    tmp_path = tmp.name
                    s3_handler.s3_client.download_file(s3_handler.bucket, s3_handler.cumulative_file_key, tmp_path)

                    # Load from temp file
                    df = pd.read_parquet(tmp_path)

                    # Clean up
                    os.unlink(tmp_path)

            logger.info(f"Loaded {len(df)} select committees records")
            return df
        except Exception as e:
            logger.error(f"Error loading select committees data from S3: {e}")
            raise
    else:
        # Local file only
        try:
            logger.info(f"Loading select committees data from {input_file}")
            df = pd.read_parquet(input_file)
            logger.info(f"Loaded {len(df)} select committees records")
            return df
        except Exception as e:
            logger.error(f"Error loading select committees data from {input_file}: {e}")
            raise


def ensure_html_available(html_file: str, html_dir: str, s3_handler: Optional[SelectCommitteesToS3] = None) -> str:
    """
    Ensure the HTML file is available locally, downloading from S3 if necessary.

    Args:
        html_file: Path or S3 URI to the HTML file
        html_dir: Local directory for HTML files
        s3_handler: S3 handler for downloading from S3

    Returns:
        Local path to the HTML file, or empty string if not available
    """
    # If it's an S3 URI, extract the filename and try to download it
    if isinstance(html_file, str) and html_file.startswith("s3://") and s3_handler:
        # Download the file if it's in S3
        local_path = s3_handler.download_html(html_file, html_dir)
        if local_path:
            return local_path
    # If it's already a local path, check if it exists
    elif isinstance(html_file, str) and os.path.exists(html_file):
        return html_file

    logger.warning(f"HTML file not found: {html_file}")
    return ""


def prepare_evidence_data_with_html_text(
    df: pd.DataFrame,
    html_dir: str,
    batch_size: int = 10,
    s3_handler: Optional[SelectCommitteesToS3] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Enhance the DataFrame with text extracted from HTML files.

    Args:
        df: DataFrame with select committees data
        html_dir: Directory for HTML files
        batch_size: Number of files to process in each batch
        s3_handler: S3 handler if use_s3 is True

    Returns:
        Tuple of (DataFrame with HTML text, stats dictionary)
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()

    # Add text column if not present
    if "text" not in result_df.columns:
        result_df["text"] = None

    # Process in batches to manage memory usage
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size

    logger.info(f"Processing {total_rows} evidence sessions in {num_batches} batches of size {batch_size}")

    # Track stats
    stats = {"total_sessions": total_rows, "html_processed": 0, "html_not_found": 0, "s3_downloads": 0}

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_rows)

        logger.info(f"Processing batch {batch_num+1}/{num_batches} (rows {start_idx+1}-{end_idx})")

        batch_df = df.iloc[start_idx:end_idx]

        # Process each evidence in the batch
        for idx, row in batch_df.iterrows():
            html_path = None
            for field in ["html_file", "s3_html_file", "file_path"]:
                if field in row and not pd.isna(row[field]):
                    html_path = row[field]
                    break

            if not html_path:
                logger.debug(f"No HTML file specified for evidence {row.get('id', '')}")
                stats["html_not_found"] += 1
                continue

            # Ensure the HTML is available locally
            local_html_path = ensure_html_available(html_path, html_dir, s3_handler)

            if not local_html_path:
                logger.warning(f"Could not obtain HTML for evidence {row.get('id', '')}")
                stats["html_not_found"] += 1
                continue

            try:
                # Read the HTML file
                with open(local_html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

                # Extract text from HTML
                logger.info(f"Extracting text from HTML: {os.path.basename(local_html_path)}")
                text_content = extract_text_content(html_content)

                if text_content:
                    stats["html_processed"] += 1
                    result_df.at[idx, "text"] = text_content
                else:
                    stats["html_not_found"] += 1  # Count as not found if extraction failed
            except Exception as e:
                logger.error(f"Error processing HTML file {local_html_path}: {e}")
                stats["html_not_found"] += 1

    # Log statistics
    logger.info(f"HTML text extraction completed:")
    logger.info(f"  Total HTML files processed: {stats['html_processed']}")
    logger.info(f"  HTML files not found or failed: {stats['html_not_found']}")

    return result_df, stats


def prepare_data_for_keyword_analysis(
    df: pd.DataFrame,
    html_dir: str,
    batch_size: int = 10,
    use_s3: bool = False,
    s3_handler: Optional[SelectCommitteesToS3] = None,
) -> Tuple[pd.DataFrame, Dict[str, int], pd.DataFrame]:
    """
    Prepare the evidence data for keyword analysis, including HTML text.

    Args:
        df: DataFrame with select committees data
        html_dir: Directory for HTML files
        batch_size: Number of files to process in each batch
        use_s3: Whether to use S3 storage
        s3_handler: S3 handler if use_s3 is True

    Returns:
        Tuple of (DataFrame for keyword analysis, stats dictionary, enhanced DataFrame)
    """
    # Check for cumulative file in S3
    text_df = None
    if use_s3 and s3_handler:
        text_file_key = f"{s3_handler.prefix}/select_committees_text.parquet"
        try:
            # Download the text file to a temp location
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = tmp.name
                s3_handler.s3_client.download_file(s3_handler.bucket, text_file_key, tmp_path)

                # Load the existing text data
                text_df = pd.read_parquet(tmp_path)
                logger.info(f"Loaded existing text data for {len(text_df)} evidence sessions from S3")

                # Clean up
                os.unlink(tmp_path)
        except ClientError:
            logger.info("No existing text data found in S3, will extract from HTML files")
        except Exception as e:
            logger.warning(f"Error loading text data from S3: {e}")

    # Find sessions that already have text
    ids_with_text = set()
    if text_df is not None and not text_df.empty and "id" in text_df.columns and "text" in text_df.columns:
        # Create a map of ID to text
        text_map = dict(zip(text_df["id"], text_df["text"]))

        # Add text to the main DataFrame where available
        for idx, row in df.iterrows():
            if row["id"] in text_map:
                df.at[idx, "text"] = text_map[row["id"]]
                ids_with_text.add(row["id"])

    # Find sessions that need text extraction
    df_needs_text = df[~df["id"].isin(ids_with_text)].copy()

    if not df_needs_text.empty:
        logger.info(f"Extracting text for {len(df_needs_text)} evidence sessions that don't have text yet")
        # Extract text for sessions that don't have it yet
        text_extracted_df, html_stats = prepare_evidence_data_with_html_text(
            df_needs_text, html_dir, batch_size, s3_handler
        )

        # Merge back into main DataFrame
        for idx, row in text_extracted_df.iterrows():
            if pd.notna(row["text"]):
                df.at[idx, "text"] = row["text"]
    else:
        logger.info("All evidence sessions already have text data")
        html_stats = {
            "total_sessions": len(df),
            "html_processed": len(ids_with_text),
            "html_not_found": 0,
            "s3_downloads": 0,
        }

    analysis_df = df[["id"]].copy()
    analysis_df["text"] = df["text"]
    analysis_df = analysis_df[analysis_df["text"].notna() & (analysis_df["text"] != "")]

    # If we're using S3, update the text cumulative file
    if use_s3 and s3_handler:
        # Create a DataFrame with just id and text
        updated_text_df = df[["id", "text"]].copy()
        updated_text_df = updated_text_df[updated_text_df["text"].notna()]

        try:
            # Upload to S3
            upload_obj(updated_text_df, s3_handler.bucket, text_file_key)
            logger.info(f"Updated text cumulative file in S3 with {len(updated_text_df)} entries")
        except Exception as e:
            logger.error(f"Error updating text cumulative file in S3: {e}")

    return analysis_df, html_stats, df


def perform_keyword_analysis(
    df: pd.DataFrame, keyword_type: str, analysis_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Perform keyword analysis on the evidence sessions.

    Args:
        df: DataFrame with evidence data
        keyword_type: Type of keywords to use for analysis {ASF, AFS, AHL, X, Nesta}
        analysis_df: Pre-prepared DataFrame for analysis with 'id' and 'text' columns

    Returns:
        DataFrame with keyword analysis results
    """
    logger.info(f"Performing keyword analysis using {keyword_type} keywords")

    if analysis_df is None:
        logger.error("No analysis DataFrame provided")
        return df

    try:
        # Apply keyword enrichment
        logger.info(f"Enriching labels with {keyword_type} keywords...")
        enriched_df = enrich_keyword_labels(analysis_df[["id", "text"]], keyword_type)
        transformed_df = transform_labels_df(enriched_df)

        result_df = pd.merge(df, transformed_df, on="id", how="left")

        # Add a column indicating whether keywords were found
        result_df["has_keywords"] = ~result_df["mission_labels"].isna()

        return result_df

    except Exception as e:
        logger.error(f"Error during keyword analysis: {e}")
        raise


def get_detailed_keyword_matches(
    df: pd.DataFrame, keyword_type: str = "ASF", analysis_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Get detailed information about keyword matches in each document.

    Args:
        df: DataFrame with evidence data
        keyword_type: Type of keywords to use for analysis
        analysis_df: Pre-prepared DataFrame for analysis with 'id' and 'text' columns

    Returns:
        DataFrame with detailed keyword match information
    """
    logger.info(f"Getting detailed keyword matches using {keyword_type} keywords")

    # Get keywords dictionary
    keywords_dict = get_keywords(keyword_type)

    if analysis_df is None:
        logger.error("No analysis DataFrame provided")
        return pd.DataFrame()

    detailed_matches = []

    for idx, row in analysis_df.iterrows():
        try:
            # Get the text to search
            text = row.get("text", "")

            if not text:
                continue

            # Get keyword hits
            hits_df = get_keyword_hits(text, keywords_dict)

            # Process hits
            if not hits_df.empty:
                # Get the original row from df for metadata
                original_row = df[df["id"] == row["id"]].iloc[0] if not df[df["id"] == row["id"]].empty else {}

                # Format the results
                for _, hit_row in hits_df.iterrows():
                    match_info = {
                        "id": row["id"],
                        "committee_id": original_row.get("committee_id", ""),
                        "committee_name": original_row.get("committee_name", ""),
                        "topic": original_row.get("topic", ""),
                        "date": original_row.get("date_str", ""),
                        "mission_labels": keyword_type,
                        "topic_labels": ", ".join(hit_row["category"]),
                        "keywords": [kw for sublist in hit_row["keyword"] for kw in sublist],
                        "sentence": hit_row["sentence"],
                        "marked_sentence": hit_row["marked_sentence"],
                    }
                    detailed_matches.append(match_info)

        except Exception as e:
            logger.error(f"Error getting keyword matches for {row['id']}: {e}")

    return pd.DataFrame(detailed_matches)


def update_cumulative_keyword_matches(
    matches_df: pd.DataFrame,
    s3_handler: SelectCommitteesToS3,
    labelstore_key: str = "select_committees_labelstore_keywords.parquet",
) -> bool:
    """
    Update the cumulative keyword matches file in S3.

    Args:
        matches_df: DataFrame with new keyword matches
        s3_handler: S3 handler for S3 operations
        labelstore_key: S3 key for the cumulative parquet file

    Returns:
        True if update was successful, False otherwise
    """
    if matches_df.empty:
        logger.warning("No keyword matches to update")
        return False

    full_key = f"{s3_handler.prefix}/{labelstore_key}"

    try:
        # Check if file exists in S3
        try:
            s3_handler.s3_client.head_object(Bucket=s3_handler.bucket, Key=full_key)
            file_exists = True
        except ClientError:
            file_exists = False

        if file_exists:
            # Download existing file
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = tmp.name
                s3_handler.s3_client.download_file(s3_handler.bucket, full_key, tmp_path)

                # Load existing data
                existing_df = pd.read_parquet(tmp_path)

                # Clean up
                os.unlink(tmp_path)

                # Concatenate with new data
                logger.info(f"Updating keyword matches: {len(existing_df)} existing + {len(matches_df)} new")
                combined_df = pd.concat([existing_df, matches_df], ignore_index=True)

                # Remove duplicates based on same sentence and keyword
                dedup_df = combined_df.drop_duplicates(subset=["id", "sentence", "keywords"], keep="last")

                logger.info(f"After deduplication: {len(dedup_df)} entries")
        else:
            logger.info(f"Creating new keyword matches parquet with {len(matches_df)} entries")
            dedup_df = matches_df

        # Upload to S3
        # First save to temp file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
            dedup_df.to_parquet(tmp_path, index=False)

            # Upload to S3
            s3_handler.s3_client.upload_file(tmp_path, s3_handler.bucket, full_key)

            # Clean up
            os.unlink(tmp_path)

        logger.info(f"Successfully updated cumulative keyword matches in S3: s3://{s3_handler.bucket}/{full_key}")
        return True

    except Exception as e:
        logger.error(f"Error updating cumulative keyword matches in S3: {e}")
        return False


def generate_pipeline_artifact(
    metadata: Dict[str, Any],
    output_dir: str,
    run_date: datetime,
    artifact_filename: str = "enrichment_artifact.json",
    use_s3: bool = False,
    s3_handler: Optional[SelectCommitteesToS3] = None,
) -> str:
    """
    Generate a pipeline artifact file from the enrichment metadata.

    Args:
        metadata: Dictionary with enrichment metadata
        output_dir: Directory to save the artifact
        run_date: Date of the run
        artifact_filename: Name of the artifact file
        use_s3: Whether to use S3 storage
        s3_handler: S3 handler if use_s3 is True

    Returns:
        Path to the created artifact file
    """
    artifact_data = {
        "run_date": run_date.isoformat(),
    }

    artifact_data["summary"] = {
        "total_sessions_analysed": metadata.get("total_sessions", 0),
        "keyword_types_used": metadata.get("keyword_types", []),
        "output_file": os.path.basename(metadata.get("output_file", "")),
        "status": "success" if not metadata.get("error") else "error",
        "error_message": metadata.get("error", ""),
    }

    artifact_data["results"] = {
        "total_keyword_matches": metadata.get("total_matches", 0),
        "unique_sessions_with_matches": metadata.get("unique_sessions_with_matches", 0),
    }

    artifact_data["keyword_results"] = {}
    for keyword_type, result in metadata.get("results", {}).items():
        artifact_data["keyword_results"][keyword_type] = {
            "sessions_with_keywords": result.get("sessions_with_keywords", 0),
            "keyword_matches": result.get("keyword_matches", 0),
        }

    if "html_stats" in metadata:
        artifact_data["html_processing"] = metadata["html_stats"]

    if use_s3 and s3_handler:
        run_dir = s3_handler.create_run_directory(run_date)
        s3_key = f"{run_dir}/enrichment_artifact.json"

        try:
            # Upload the artifact JSON to S3
            upload_obj(artifact_data, s3_handler.bucket, s3_key)
            logger.info(f"Generated enrichment pipeline artifact in S3: s3://{s3_handler.bucket}/{s3_key}")
            artifact_path = f"s3://{s3_handler.bucket}/{s3_key}"
        except Exception as e:
            logger.error(f"Error uploading artifact to S3: {e}")

            # Fall back to local storage
            artifact_path = os.path.join(output_dir, artifact_filename)
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(artifact_data, f, indent=2)
            logger.info(f"Generated enrichment pipeline artifact locally: {artifact_path}")
    else:
        # Save locally
        artifact_path = os.path.join(output_dir, artifact_filename)
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact_data, f, indent=2)
        logger.info(f"Generated enrichment pipeline artifact locally: {artifact_path}")

    return artifact_path


def analyse_select_committees(
    input_file: str = None,
    output_dir: str = "analysis_output",
    keyword_types: List[str] = ["ASF", "AFS", "AHL", "X", "Nesta"],
    batch_size: int = 10,
    use_s3: bool = False,
    s3_prefix: str = None,
) -> Dict:
    """
    Analyse select committees evidence with keyword analysis.

    Args:
        input_file: Path to the select committees parquet file (if None, will search in output_dir)
        output_dir: Directory to save the analysis results
        keyword_types: List of keyword types to use for analysis
        batch_size: Number of files to process in each batch
        use_s3: Whether to use S3 storage
        s3_prefix: S3 prefix (folder path) if use_s3 is True

    Returns:
        Dictionary with analysis metadata
    """
    run_date = datetime.now()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create HTML directory
    html_dir = os.path.join(output_dir, "htmls")
    os.makedirs(html_dir, exist_ok=True)

    # Initialise S3 handler if using S3
    s3_handler = None
    if use_s3:
        s3_handler = SelectCommitteesToS3(s3_prefix)

    # If no input file provided, try to find the latest one
    if not input_file:
        error_msg = f"No select committees file found"
        logger.error(error_msg)
        metadata = {"error": error_msg, "run_date": run_date.isoformat(), "status": "error"}
        generate_pipeline_artifact(metadata, output_dir, run_date, use_s3=use_s3, s3_handler=s3_handler)
        return metadata

    # Load select committees data
    df = load_select_committees_data(input_file, use_s3, s3_handler)

    if df.empty:
        error_msg = f"No select committees data found in {input_file}"
        logger.error(error_msg)
        metadata = {"error": error_msg, "run_date": run_date.isoformat(), "status": "error"}
        generate_pipeline_artifact(metadata, output_dir, run_date, use_s3=use_s3, s3_handler=s3_handler)
        return metadata

    # Prepare data for keyword analysis (including HTML text)
    logger.info("Preparing data for keyword analysis (including HTML text)...")
    analysis_df, html_stats, enhanced_df = prepare_data_for_keyword_analysis(
        df=df, html_dir=html_dir, batch_size=batch_size, use_s3=use_s3, s3_handler=s3_handler
    )

    # Collect all matches across keyword types
    all_matches = []
    all_results = {}

    for keyword_type in keyword_types:
        try:
            result_df = perform_keyword_analysis(df, keyword_type, analysis_df)
            matches_df = get_detailed_keyword_matches(df, keyword_type, analysis_df)

            if not matches_df.empty:
                # Add to collection of all matches
                all_matches.append(matches_df)

                # Store results
                all_results[keyword_type] = {
                    "total_sessions": len(df),
                    "sessions_with_keywords": int(result_df["has_keywords"].sum()),
                    "keyword_matches": len(matches_df),
                }

                logger.info(f"Found {len(matches_df)} matches for {keyword_type}")
            else:
                logger.info(f"No keyword matches found for {keyword_type}")
                all_results[keyword_type] = {
                    "total_sessions": len(df),
                    "sessions_with_keywords": 0,
                    "keyword_matches": 0,
                }

        except Exception as e:
            logger.error(f"Error processing {keyword_type} keyword analysis: {e}")
            all_results[keyword_type] = {"error": str(e)}

    # Combine all matches
    if all_matches:
        combined_matches = pd.concat(all_matches, ignore_index=True)
        logger.info(f"Combined {len(combined_matches)} matches from all keyword types")

        # Save to parquet file
        date_str = run_date.strftime("%Y%m%d")
        local_output_file = os.path.join(output_dir, f"select_committees_enriched_{date_str}.parquet")
        combined_matches.to_parquet(local_output_file, index=False)
        logger.info(f"Saved all keyword matches locally to {local_output_file}")

        # If using S3, also update the cumulative keyword matches file
        if use_s3 and s3_handler:
            update_cumulative_keyword_matches(combined_matches, s3_handler)

        # Calculate summary statistics
        unique_sessions = len(combined_matches["id"].unique())
        total_matches = len(combined_matches)
    else:
        logger.warning("No matches found for any keyword type")
        # Create empty file
        combined_matches = pd.DataFrame(
            columns=[
                "id",
                "committee_id",
                "committee_name",
                "topic",
                "date",
                "mission_labels",
                "topic_labels",
                "keywords",
                "sentence",
                "marked_sentence",
            ]
        )
        date_str = run_date.strftime("%Y%m%d")
        local_output_file = os.path.join(output_dir, f"select_committees_enriched.parquet")
        combined_matches.to_parquet(local_output_file, index=False)
        logger.info(f"Saved empty matches file locally to {local_output_file}")

        unique_sessions = 0
        total_matches = 0

    # Save overall metadata
    metadata = {
        "total_sessions": len(df),
        "keyword_types": keyword_types,
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
        "run_date": run_date.isoformat(),
        "output_file": local_output_file,
        "total_matches": total_matches,
        "unique_sessions_with_matches": unique_sessions,
        "html_stats": {
            "total_sessions": html_stats["total_sessions"],
            "html_processed": html_stats["html_processed"],
            "html_not_found": html_stats["html_not_found"],
        },
    }

    # Generate pipeline artifact
    artifact_path = generate_pipeline_artifact(metadata, output_dir, run_date, use_s3=use_s3, s3_handler=s3_handler)

    # Add artifact path to returned metadata
    metadata["artifact_path"] = artifact_path

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse Select Committee Evidence")
    parser.add_argument(
        "--input-file",
        "-i",
        default="outputs/select_committees/select_committees.parquet",
        help="Path to the select committees parquet file (if not provided, will look in output directory)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="outputs/select_committees/enrichment",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--keywords",
        "-k",
        nargs="+",
        default=["ASF", "AFS", "AHL", "X", "Nesta"],
        help="Keyword types to use for analysis",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Number of files to process in each batch (default: 10)"
    )
    parser.add_argument("--use-s3", action="store_true", help="Store files in S3 instead of locally")
    parser.add_argument(
        "--s3-prefix",
        default="data/policy/select_committees",
        help="S3 prefix (folder path) (required if --use-s3 is specified)",
    )

    args = parser.parse_args()

    # Perform keyword analysis
    analyse_select_committees(
        input_file=args.input_file,
        output_dir=args.output_dir,
        keyword_types=args.keywords,
        batch_size=args.batch_size,
        use_s3=args.use_s3,
        s3_prefix=args.s3_prefix,
    )
