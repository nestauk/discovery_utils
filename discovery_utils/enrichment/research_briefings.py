"""
Research Briefings Enrichment with S3 Integration

This script processes downloaded research briefings data by performing keyword searches
on both the abstracts and the PDF content. It integrates with S3 for storage and retrieval.
"""

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

import boto3
import pandas as pd
import pdfplumber

from botocore.exceptions import ClientError

from discovery_utils.getters.research_briefings import ResearchBriefingsToS3
from discovery_utils.utils.keywords import enrich_keyword_labels
from discovery_utils.utils.keywords import get_keyword_hits
from discovery_utils.utils.keywords import get_keywords
from discovery_utils.utils.keywords import transform_labels_df
from discovery_utils.utils.s3 import s3_client
from discovery_utils.utils.s3 import upload_obj


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("research_briefings_enrichment.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract (None for all pages)

    Returns:
        Extracted text as a string
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Determine page range
            pages_to_extract = min(len(pdf.pages), max_pages or float("inf"))

            # Extract text from each page
            text = ""
            for i in range(int(pages_to_extract)):
                try:
                    page = pdf.pages[i]
                    page_text = page.extract_text() or ""
                    text += page_text
                    text += "\n\n"  # Add extra newlines between pages
                except Exception as e:
                    logger.warning(f"Error extracting text from page {i+1} in {pdf_path}: {e}")

        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


def load_research_briefings(
    input_file: str = "research_briefings.parquet",
    use_s3: bool = False,
    s3_handler: Optional[ResearchBriefingsToS3] = None,
) -> pd.DataFrame:
    """
    Load research briefings from a parquet file.

    Args:
        input_file: Path to the briefings parquet file
        use_s3: Whether to use S3 storage
        s3_handler: S3 handler if use_s3 is True

    Returns:
        DataFrame with research briefing data
    """
    if use_s3 and s3_handler:
        try:
            # Check if local file exists first
            if os.path.exists(input_file):
                logger.info(f"Loading research briefings from local file: {input_file}")
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

            logger.info(f"Loaded {len(df)} research briefings")
            return df
        except Exception as e:
            logger.error(f"Error loading research briefings from S3: {e}")
            raise
    else:
        # Local file only
        try:
            logger.info(f"Loading research briefings from {input_file}")
            df = pd.read_parquet(input_file)
            logger.info(f"Loaded {len(df)} research briefings")
            return df
        except Exception as e:
            logger.error(f"Error loading research briefings from {input_file}: {e}")
            raise


def ensure_pdf_available(pdf_file: str, s3_handler: Optional[ResearchBriefingsToS3] = None) -> str:
    """
    Ensure the PDF is available locally, downloading from S3 if necessary.

    Args:
        pdf_file: Path to the PDF file (could be local or S3 URI)
        s3_handler: S3 handler for downloading from S3

    Returns:
        Local path to the PDF file, or empty string if not available
    """
    # If it's an S3 URI, extract the filename and try to download it
    if pdf_file.startswith("s3://") and s3_handler:
        # Extract the key from the S3 URI
        s3_parts = pdf_file.replace("s3://", "").split("/")
        bucket = s3_parts[0]
        key = "/".join(s3_parts[1:])

        # Extract just the filename
        filename = os.path.basename(key)

        # Define local path
        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs")
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)

        # Check if it exists locally first
        if os.path.exists(local_path):
            logger.info(f"PDF already exists locally: {local_path}")
            return local_path

        # Download from S3
        try:
            logger.info(f"Downloading PDF from S3: {pdf_file}")
            s3_handler.s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded PDF to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Error downloading PDF from S3: {e}")
            return ""

    # If it's already a local path, check if it exists
    elif os.path.exists(pdf_file):
        logger.info(f"Using local PDF: {pdf_file}")
        return pdf_file

    # If local_pdf_file is provided in metadata, use that
    elif "local_pdf_file" in pdf_file:
        local_path = pdf_file["local_pdf_file"]
        if os.path.exists(local_path):
            logger.info(f"Using local PDF from metadata: {local_path}")
            return local_path

    logger.warning(f"PDF not found: {pdf_file}")
    return ""


def prepare_briefing_data_with_pdf_text(
    df: pd.DataFrame,
    max_pages: Optional[int] = None,
    batch_size: int = 10,
    s3_handler: Optional[ResearchBriefingsToS3] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Enhance the DataFrame with text extracted from PDFs in batches.

    Args:
        df: DataFrame with research briefing data
        max_pages: Maximum number of pages to extract per PDF
        batch_size: Number of PDFs to process in each batch
        s3_handler: S3 handler if use_s3 is True

    Returns:
        Tuple containing:
        - DataFrame enhanced with PDF text content
        - Dictionary with PDF processing statistics
    """
    result_df = df.copy()
    result_df["pdf_text"] = None

    # Process in batches to manage memory usage
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size

    logger.info(f"Processing {total_rows} briefings in {num_batches} batches of size {batch_size}")

    # Track stats
    stats = {"total_briefings": total_rows, "pdfs_processed": 0, "pdfs_not_found": 0, "s3_downloads": 0}

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_rows)

        logger.info(f"Processing batch {batch_num+1}/{num_batches} (rows {start_idx+1}-{end_idx})")

        batch_df = df.iloc[start_idx:end_idx]

        # Process each briefing in the batch
        for idx, row in batch_df.iterrows():
            # First check for local_pdf_file
            pdf_path = row.get("local_pdf_file", None)

            # If not found, check regular pdf_file
            if not pdf_path or pd.isna(pdf_path):
                pdf_path = row.get("pdf_file", None)

            if not pdf_path or pd.isna(pdf_path):
                logger.debug(f"No PDF file specified for briefing {row.get('id', '')}")
                stats["pdfs_not_found"] += 1
                continue

            # Ensure the PDF is available locally
            local_pdf_path = ensure_pdf_available(pdf_path, s3_handler)

            if not local_pdf_path:
                logger.warning(f"Could not obtain PDF for briefing {row.get('id', '')}")
                stats["pdfs_not_found"] += 1
                continue

            # Extract text from PDF
            logger.info(f"Extracting text from PDF: {os.path.basename(local_pdf_path)}")
            pdf_text = extract_text_from_pdf(local_pdf_path, max_pages=max_pages)

            if pdf_text:
                stats["pdfs_processed"] += 1
                result_df.at[idx, "pdf_text"] = pdf_text
            else:
                stats["pdfs_not_found"] += 1

    # Log statistics
    logger.info(f"PDF text extraction completed:")
    logger.info(f"  Total PDFs processed: {stats['pdfs_processed']}")
    logger.info(f"  PDFs not found: {stats['pdfs_not_found']}")

    return result_df, stats


def prepare_data_for_keyword_analysis(
    df: pd.DataFrame,
    max_pages: Optional[int] = None,
    batch_size: int = 10,
    use_s3: bool = False,
    s3_handler: Optional[ResearchBriefingsToS3] = None,
) -> Tuple[pd.DataFrame, Dict[str, int], pd.DataFrame]:
    """
    Prepare the briefing data for keyword analysis, including PDF text.

    Args:
        df: DataFrame with research briefing data
        max_pages: Maximum number of pages to extract per PDF
        batch_size: Number of PDFs to process in each batch
        use_s3: Whether to use S3 storage
        s3_handler: S3 handler if use_s3 is True

    Returns:
        Tuple containing:
        - DataFrame ready for keyword analysis with 'id' and 'text' columns
        - Dictionary with PDF processing statistics
        - Enhanced DataFrame with PDF text added
    """
    # Check if we already have a text cumulative file in S3
    text_df = None
    if use_s3 and s3_handler:
        text_file_key = f"{s3_handler.prefix}/research_briefings_text.parquet"
        try:
            # Download the text file to a temp location
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = tmp.name
                s3_handler.s3_client.download_file(s3_handler.bucket, text_file_key, tmp_path)

                # Load the existing text data
                text_df = pd.read_parquet(tmp_path)
                logger.info(f"Loaded existing text data for {len(text_df)} briefings from S3")

                # Clean up
                os.unlink(tmp_path)
        except ClientError:
            logger.info("No existing text data found in S3, will extract from PDFs")
        except Exception as e:
            logger.warning(f"Error loading text data from S3: {e}")

    # Find briefings that already have text
    ids_with_text = set()
    if text_df is not None and not text_df.empty:
        # Create a map of ID -> text
        text_map = dict(zip(text_df["id"], text_df["pdf_text"]))

        # Add text to the main DataFrame where available
        for idx, row in df.iterrows():
            if row["id"] in text_map:
                df.at[idx, "pdf_text"] = text_map[row["id"]]
                ids_with_text.add(row["id"])

    # Find briefings that need text extraction
    df_needs_text = df[~df["id"].isin(ids_with_text)].copy()

    if not df_needs_text.empty:
        logger.info(f"Extracting text for {len(df_needs_text)} briefings that don't have text yet")
        # Extract text for briefings that don't have it yet
        text_extracted_df, pdf_stats = prepare_briefing_data_with_pdf_text(
            df_needs_text, max_pages=max_pages, batch_size=batch_size, s3_handler=s3_handler
        )

        # Merge back into main DataFrame
        for idx, row in text_extracted_df.iterrows():
            if pd.notna(row["pdf_text"]):
                df.at[idx, "pdf_text"] = row["pdf_text"]
    else:
        logger.info("All briefings already have text data")
        pdf_stats = {
            "total_briefings": len(df),
            "pdfs_processed": len(ids_with_text),
            "pdfs_not_found": 0,
            "s3_downloads": 0,
        }

    # Create a clean DataFrame with just id and text for analysis
    analysis_df = df[["id"]].copy()

    # For keyword analysis, we'll combine abstract and PDF text
    # But keep track of the source for each text segment
    analysis_df["abstract_text"] = df.apply(
        lambda row: str(row["abstract"]) if pd.notna(row["abstract"]) else "",
        axis=1,
    )

    analysis_df["pdf_content"] = df.apply(
        lambda row: str(row["pdf_text"]) if pd.notna(row["pdf_text"]) else "",
        axis=1,
    )

    # Combined text for keyword detection
    analysis_df["text"] = analysis_df["abstract_text"] + " " + analysis_df["pdf_content"]

    # Remove rows with empty or missing text
    analysis_df = analysis_df[analysis_df["text"].notna() & (analysis_df["text"] != "")]

    # If we're using S3, update the text cumulative file
    if use_s3 and s3_handler:
        # Create a DataFrame with just id and pdf_text
        updated_text_df = df[["id", "pdf_text"]].copy()
        updated_text_df = updated_text_df[updated_text_df["pdf_text"].notna()]

        try:
            # Upload to S3
            upload_obj(updated_text_df, s3_handler.bucket, text_file_key)
            logger.info(f"Updated text cumulative file in S3 with {len(updated_text_df)} entries")
        except Exception as e:
            logger.error(f"Error updating text cumulative file in S3: {e}")

    return analysis_df, pdf_stats, df


def perform_keyword_analysis(
    df: pd.DataFrame, keyword_type: str = "ASF", analysis_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Perform keyword analysis on the research briefings.

    Args:
        df: DataFrame with research briefing data
        keyword_type: Type of keywords to use for analysis (e.g., "ASF", "AFS", "AHL", "X")
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

        # Transform labels to get the expected format
        logger.info("Transforming labels...")
        transformed_df = transform_labels_df(enriched_df)

        # Merge the results back to the original DataFrame
        result_df = pd.merge(df, transformed_df, on="id", how="left")

        # Add a column indicating whether keywords were found
        result_df["has_keywords"] = ~result_df["mission_labels"].isna()

        return result_df

    except Exception as e:
        logger.error(f"Error during keyword analysis: {e}")
        raise


def determine_match_location(row, analysis_df):
    """
    Determine if a keyword match is from abstract or PDF text.

    Args:
        row: Row from matches DataFrame with sentence
        analysis_df: DataFrame with separate abstract and PDF text columns

    Returns:
        String indicating match location: "abstract", "pdf_text", or "both"
    """
    # Get briefing id and sentence
    briefing_id = row["id"]
    sentence = row["sentence"]

    # Get abstract and PDF text for this briefing
    briefing_row = analysis_df.loc[analysis_df["id"] == briefing_id].iloc[0]
    abstract_text = briefing_row["abstract_text"]
    pdf_content = briefing_row["pdf_content"]

    # Check if sentence appears in abstract and/or PDF
    in_abstract = sentence in abstract_text
    in_pdf = sentence in pdf_content

    if in_abstract and in_pdf:
        return "both"
    elif in_abstract:
        return "abstract"
    elif in_pdf:
        return "pdf_text"
    else:
        # If we can't find an exact match, make a best guess
        # This might happen with formatting differences
        if len(abstract_text) < 100:  # Very short or empty abstract
            return "pdf_text"
        elif len(pdf_content) < 100:  # No PDF text
            return "abstract"
        else:
            # Use a more fuzzy approach - look for most of the words in the sentence
            sentence_words = set(sentence.lower().split())
            if len(sentence_words) > 3:  # Only meaningful for longer sentences
                abstract_words = set(abstract_text.lower().split())
                pdf_words = set(pdf_content.lower().split())

                abstract_overlap = len(sentence_words.intersection(abstract_words)) / len(sentence_words)
                pdf_overlap = len(sentence_words.intersection(pdf_words)) / len(sentence_words)

                if abstract_overlap > 0.7 and pdf_overlap > 0.7:
                    return "both"
                elif abstract_overlap > pdf_overlap:
                    return "abstract"
                else:
                    return "pdf_text"

            # Default case
            return "unknown"


def get_detailed_keyword_matches(
    df: pd.DataFrame, keyword_type: str = "ASF", analysis_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Get detailed information about keyword matches in each document's text.

    Args:
        df: DataFrame with research briefing data
        keyword_type: Type of keywords to use for analysis
        analysis_df: Pre-prepared DataFrame for analysis with 'id', 'text', 'abstract_text', and 'pdf_content' columns

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
                        "title": original_row.get("title", ""),
                        "identifier": original_row.get("identifier", ""),
                        "date": original_row.get("date", ""),
                        "mission_labels": keyword_type,
                        "topic_labels": ", ".join(hit_row["category"]),
                        "keywords": [kw for sublist in hit_row["keyword"] for kw in sublist],
                        "sentence": hit_row["sentence"],
                        "marked_sentence": hit_row["marked_sentence"],
                    }
                    detailed_matches.append(match_info)

        except Exception as e:
            logger.error(f"Error getting keyword matches for {row['id']}: {e}")

    # Create DataFrame from matches
    matches_df = pd.DataFrame(detailed_matches)

    # Determine match location for each match
    if not matches_df.empty:
        matches_df["match_location"] = matches_df.apply(lambda row: determine_match_location(row, analysis_df), axis=1)

        # Log match location statistics
        location_counts = matches_df["match_location"].value_counts()
        logger.info(f"Match locations: {dict(location_counts)}")

    return matches_df


def validate_keyword_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that keywords actually appear in the sentences as claimed.

    Args:
        matches_df: DataFrame with keyword matches

    Returns:
        DataFrame with validated keyword matches
    """
    logger.info("Validating keyword matches...")

    valid_matches = []
    invalid_matches = []

    for idx, row in matches_df.iterrows():
        all_valid = True

        for keyword in row["keywords"]:
            # Check if the keyword is actually in the sentence (case insensitive)
            if keyword.lower() not in row["sentence"].lower():
                all_valid = False
                logger.warning(f"Invalid match: '{keyword}' not found in sentence from {row['id']}")
                logger.warning(f"Sentence: {row['sentence']}")
                break

        if all_valid:
            valid_matches.append(row)
        else:
            invalid_matches.append(row)

    logger.info(f"Found {len(valid_matches)} valid matches and {len(invalid_matches)} invalid matches")

    # Return only valid matches
    return pd.DataFrame(valid_matches) if valid_matches else matches_df.head(0)


def update_cumulative_keyword_matches(
    matches_df: pd.DataFrame,
    s3_handler: ResearchBriefingsToS3,
    csv_key: str = "research_briefings_labelstore_keywords.csv",
) -> bool:
    """
    Update the cumulative keyword matches file in S3.

    Args:
        matches_df: DataFrame with new keyword matches
        s3_handler: S3 handler for S3 operations
        csv_key: S3 key for the cumulative CSV file

    Returns:
        True if update was successful, False otherwise
    """
    if matches_df.empty:
        logger.warning("No keyword matches to update")
        return False

    full_key = f"{s3_handler.prefix}/{csv_key}"

    try:
        # Check if file exists in S3
        try:
            s3_handler.s3_client.head_object(Bucket=s3_handler.bucket, Key=full_key)
            file_exists = True
        except ClientError:
            file_exists = False

        if file_exists:
            # Download existing file
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp_path = tmp.name
                s3_handler.s3_client.download_file(s3_handler.bucket, full_key, tmp_path)

                # Load existing data
                existing_df = pd.read_csv(tmp_path)

                # Clean up
                os.unlink(tmp_path)

                # Concatenate with new data
                logger.info(f"Updating keyword matches: {len(existing_df)} existing + {len(matches_df)} new")
                combined_df = pd.concat([existing_df, matches_df], ignore_index=True)

                # Remove duplicates based on same sentence and keyword
                dedup_df = combined_df.drop_duplicates(subset=["id", "sentence", "keywords"], keep="last")

                logger.info(f"After deduplication: {len(dedup_df)} entries")
        else:
            # First time creating file
            logger.info(f"Creating new keyword matches CSV with {len(matches_df)} entries")
            dedup_df = matches_df

        # Upload to S3
        # First save to temp file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            dedup_df.to_csv(tmp_path, index=False)

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
    s3_handler: Optional[ResearchBriefingsToS3] = None,
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
    # Add timestamp to artifact
    artifact_data = {
        "timestamp": datetime.now().isoformat(),
        "run_date": run_date.isoformat(),
        "analysis_type": "keyword_enrichment",
    }

    # Add formatted summary
    artifact_data["summary"] = {
        "total_briefings_analysed": metadata.get("total_briefings", 0),
        "keyword_types_used": metadata.get("keyword_types", []),
        "output_file": os.path.basename(metadata.get("output_file", "")),
        "status": "success" if not metadata.get("error") else "error",
        "error_message": metadata.get("error", ""),
    }

    # Add detailed results
    artifact_data["results"] = {
        "total_keyword_matches": metadata.get("total_matches", 0),
        "unique_briefings_with_matches": metadata.get("unique_briefings_with_matches", 0),
    }

    # Add detailed results for each keyword type
    artifact_data["keyword_results"] = {}
    for keyword_type, result in metadata.get("results", {}).items():
        artifact_data["keyword_results"][keyword_type] = {
            "briefings_with_keywords": result.get("briefings_with_keywords", 0),
            "keyword_matches": result.get("keyword_matches", 0),
        }

    # Add PDF processing stats if available
    if "pdf_stats" in metadata:
        artifact_data["pdf_processing"] = metadata["pdf_stats"]

    # Generate artifact based on storage mode
    if use_s3 and s3_handler:
        # Upload to the same runs directory in S3
        date_str = run_date.strftime("%Y%m%d")
        s3_key = f"{s3_handler.runs_prefix}/enrichment_{date_str}.json"

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


def analyse_research_briefings(
    input_file: str = "research_briefings.parquet",
    output_dir: str = "analysis_output",
    keyword_types: List[str] = ["ASF", "AFS", "AHL", "X", "Nesta"],
    max_pages: Optional[int] = None,
    batch_size: int = 10,
    use_s3: bool = False,
    s3_prefix: str = None,
) -> Dict:
    """
    Analyse research briefings data with keyword analysis.

    Args:
        input_file: Path to the briefings parquet file
        output_dir: Directory to save the analysis results
        keyword_types: List of keyword types to use for analysis
        max_pages: Maximum number of pages to extract from each PDF
        batch_size: Number of PDFs to process in each batch
        use_s3: Whether to use S3 storage
        s3_prefix: S3 prefix (folder path) if use_s3 is True

    Returns:
        Dictionary with analysis metadata
    """
    # Set the run date (used for file naming)
    run_date = datetime.now()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize S3 handler if using S3
    s3_handler = None
    if use_s3:
        s3_handler = ResearchBriefingsToS3(s3_prefix)

    # Load research briefings from parquet
    df = load_research_briefings(input_file, use_s3, s3_handler)

    if df.empty:
        error_msg = f"No research briefings found in {input_file}"
        logger.error(error_msg)
        metadata = {"error": error_msg, "run_date": run_date.isoformat(), "status": "error"}
        generate_pipeline_artifact(metadata, output_dir, run_date, use_s3=use_s3, s3_handler=s3_handler)
        return metadata

    # Prepare data for keyword analysis (including PDF text)
    logger.info("Preparing data for keyword analysis (including PDF text)...")
    analysis_df, pdf_stats, enhanced_df = prepare_data_for_keyword_analysis(
        df=df, max_pages=max_pages, batch_size=batch_size, use_s3=use_s3, s3_handler=s3_handler
    )

    # Collect all matches across keyword types
    all_matches = []
    all_results = {}

    for keyword_type in keyword_types:
        try:
            # Perform keyword analysis
            result_df = perform_keyword_analysis(df, keyword_type, analysis_df)

            # Get detailed keyword matches
            matches_df = get_detailed_keyword_matches(df, keyword_type, analysis_df)

            if not matches_df.empty:
                # Validate the matches
                valid_matches_df = validate_keyword_matches(matches_df)

                # Add to collection of all matches
                all_matches.append(valid_matches_df)

                # Store results
                all_results[keyword_type] = {
                    "total_briefings": len(df),
                    "briefings_with_keywords": int(result_df["has_keywords"].sum()),
                    "keyword_matches": len(valid_matches_df),
                }

                logger.info(f"Found {len(valid_matches_df)} valid matches for {keyword_type}")
            else:
                logger.info(f"No keyword matches found for {keyword_type}")
                all_results[keyword_type] = {
                    "total_briefings": len(df),
                    "briefings_with_keywords": 0,
                    "keyword_matches": 0,
                }

        except Exception as e:
            logger.error(f"Error processing {keyword_type} keyword analysis: {e}")
            all_results[keyword_type] = {"error": str(e)}

    # Combine all matches
    if all_matches:
        combined_matches = pd.concat(all_matches, ignore_index=True)
        logger.info(f"Combined {len(combined_matches)} matches from all keyword types")

        # Save to CSV file
        date_str = run_date.strftime("%Y%m%d")
        local_output_file = os.path.join(output_dir, f"research_briefings_enriched_{date_str}.csv")
        combined_matches.to_csv(local_output_file, index=False)
        logger.info(f"Saved all keyword matches locally to {local_output_file}")

        # If using S3, also update the cumulative keyword matches file
        if use_s3 and s3_handler:
            update_cumulative_keyword_matches(combined_matches, s3_handler)

        # Calculate summary statistics
        unique_briefings = len(combined_matches["id"].unique())
        total_matches = len(combined_matches)
    else:
        logger.warning("No matches found for any keyword type")
        # Create empty file
        combined_matches = pd.DataFrame(
            columns=[
                "id",
                "title",
                "identifier",
                "date",
                "mission_labels",
                "topic_labels",
                "keywords",
                "sentence",
                "marked_sentence",
                "match_location",
            ]
        )
        date_str = run_date.strftime("%Y%m%d")
        local_output_file = os.path.join(output_dir, f"research_briefings_enriched_{date_str}.csv")
        combined_matches.to_csv(local_output_file, index=False)
        logger.info(f"Saved empty matches file locally to {local_output_file}")

        unique_briefings = 0
        total_matches = 0

    # Save overall metadata
    metadata = {
        "total_briefings": len(df),
        "keyword_types": keyword_types,
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
        "run_date": run_date.isoformat(),
        "output_file": local_output_file,
        "total_matches": total_matches,
        "unique_briefings_with_matches": unique_briefings,
        "pdf_stats": {
            "total_briefings": pdf_stats["total_briefings"],
            "pdfs_processed": pdf_stats["pdfs_processed"],
            "pdfs_not_found": pdf_stats["pdfs_not_found"],
            "max_pages_per_pdf": max_pages if max_pages else "all",
        },
    }

    # Generate pipeline artifact
    artifact_path = generate_pipeline_artifact(metadata, output_dir, run_date, use_s3=use_s3, s3_handler=s3_handler)

    # Add artifact path to returned metadata
    metadata["artifact_path"] = artifact_path

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse Research Briefings")
    parser.add_argument(
        "--input-file",
        "-i",
        default="outputs/research_briefings/research_briefings.parquet",
        help="Path to the research briefings parquet file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="outputs/research_briefings/enrichment",
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
        "--max-pages", type=int, help="Maximum number of pages to extract from each PDF (default: all pages)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Number of PDFs to process in each batch (default: 10)"
    )
    parser.add_argument("--use-s3", action="store_true", help="Store files in S3 instead of locally")
    parser.add_argument(
        "--s3-bucket", default="discovery-iss", help="S3 bucket name (required if --use-s3 is specified)"
    )
    parser.add_argument(
        "--s3-prefix",
        default="data/policy/research_briefings",
        help="S3 prefix (folder path) (required if --use-s3 is specified)",
    )

    args = parser.parse_args()

    # Perform keyword analysis
    analyse_research_briefings(
        input_file=args.input_file,
        output_dir=args.output_dir,
        keyword_types=args.keywords,
        max_pages=args.max_pages,
        batch_size=args.batch_size,
        use_s3=args.use_s3,
        s3_prefix=args.s3_prefix,
    )
