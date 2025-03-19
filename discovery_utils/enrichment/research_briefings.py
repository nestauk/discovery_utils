"""
Parliament Research Briefings Enrichment

This script analyses research briefings data by performing keyword searches
on both the abstracts and the PDF content from the files downloaded by
the research_briefings_getter.py module.
"""

import glob
import json
import logging
import os
import re
import sys

from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
import pdfplumber

from discovery_utils.utils.keywords import enrich_keyword_labels
from discovery_utils.utils.keywords import get_keyword_hits
from discovery_utils.utils.keywords import get_keywords
from discovery_utils.utils.keywords import transform_labels_df


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("research_briefings_enrichment.log"), logging.StreamHandler()],
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


def find_latest_briefings_file(output_dir: str) -> Optional[str]:
    """
    Find the most recent research briefings parquet file in the given directory.

    Args:
        output_dir: Directory to search for briefings files

    Returns:
        Path to the most recent briefings file, or None if none found
    """

    # Look for parquet files matching the likely pattern
    patterns = [
        "research_briefing*.parquet",  # Matches both briefings and briefings_with_text
        "*briefing*.parquet",  # More generic pattern
        "*.parquet",  # Any parquet file
    ]

    for pattern in patterns:
        search_path = os.path.join(output_dir, pattern)
        matching_files = glob.glob(search_path)

        if matching_files:
            # Sort by modification time (most recent first)
            matching_files.sort(key=os.path.getmtime, reverse=True)
            logger.info(f"Found {len(matching_files)} potential briefings files in {output_dir}")
            logger.info(f"Using most recent file: {matching_files[0]}")
            return matching_files[0]

    return None


def load_research_briefings(briefings_file: str) -> pd.DataFrame:
    """
    Load research briefings from a parquet file.

    Args:
        briefings_file: Path to the briefings parquet file

    Returns:
        DataFrame with research briefing data
    """
    try:
        df = pd.read_parquet(briefings_file)
        logger.info(f"Loaded {len(df)} research briefings from {briefings_file}")
        return df
    except Exception as e:
        logger.error(f"Error loading research briefings from {briefings_file}: {e}")
        return pd.DataFrame()


def prepare_briefing_data_with_pdf_text(
    df: pd.DataFrame, max_pages: Optional[int] = None, batch_size: int = 10
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Enhance the DataFrame with text extracted from PDFs in batches.

    Args:
        df: DataFrame with research briefing data
        max_pages: Maximum number of pages to extract per PDF
        batch_size: Number of PDFs to process in each batch

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
    stats = {"total_briefings": total_rows, "pdfs_processed": 0, "pdfs_not_found": 0}

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_rows)

        logger.info(f"Processing batch {batch_num+1}/{num_batches} (rows {start_idx+1}-{end_idx})")

        batch_df = df.iloc[start_idx:end_idx]

        # Process each briefing in the batch
        for idx, row in batch_df.iterrows():
            pdf_path = row.get("pdf_file")

            if not pdf_path or pd.isna(pdf_path) or not os.path.exists(pdf_path):
                logger.debug(f"No PDF found for briefing {row.get('id', '')}")
                stats["pdfs_not_found"] += 1
                continue

            # Extract text from PDF
            logger.info(f"Extracting text from PDF: {os.path.basename(pdf_path)}")
            pdf_text = extract_text_from_pdf(pdf_path, max_pages=max_pages)
            stats["pdfs_processed"] += 1

            # Update the DataFrame with the PDF text
            if pdf_text:
                result_df.at[idx, "pdf_text"] = pdf_text

    # Log statistics
    logger.info(f"PDF text extraction completed:")
    logger.info(f"  Total PDFs processed: {stats['pdfs_processed']}")
    logger.info(f"  PDFs not found: {stats['pdfs_not_found']}")

    return result_df, stats


def prepare_data_for_keyword_analysis(
    df: pd.DataFrame, max_pages: Optional[int] = None, batch_size: int = 10
) -> Tuple[pd.DataFrame, Dict[str, int], pd.DataFrame]:
    """
    Prepare the briefing data for keyword analysis, including PDF text.

    Args:
        df: DataFrame with research briefing data
        max_pages: Maximum number of pages to extract per PDF
        batch_size: Number of PDFs to process in each batch

    Returns:
        Tuple containing:
        - DataFrame ready for keyword analysis with 'id' and 'text' columns
        - Dictionary with PDF processing statistics
        - Enhanced DataFrame with PDF text added
    """
    # Extract text from PDFs and add to DataFrame
    enhanced_df, pdf_stats = prepare_briefing_data_with_pdf_text(df, max_pages=max_pages, batch_size=batch_size)

    # Create a clean DataFrame with just id and text for analysis
    analysis_df = enhanced_df[["id"]].copy()

    # For keyword analysis, we'll combine abstract and PDF text
    # But keep track of the source for each text segment
    analysis_df["abstract_text"] = enhanced_df.apply(
        lambda row: str(row["abstract"]) if pd.notna(row["abstract"]) else "",
        axis=1,
    )

    analysis_df["pdf_content"] = enhanced_df.apply(
        lambda row: str(row["pdf_text"]) if pd.notna(row["pdf_text"]) else "",
        axis=1,
    )

    # Combined text for keyword detection
    analysis_df["text"] = analysis_df["abstract_text"] + " " + analysis_df["pdf_content"]

    # Remove rows with empty or missing text
    analysis_df = analysis_df[analysis_df["text"].notna() & (analysis_df["text"] != "")]

    return analysis_df, pdf_stats, enhanced_df


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


def generate_pipeline_artifact(
    metadata: Dict[str, Any], output_dir: str, artifact_filename: str = "enrichment_artifact.json"
) -> str:
    """
    Generate a pipeline artifact file from the enrichment metadata.

    Args:
        metadata: Dictionary with enrichment metadata
        output_dir: Directory to save the artifact
        artifact_filename: Name of the artifact file

    Returns:
        Path to the created artifact file
    """
    artifact_path = os.path.join(output_dir, artifact_filename)

    # Add timestamp to artifact
    artifact_data = {
        "timestamp": datetime.now().isoformat(),
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

    # Write to file
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact_data, f, indent=2)

    logger.info(f"Generated enrichment pipeline artifact at {artifact_path}")
    return artifact_path


def analyse_research_briefings(
    briefings_file: str,
    output_dir: str = "analysis_output",
    output_file: str = "research_briefings_enriched.parquet",
    keyword_types: List[str] = ["ASF", "AFS", "AHL", "X", "Nesta"],
    max_pages: Optional[int] = None,
    batch_size: int = 10,
) -> Dict:
    """
    Analyse research briefings data with keyword analysis.

    Args:
        briefings_file: Path to the briefings parquet file
        output_dir: Directory to save the analysis results
        output_file: Name of the output parquet file
        keyword_types: List of keyword types to use for analysis
        max_pages: Maximum number of pages to extract from each PDF
        batch_size: Number of PDFs to process in each batch

    Returns:
        Dictionary with analysis metadata
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load research briefings from parquet
    df = load_research_briefings(briefings_file)

    if df.empty:
        error_msg = f"No research briefings found in {briefings_file}"
        logger.error(error_msg)
        return {"error": error_msg}

    # Save enhanced DataFrame with PDF text added
    logger.info("Preparing data for keyword analysis (including PDF text)...")
    analysis_df, pdf_stats, enhanced_df = prepare_data_for_keyword_analysis(
        df=df, max_pages=max_pages, batch_size=batch_size
    )

    # Update the briefings file with PDF text
    enhanced_file = os.path.join(output_dir, "research_briefings_with_text.parquet")
    enhanced_df.to_parquet(enhanced_file, index=False)
    logger.info(f"Saved enhanced briefings data with PDF text to {enhanced_file}")

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

    # Combine all matches and write to single parquet file
    if all_matches:
        combined_matches = pd.concat(all_matches, ignore_index=True)
        logger.info(f"Combined {len(combined_matches)} matches from all keyword types")

        # Save to parquet file
        output_path = os.path.join(output_dir, output_file)
        combined_matches.to_parquet(output_path, index=False)
        logger.info(f"Saved all keyword matches to {output_path}")

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
        output_path = os.path.join(output_dir, output_file)
        combined_matches.to_parquet(output_path, index=False)
        logger.info(f"Saved empty matches file to {output_path}")

        unique_briefings = 0
        total_matches = 0

    # Save overall metadata
    metadata = {
        "total_briefings": len(df),
        "keyword_types": keyword_types,
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
        "output_file": output_path,
        "total_matches": total_matches,
        "unique_briefings_with_matches": unique_briefings,
        "pdf_stats": {
            "total_briefings": pdf_stats["total_briefings"],
            "pdfs_processed": pdf_stats["pdfs_processed"],
            "pdfs_not_found": pdf_stats["pdfs_not_found"],
            "briefings_with_text": len(analysis_df),
            "max_pages_per_pdf": max_pages if max_pages else "all",
        },
    }

    # Generate pipeline artifact for Prefect
    artifact_path = generate_pipeline_artifact(metadata, output_dir)

    # Add artifact path to returned metadata
    metadata["artifact_path"] = artifact_path

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse Research Briefings")
    parser.add_argument(
        "--briefings-file",
        "-b",
        help="Path to the research briefings parquet file (if not provided, will look in output directory)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="outputs/research_briefings/enrichment",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--output-file", "-f", default="research_briefings_enriched.parquet", help="Name of the output parquet file"
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

    args = parser.parse_args()

    # If no briefings file provided, search in the output directory
    briefings_file = args.briefings_file
    if not briefings_file:
        logger.info(f"No briefings file provided, searching in {args.output_dir}")
        # Create the output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        # Search for the most recent briefings file
        briefings_file = find_latest_briefings_file(args.output_dir)

        if not briefings_file:
            # Check parent directory
            parent_dir = os.path.dirname(args.output_dir)
            logger.info(f"No briefings file found in {args.output_dir}, checking parent directory {parent_dir}")
            briefings_file = find_latest_briefings_file(parent_dir)

            if not briefings_file:
                logger.error("No briefings file found. Please provide a valid file path.")
                sys.exit(1)

    # Perform keyword analysis
    analyse_research_briefings(
        briefings_file=briefings_file,
        output_dir=args.output_dir,
        output_file=args.output_file,
        keyword_types=args.keywords,
        max_pages=args.max_pages,
        batch_size=args.batch_size,
    )
