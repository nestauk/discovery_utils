"""
UK Parliament Research Briefings Analysis

This script analyzes research briefings data by performing keyword searches
on the abstracts and other textual content from the metadata file generated
by the research_briefings.py module.
"""

import json
import logging
import os
import re
import sys

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd

from discovery_utils.enrichment.research_briefings_pdf import prepare_data_for_keyword_analysis

# Import relevant functions from the discovery_utils package
from discovery_utils.utils.keywords import enrich_keyword_labels
from discovery_utils.utils.keywords import get_keyword_hits
from discovery_utils.utils.keywords import get_keywords
from discovery_utils.utils.keywords import transform_labels_df


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("research_briefings_analysis.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_research_briefings(metadata_file: str) -> List[Dict]:
    """
    Load research briefings from a metadata file.

    Args:
        metadata_file: Path to the metadata file

    Returns:
        List of research briefing metadata dictionaries
    """
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            briefings = json.load(f)
            logger.info(f"Loaded {len(briefings)} research briefings from {metadata_file}")
            return briefings
    except Exception as e:
        logger.error(f"Error loading research briefings from {metadata_file}: {e}")
        return []


def convert_to_dataframe(briefings: List[Dict]) -> pd.DataFrame:
    """
    Convert a list of research briefing dictionaries to a pandas DataFrame.

    Args:
        briefings: List of research briefing metadata dictionaries

    Returns:
        DataFrame with research briefing data
    """
    # Extract relevant fields for analysis
    data = []
    for briefing in briefings:
        # Create a row for each briefing with the fields we want to analyze
        row = {
            "id": briefing.get("id", ""),
            "title": briefing.get("title", ""),
            "identifier": briefing.get("identifier", ""),
            "abstract": briefing.get("abstract", ""),
            "description": briefing.get("description", ""),
            "htmlsummary": briefing.get("htmlsummary", ""),
            "created": briefing.get("created", ""),
            "modified": briefing.get("modified", ""),
            "date": briefing.get("date", ""),
            "status": briefing.get("status", ""),
            "published": briefing.get("published", ""),
            "type": briefing.get("type", ""),
            "subType": briefing.get("subType", ""),
        }

        # Add topics as a comma-separated string
        topics = briefing.get("topics", [])
        row["topics"] = ", ".join(topics) if topics else ""

        # Extract creator information
        creator = briefing.get("creator", {})
        if creator:
            row["creator_given_name"] = creator.get("givenName", "")
            row["creator_family_name"] = creator.get("familyName", "")
            row["creator_name"] = creator.get("name", "")

        # Add file path
        row["json_file"] = briefing.get("json_file", "")

        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)
    logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    return df


def prepare_for_keyword_analysis(
    df: pd.DataFrame, include_pdf_text: bool = False, pdf_dir: str = None
) -> pd.DataFrame:
    """
    Prepare the DataFrame for keyword analysis.

    Args:
        df: DataFrame with research briefing data
        include_pdf_text: Whether to include text extracted from PDFs
        pdf_dir: Directory containing PDF files

    Returns:
        DataFrame ready for keyword analysis
    """
    # Use the new function from pdf_analyzer.py
    return prepare_data_for_keyword_analysis(df=df, include_pdf_text=include_pdf_text, pdf_dir=pdf_dir)


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

    # Prepare data for keyword analysis if not provided
    if analysis_df is None:
        analysis_df = prepare_for_keyword_analysis(df)

    try:
        # Apply keyword enrichment
        logger.info(f"Enriching labels with {keyword_type} keywords...")
        enriched_df = enrich_keyword_labels(analysis_df, keyword_type)

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


def get_detailed_keyword_matches(
    df: pd.DataFrame, keyword_type: str = "ASF", analysis_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Get detailed information about keyword matches in each document's text.

    Args:
        df: DataFrame with research briefing data
        keyword_type: Type of keywords to use for analysis
        analysis_df: Pre-prepared DataFrame for analysis with 'id' and 'text' columns

    Returns:
        DataFrame with detailed keyword match information
    """
    logger.info(f"Getting detailed keyword matches using {keyword_type} keywords")

    # Get keywords dictionary
    keywords_dict = get_keywords(keyword_type)

    # Use analysis_df if provided, otherwise get text from 'abstract'
    if analysis_df is None:
        analysis_df = prepare_for_keyword_analysis(df)

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
                        "categories": ", ".join(hit_row["category"]),
                        "keywords": [kw for sublist in hit_row["keyword"] for kw in sublist],
                        "sentence": hit_row["sentence"],
                        "marked_sentence": hit_row["marked_sentence"],
                        "match_location": "full_text" if "pdf_text" in analysis_df.columns else "abstract",
                    }
                    detailed_matches.append(match_info)

        except Exception as e:
            logger.error(f"Error getting keyword matches for {row['id']}: {e}")

    return pd.DataFrame(detailed_matches)


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


def analyze_research_briefings(
    metadata_file: str,
    output_dir: str = "analysis_output",
    keyword_types: List[str] = ["ASF", "AFS", "AHL", "X", "Nesta"],
    include_pdf_text: bool = False,
    pdf_dir: Optional[str] = None,
    max_pages: Optional[int] = None,
    batch_size: int = 10,
    use_cache: bool = True,
) -> Dict:
    """
    Analyze research briefings data with keyword analysis.

    Args:
        metadata_file: Path to the metadata file
        output_dir: Directory to save the analysis results
        keyword_types: List of keyword types to use for analysis
        include_pdf_text: Whether to include text extracted from PDFs
        pdf_dir: Directory containing PDF files
        max_pages: Maximum number of pages to extract from each PDF
        batch_size: Number of PDFs to process in each batch
        use_cache: Whether to cache extracted PDF text

    Returns:
        Dictionary with analysis metadata
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load research briefings
    briefings = load_research_briefings(metadata_file)

    if not briefings:
        logger.error(f"No research briefings found in {metadata_file}")
        return {"error": "No research briefings found"}

    # Convert to DataFrame
    df = convert_to_dataframe(briefings)

    # Save the base DataFrame
    base_output_file = os.path.join(output_dir, "research_briefings_base.csv")
    df.to_csv(base_output_file, index=False)
    logger.info(f"Saved base research briefings data to {base_output_file}")

    # Results for each keyword type
    results = {}

    # Prepare data for keyword analysis
    if include_pdf_text and pdf_dir:
        logger.info(f"Including PDF text in analysis")

        # Use the new function for PDF text extraction
        analysis_df = prepare_data_for_keyword_analysis(
            df=df,
            include_pdf_text=include_pdf_text,
            pdf_dir=pdf_dir,
            max_pages=max_pages,
            batch_size=batch_size,
            use_cache=use_cache,
        )
    else:
        # Use the original prepare_for_keyword_analysis function
        analysis_df = prepare_for_keyword_analysis(df)

    analysis_output_file = os.path.join(output_dir, f"research_briefings_df.csv")
    analysis_df.to_csv(analysis_output_file, index=False)

    for keyword_type in keyword_types:
        try:
            # Perform keyword analysis
            result_df = perform_keyword_analysis(df, keyword_type, analysis_df)

            # Save results
            output_file = os.path.join(output_dir, f"research_briefings_{keyword_type.lower()}_keywords.csv")
            result_df.to_csv(output_file, index=False)
            logger.info(f"Saved {keyword_type} keyword analysis to {output_file}")

            # Get detailed keyword matches
            matches_df = get_detailed_keyword_matches(df, keyword_type, analysis_df)

            if not matches_df.empty:
                # Validate the matches
                valid_matches_df = validate_keyword_matches(matches_df)

                # Save valid matches
                matches_file = os.path.join(output_dir, f"research_briefings_{keyword_type.lower()}_matches.csv")
                valid_matches_df.to_csv(matches_file, index=False)
                logger.info(f"Saved validated {keyword_type} keyword matches to {matches_file}")

                # Store results
                results[keyword_type] = {
                    "total_briefings": len(df),
                    "briefings_with_keywords": int(result_df["has_keywords"].sum()),
                    "keyword_matches": len(valid_matches_df),
                }
            else:
                logger.info(f"No keyword matches found for {keyword_type}")
                results[keyword_type] = {
                    "total_briefings": len(df),
                    "briefings_with_keywords": 0,
                    "keyword_matches": 0,
                }

        except Exception as e:
            logger.error(f"Error processing {keyword_type} keyword analysis: {e}")
            results[keyword_type] = {"error": str(e)}

    # Save overall metadata
    metadata = {"total_briefings": len(df), "keyword_types": keyword_types, "results": results}

    metadata_output_file = os.path.join(output_dir, "analysis_metadata.json")
    with open(metadata_output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved analysis metadata to {metadata_output_file}")

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze UK Parliament Research Briefings")
    parser.add_argument("--metadata", "-m", required=True, help="Path to the research briefings metadata file")
    parser.add_argument("--output", "-o", default="analysis_output", help="Output directory for analysis results")
    parser.add_argument(
        "--keywords",
        "-k",
        nargs="+",
        default=["ASF", "AFS", "AHL", "X", "Nesta"],
        help="Keyword types to use for analysis",
    )

    # Add new arguments for PDF text analysis
    parser.add_argument(
        "--include-pdf-text", action="store_true", help="Include text extracted from PDFs in keyword analysis"
    )
    parser.add_argument(
        "--pdf-dir",
        help="Directory containing PDF files (defaults to 'pdfs' subdirectory of metadata file's directory)",
    )
    parser.add_argument(
        "--max-pages", type=int, help="Maximum number of pages to extract from each PDF (default: all pages)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Number of PDFs to process in each batch (default: 10)"
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of extracted PDF text")

    args = parser.parse_args()

    # If PDF directory not specified, use default
    pdf_dir = args.pdf_dir
    if args.include_pdf_text and not pdf_dir:
        # Get directory of metadata file
        metadata_dir = os.path.dirname(os.path.abspath(args.metadata))
        pdf_dir = os.path.join(metadata_dir, "pdfs")
        logger.info(f"Using default PDF directory: {pdf_dir}")

    # Perform analysis
    analyze_research_briefings(
        metadata_file=args.metadata,
        output_dir=args.output,
        keyword_types=args.keywords,
        include_pdf_text=args.include_pdf_text,
        pdf_dir=pdf_dir,
        max_pages=args.max_pages,
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
    )
