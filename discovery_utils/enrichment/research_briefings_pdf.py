"""
UK Parliament Research Briefings PDF Text Analyzer

This module extends the research_briefings_enrichment.py script to extract text from
PDF documents and include it in the keyword analysis process.
"""

import logging
import os
import tempfile

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd


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
    # First try pdfplumber which handles complex PDFs better
    try:
        import pdfplumber

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
        logger.warning(f"Error using pdfplumber on {pdf_path}: {e}")

        # Fall back to PyPDF2
        try:
            import PyPDF2

            text = ""
            with open(pdf_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                # Determine page range
                pages_to_extract = min(len(pdf_reader.pages), max_pages or float("inf"))

                for page_num in range(int(pages_to_extract)):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text() or ""
                        text += page_text
                        text += "\n\n"  # Add extra newlines between pages
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num+1} in {pdf_path}: {e}")

            return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction also failed for {pdf_path}: {e}")
            return ""


def prepare_briefing_data_with_pdf_text(
    df: pd.DataFrame,
    pdf_dir: str,
    max_pages: Optional[int] = None,
    batch_size: int = 10,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Enhance the DataFrame with text extracted from PDFs in batches.

    Args:
        df: DataFrame with research briefing data
        pdf_dir: Directory containing PDF files
        max_pages: Maximum number of pages to extract per PDF
        batch_size: Number of PDFs to process in each batch
        cache_dir: Directory to cache extracted text (None for no caching)

    Returns:
        DataFrame enhanced with PDF text content
    """
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()

    # Create a new column for PDF text if it doesn't exist
    if "pdf_text" not in result_df.columns:
        result_df["pdf_text"] = ""

    # Create or ensure cache directory exists
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir}")

    # Process in batches to manage memory usage
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size

    logger.info(f"Processing {total_rows} briefings in {num_batches} batches of size {batch_size}")

    # Track stats
    pdfs_processed = 0
    pdfs_from_cache = 0
    pdfs_not_found = 0

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_rows)

        logger.info(f"Processing batch {batch_num+1}/{num_batches} (rows {start_idx+1}-{end_idx})")

        batch_df = df.iloc[start_idx:end_idx]

        # Process each briefing in the batch
        for idx, row in batch_df.iterrows():
            # Try multiple identifier formats to find the PDF
            potential_ids = []

            # First try the ID field
            briefing_id = row.get("id", "")
            if briefing_id:
                potential_ids.append(briefing_id)

            # Then try the identifier field (e.g., SN05035)
            identifier = row.get("identifier", "")
            if identifier:
                potential_ids.append(identifier)

            # Finally, try extracting from the json_file path
            json_file = row.get("json_file", "")
            if json_file:
                base_name = os.path.splitext(os.path.basename(json_file))[0]
                potential_ids.append(base_name)

            # Try each potential ID to find a matching PDF
            pdf_path = None
            for potential_id in potential_ids:
                test_path = os.path.join(pdf_dir, f"{potential_id}.pdf")
                if os.path.exists(test_path):
                    pdf_path = test_path
                    break

            if not pdf_path:
                logger.debug(f"No PDF found for briefing with potential IDs: {potential_ids}")
                pdfs_not_found += 1
                continue

            # Check if we have cached text for this PDF
            pdf_text = None
            cache_hit = False

            if cache_dir:
                # Generate a cache filename based on PDF path and max_pages
                pdf_basename = os.path.basename(pdf_path)
                cache_filename = f"{os.path.splitext(pdf_basename)[0]}"
                if max_pages:
                    cache_filename += f"_p{max_pages}"
                cache_filename += ".txt"

                cache_path = os.path.join(cache_dir, cache_filename)

                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            pdf_text = f.read()
                        cache_hit = True
                        pdfs_from_cache += 1
                        logger.debug(f"Using cached text for {pdf_basename}")
                    except Exception as e:
                        logger.warning(f"Error reading cache file {cache_path}: {e}")

            # Extract text if not found in cache
            if pdf_text is None:
                logger.info(f"Extracting text from PDF: {os.path.basename(pdf_path)}")
                pdf_text = extract_text_from_pdf(pdf_path, max_pages=max_pages)
                pdfs_processed += 1

                # Cache the extracted text
                if cache_dir and pdf_text:
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            f.write(pdf_text)
                        logger.debug(f"Cached extracted text to {cache_path}")
                    except Exception as e:
                        logger.warning(f"Error writing cache file {cache_path}: {e}")

            # Update the DataFrame with the PDF text
            if pdf_text:
                result_df.at[idx, "pdf_text"] = pdf_text

    # Log statistics
    logger.info(f"PDF text extraction completed:")
    logger.info(f"  Total PDFs processed: {pdfs_processed}")
    logger.info(f"  PDFs loaded from cache: {pdfs_from_cache}")
    logger.info(f"  PDFs not found: {pdfs_not_found}")

    return result_df


def prepare_data_for_keyword_analysis(
    df: pd.DataFrame,
    include_pdf_text: bool = False,
    pdf_dir: Optional[str] = None,
    max_pages: Optional[int] = None,
    batch_size: int = 10,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Prepare the briefing data for keyword analysis, optionally including PDF text.

    Args:
        df: DataFrame with research briefing data
        include_pdf_text: Whether to include text extracted from PDFs
        pdf_dir: Directory containing PDF files
        max_pages: Maximum number of pages to extract per PDF
        batch_size: Number of PDFs to process in each batch
        use_cache: Whether to cache extracted PDF text

    Returns:
        DataFrame ready for keyword analysis with 'id' and 'text' columns
    """
    # Create a clean DataFrame with just id and abstract
    analysis_df = df[["id"]].copy()

    # Make sure abstract is included
    if "abstract" in df.columns:
        analysis_df["abstract"] = df["abstract"]
    else:
        analysis_df["abstract"] = ""

    # Add PDF text if requested
    if include_pdf_text and pdf_dir:
        logger.info("Including PDF text in keyword analysis...")

        # Set up cache directory
        cache_dir = None
        if use_cache:
            # Create a temporary directory for cache if not specified
            cache_dir = tempfile.mkdtemp(prefix="pdf_text_cache_")
            logger.info(f"Created temporary cache directory: {cache_dir}")

        # Extract text from PDFs and add to DataFrame
        enhanced_df = prepare_briefing_data_with_pdf_text(
            df, pdf_dir, max_pages=max_pages, batch_size=batch_size, cache_dir=cache_dir
        )

        # Add PDF text column to analysis_df
        analysis_df["pdf_text"] = enhanced_df["pdf_text"]

        # Combine abstract and PDF text
        analysis_df["text"] = analysis_df.apply(
            lambda row: (str(row["abstract"]) if pd.notna(row["abstract"]) else "")
            + " "
            + (str(row["pdf_text"]) if pd.notna(row["pdf_text"]) else ""),
            axis=1,
        )
    else:
        # Just use the abstract
        analysis_df["text"] = analysis_df["abstract"]

    # Remove rows with empty or missing text
    analysis_df = analysis_df[analysis_df["text"].notna() & (analysis_df["text"] != "")]

    # Final DataFrame should only have id and text columns
    return analysis_df[["id", "text"]]
