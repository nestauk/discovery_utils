"""
Parliament Oral Evidence Sessions Enrichment

This script processes downloaded oral evidence transcripts from Parliament select committees
and performs keyword analysis on them
"""

import glob
import logging
import os
import re

from datetime import datetime

import pandas as pd

from bs4 import BeautifulSoup

# Import the keyword module functions
from discovery_utils.utils.keywords import enrich_keyword_labels
from discovery_utils.utils.keywords import get_keyword_hits
from discovery_utils.utils.keywords import get_keywords
from discovery_utils.utils.keywords import transform_labels_df


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evidence_keyword_search.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def find_evidence_files(base_dir="uk_parliament_oral_evidence"):
    """Find all HTML evidence files in the specified directory structure."""
    logger.info(f"Searching for evidence files in {base_dir}")

    # Find all HTML files recursively
    html_files = glob.glob(f"{base_dir}/**/*.html", recursive=True)

    logger.info(f"Found {len(html_files)} HTML evidence files")
    return html_files


def extract_metadata_from_path(file_path):
    """
    Extract metadata from the file path and name.

    Expected structure: base_dir/committee_id_committee_name/YYYY-MM-DD_Topic.html
    """
    metadata = {}

    # Get full path components
    path_parts = os.path.normpath(file_path).split(os.sep)

    # Extract committee info from the directory name
    if len(path_parts) > 1:
        committee_dir = path_parts[-2]  # The directory containing the file

        # Try to extract committee ID and name
        committee_match = re.match(r"(\d+)_(.+)", committee_dir)
        if committee_match:
            metadata["committee_id"] = committee_match.group(1)
            metadata["committee"] = committee_match.group(2).replace("_", " ")
        else:
            # If no ID pattern found, just use the directory name
            metadata["committee"] = committee_dir.replace("_", " ")

    # Extract date and topic from the filename
    basename = os.path.basename(file_path)
    date_topic_match = re.match(r"(\d{4}-\d{2}-\d{2})_(.+)\.html", basename)

    if date_topic_match:
        date_str = date_topic_match.group(1)
        metadata["topic"] = date_topic_match.group(2).replace("_", " ")

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
    logger.info(f"Detected document structure: {document_structure}")

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
    This handles the format seen in the screenshots with labeled speakers and questions.
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


def extract_witnesses_from_html(html_content):
    """Extract witness information from the HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")

    witnesses = []

    # Look for "Witness" or "Witnesses" section
    witness_section = soup.find(string=re.compile(r"Witness(es)?", re.IGNORECASE))
    if witness_section:
        # First try to find a dedicated witness section
        section_elem = witness_section.find_parent()

        # Check if there's a witness list after the heading
        witness_list = []
        for elem in section_elem.find_next_siblings():
            # Look for paragraph with witness info
            text = elem.get_text().strip()
            if text and not text.startswith("Q") and not text.startswith("The Chair:"):
                witness_list.append(text)
            # Stop if we hit another heading or question
            if re.match(r"Q\d+", text) or text.startswith("Examination of witness"):
                break

        if witness_list:
            # Process the witness list
            witness_text = " ".join(witness_list)
            # Split by common separators
            witness_parts = re.split(r"[;,]\s*|\s+and\s+", witness_text)
            witnesses.extend([part.strip() for part in witness_parts if part.strip()])

    # If no witnesses found, look for "I:" pattern
    if not witnesses:
        i_patterns = soup.find_all(string=re.compile(r"^\s*I:\s+", re.MULTILINE))
        for pattern in i_patterns:
            text = pattern.strip()
            if text.startswith("I:"):
                name = re.sub(r"^\s*I:\s+", "", text).strip()
                # Extract just the name (before any comma or title)
                if "," in name:
                    name = name.split(",")[0].strip()
                if name and name not in witnesses:
                    witnesses.append(name)

    return witnesses


def process_evidence_files(html_files):
    """Process each HTML file to extract content and metadata."""
    evidence_data = []

    for html_file in html_files:
        logger.info(f"Processing {html_file}")

        try:
            # Read the HTML file
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Extract metadata from file path
            metadata = extract_metadata_from_path(html_file)

            # Extract witnesses from HTML
            witnesses = extract_witnesses_from_html(html_content)
            metadata["witnesses"] = "; ".join(witnesses) if witnesses else ""

            # Extract text content using the appropriate approach
            text_content = extract_text_content(html_content)

            # Combine metadata
            evidence_entry = {
                "id": os.path.basename(html_file),
                "file_path": html_file,
                "text": text_content,
                "word_count": len(text_content.split()),
                **metadata,
            }

            evidence_data.append(evidence_entry)

        except Exception as e:
            logger.error(f"Error processing {html_file}: {e}")

    return pd.DataFrame(evidence_data)


def perform_keyword_analysis(evidence_df, keyword_type="Nesta"):
    """Perform keyword analysis on the evidence content."""
    logger.info(f"Performing keyword analysis using {keyword_type} keywords")

    # Ensure we have the necessary columns
    if "id" not in evidence_df.columns or "text" not in evidence_df.columns:
        logger.error("DataFrame must contain 'id' and 'text' columns")
        return evidence_df

    # Create a copy with just the columns needed for keyword analysis
    analysis_df = evidence_df[["id", "text"]].copy()

    try:
        # Apply keyword enrichment
        logger.info(f"Enriching labels with {keyword_type} keywords...")
        enriched_df = enrich_keyword_labels(analysis_df, keyword_type)

        # Log the enriched DataFrame structure
        logger.info(f"Initial enrichment completed. Columns: {enriched_df.columns.tolist()}")

        # Apply transform_labels_df to get the expected mission_labels and topic_labels columns
        logger.info("Transforming labels...")
        transformed_df = transform_labels_df(enriched_df)

        logger.info(f"Transformation completed. Columns: {transformed_df.columns.tolist()}")

        # Merge the results back
        result_df = pd.merge(evidence_df, transformed_df, on="id", how="left")

        # Add a column indicating whether keywords were found
        result_df["has_keywords"] = ~result_df["mission_labels"].isna()

        return result_df

    except Exception as e:
        logger.error(f"Error during keyword analysis: {e}")
        raise


def get_detailed_keyword_matches(evidence_df, keyword_type="Nesta"):
    """Get detailed information about keyword matches in each document."""
    logger.info(f"Getting detailed keyword matches using {keyword_type} keywords")

    # Get keywords dictionary
    keywords_dict = get_keywords(keyword_type)

    detailed_matches = []

    for idx, row in evidence_df.iterrows():
        try:
            # Search both the main text and witnesses
            search_text = row["text"]
            witnesses_text = row.get("witnesses", "")

            # Get keyword hits for the main text
            hits_df = get_keyword_hits(search_text, keywords_dict)

            # Process hits
            if not hits_df.empty:
                # Format the results
                for _, hit_row in hits_df.iterrows():
                    match_info = {
                        "id": row["id"],
                        "committee": row.get("committee", ""),
                        "topic": row.get("topic", ""),
                        "date": row.get("date_str", ""),
                        "categories": ", ".join(hit_row["category"]),
                        "keywords": [kw for sublist in hit_row["keyword"] for kw in sublist],
                        "sentence": hit_row["sentence"],
                        "marked_sentence": hit_row["marked_sentence"],
                        "match_location": "content",
                    }
                    detailed_matches.append(match_info)

            # Also check witnesses text if it exists
            if witnesses_text:
                # Get keyword hits for witnesses
                witness_hits_df = get_keyword_hits(witnesses_text, keywords_dict)

                if not witness_hits_df.empty:
                    # Format the results
                    for _, hit_row in witness_hits_df.iterrows():
                        match_info = {
                            "id": row["id"],
                            "committee": row.get("committee", ""),
                            "topic": row.get("topic", ""),
                            "date": row.get("date_str", ""),
                            "categories": ", ".join(hit_row["category"]),
                            "keywords": [kw for sublist in hit_row["keyword"] for kw in sublist],
                            "sentence": hit_row["sentence"],
                            "marked_sentence": hit_row["marked_sentence"],
                            "match_location": "witnesses",
                        }
                        detailed_matches.append(match_info)

        except Exception as e:
            logger.error(f"Error getting keyword matches for {row['id']}: {e}")

    return pd.DataFrame(detailed_matches)


def validate_keyword_matches(matches_df):
    """Validate that keywords actually appear in the sentences as claimed."""
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


def search_witnesses_for_keywords(evidence_df, keyword_type="Nesta"):
    """Specifically search witness information for keywords."""
    logger.info(f"Searching witnesses for {keyword_type} keywords")

    # Get keywords dictionary
    keywords_dict = get_keywords(keyword_type)

    witness_matches = []

    for idx, row in evidence_df.iterrows():
        witnesses_text = row.get("witnesses", "")

        if not witnesses_text:
            continue

        # For each witness
        for witness in witnesses_text.split(";"):
            witness = witness.strip()
            if not witness:
                continue

            # Check each keyword against this witness
            for category, keyword_lists in keywords_dict.items():
                for keywords in keyword_lists:
                    for keyword in keywords:
                        if keyword.lower() in witness.lower():
                            match_info = {
                                "id": row["id"],
                                "committee": row.get("committee", ""),
                                "topic": row.get("topic", ""),
                                "date": row.get("date_str", ""),
                                "categories": category,
                                "keywords": [keyword],
                                "witness": witness,
                                "marked_witness": witness.replace(keyword, f"*{keyword}*"),
                            }
                            witness_matches.append(match_info)

    return pd.DataFrame(witness_matches)


def main():
    """Main function to run the keyword search on evidence files."""
    logger.info("Starting Oral Evidence Session Keyword Search")

    # Find evidence files
    html_files = find_evidence_files()

    if not html_files:
        logger.warning("No evidence files found. Please check the directory.")
        return

    # Process evidence files to extract content and metadata
    evidence_df = process_evidence_files(html_files)

    logger.info(f"Processed {len(evidence_df)} evidence files")

    # Save the evidence DataFrame
    evidence_df.to_csv("evidence_data.csv", index=False)
    logger.info("Saved evidence data to evidence_data.csv")

    # Only process Nesta keywords
    keyword_type = "Nesta"

    try:
        # Perform analysis on text content
        result_df = perform_keyword_analysis(evidence_df, keyword_type)

        # Save results
        output_file = f"evidence_keywords_{keyword_type.lower()}.csv"
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved {keyword_type} keyword analysis to {output_file}")

        # Get detailed keyword matches from text
        matches_df = get_detailed_keyword_matches(evidence_df, keyword_type)

        if not matches_df.empty:
            # Validate the matches
            valid_matches_df = validate_keyword_matches(matches_df)

            # Save valid matches
            matches_file = f"evidence_keyword_matches_{keyword_type.lower()}.csv"
            valid_matches_df.to_csv(matches_file, index=False)
            logger.info(f"Saved validated {keyword_type} keyword matches to {matches_file}")
        else:
            logger.info("No keyword matches found in content")

        # Perform specific search for witnesses
        witness_matches_df = search_witnesses_for_keywords(evidence_df, keyword_type)

        if not witness_matches_df.empty:
            # Save witness matches
            witness_file = f"evidence_witness_matches_{keyword_type.lower()}.csv"
            witness_matches_df.to_csv(witness_file, index=False)
            logger.info(f"Saved {keyword_type} witness matches to {witness_file}")
        else:
            logger.info("No keyword matches found in witnesses")

    except Exception as e:
        logger.error(f"Error processing {keyword_type} keyword analysis: {e}")

    logger.info("Keyword search completed successfully")


if __name__ == "__main__":
    main()
