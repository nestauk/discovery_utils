"""
Streamlit app for evaluating text labelling results.

This app allows reviewers to evaluate text labelling results by:
1. Loading data from a Google Sheet
2. Displaying text entries by theme
3. Highlighting relevant keywords
4. Collecting reviewer decisions on relevance
5. Saving results back to the Google Sheet
"""

import os

from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
import streamlit as st

from discovery_utils import logging
from discovery_utils.utils import google
from discovery_utils.utils.keywords import get_keyword_hits
from discovery_utils.utils.keywords import get_keywords


# Set page config
st.set_page_config(
    page_title="Text Labelling Evaluation",
    page_icon="📝",
    layout="wide",
)

# Default Sheet ID and Sheet Name
DEFAULT_SHEET_ID = "1m9_tKyJDaSy2vDWxYVP_9HlfBbGysQUV-xrb1FW3vok"
DEFAULT_SHEET_NAME = "crunchbase_check"

# Session state initialization
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "decisions" not in st.session_state:
    st.session_state.decisions = {}
if "reviewed_count" not in st.session_state:
    st.session_state.reviewed_count = 0
if "theme_keywords" not in st.session_state:
    st.session_state.theme_keywords = {}


def load_data(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    """
    Load data from a Google Sheet.

    Args:
        sheet_id: ID of the Google Sheet
        sheet_name: Name of the sheet in the Google Sheet

    Returns:
        DataFrame containing the data from the sheet
    """
    try:
        df = google.access_google_sheet(sheet_id, sheet_name)
        st.session_state.data_loaded = True
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def filter_by_theme(df: pd.DataFrame, theme: str) -> pd.DataFrame:
    """
    Filter the dataframe by theme.

    Args:
        df: DataFrame containing the data
        theme: Theme to filter by

    Returns:
        DataFrame containing only rows for the specified theme
    """
    if "theme" in df.columns:
        return df[df["theme"] == theme].reset_index(drop=True)
    else:
        st.warning("No 'theme' column found in the data")
        return df


def load_theme_keywords(theme: str) -> Dict:
    """
    Load keywords for the specified theme.

    Args:
        theme: Theme to load keywords for

    Returns:
        Dictionary of keywords for the theme
    """
    try:
        # Map theme names to keyword types (ASF, AFS, AHL, X)
        theme_to_keyword_type = {"ASF": "ASF", "AFS": "AFS", "AHL": "AHL", "X": "X"}

        keyword_type = theme_to_keyword_type.get(theme, theme)  # Use theme as fallback
        keywords = get_keywords(keyword_type)
        return keywords
    except Exception as e:
        st.warning(f"Could not load keywords for theme {theme}: {e}")
        return {}


def highlight_text_with_keywords(text: str, keywords_dict: Dict) -> str:
    """
    Highlight text with keywords.

    Args:
        text: Text to highlight
        keywords_dict: Dictionary of keywords to highlight

    Returns:
        HTML-formatted text with keywords highlighted
    """
    try:
        hits_df = get_keyword_hits(text, keywords_dict)
        if not hits_df.empty:
            return hits_df["marked_sentence"].iloc[0]
        return text
    except Exception as e:
        logging.warning(f"Error highlighting keywords: {e}")
        return text


def save_results(sheet_id: str, sheet_name: str, reviewer_name: str, decisions: Dict) -> None:
    """
    Save the reviewer's decisions back to the Google Sheet.

    Args:
        sheet_id: ID of the Google Sheet
        sheet_name: Name of the sheet in the Google Sheet
        reviewer_name: Name of the reviewer
        decisions: Dictionary mapping row IDs to decisions
    """
    try:
        # Load the current data
        df = google.access_google_sheet(sheet_id, sheet_name)

        # Create a new column for the reviewer's decisions
        column_name = f"is_relevant_{reviewer_name}"

        # Update the dataframe with the reviewer's decisions
        for idx, decision in decisions.items():
            if idx < len(df):
                df.loc[idx, column_name] = decision

        # Upload the updated dataframe back to Google Sheets
        upload_dict = {sheet_name: df}
        google.upload_data_to_gsheet(sheet_id, upload_dict)

        st.success(f"Successfully saved {len(decisions)} decisions to Google Sheet!")
    except Exception as e:
        st.error(f"Error saving results: {e}")


def main():
    """Main function for the Streamlit app."""

    st.title("Text Labelling Evaluation")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Google Sheet details
        sheet_id = st.text_input("Google Sheet ID", value=DEFAULT_SHEET_ID)
        sheet_name = st.text_input("Sheet Name", value=DEFAULT_SHEET_NAME)

        # Reviewer name
        reviewer_name = st.text_input("Reviewer Name", key="reviewer_name")

        # Load data button
        if st.button("Load Data"):
            if not reviewer_name:
                st.warning("Please enter your name before loading data")
            else:
                with st.spinner("Loading data..."):
                    st.session_state.df = load_data(sheet_id, sheet_name)
                    if st.session_state.data_loaded:
                        # display the table as a streamlit table
                        st.dataframe(st.session_state.df)

                        st.session_state.all_themes = sorted(st.session_state.df["theme"].unique())
                        st.success(f"Loaded {len(st.session_state.df)} rows of data")

        # Theme selection
        if st.session_state.data_loaded:
            selected_theme = st.selectbox("Select Theme", options=st.session_state.all_themes)

            # Filter data by theme when changed
            if st.button("Apply Theme Filter"):
                st.session_state.filtered_df = filter_by_theme(st.session_state.df, selected_theme)
                st.session_state.current_index = 0
                st.session_state.theme_keywords = load_theme_keywords(selected_theme)
                st.success(f"Filtered to {len(st.session_state.filtered_df)} entries for theme: {selected_theme}")

        # Progress information
        if st.session_state.data_loaded and "filtered_df" in st.session_state:
            st.write(f"Progress: {st.session_state.reviewed_count}/{len(st.session_state.filtered_df)} reviewed")

            # Save results button
            if st.button("Save Results"):
                if len(st.session_state.decisions) > 0:
                    with st.spinner("Saving results..."):
                        save_results(sheet_id, sheet_name, reviewer_name, st.session_state.decisions)
                else:
                    st.warning("No decisions to save")

    # Main content area
    if st.session_state.data_loaded and "filtered_df" in st.session_state:
        if len(st.session_state.filtered_df) > 0:
            # Display current entry
            if st.session_state.current_index < len(st.session_state.filtered_df):
                current_row = st.session_state.filtered_df.iloc[st.session_state.current_index]
                original_index = st.session_state.filtered_df.index[st.session_state.current_index]

                # Display text with highlighted keywords
                st.header("Review Text")

                # Highlight keywords if available
                if st.session_state.theme_keywords:
                    highlighted_text = highlight_text_with_keywords(
                        current_row["text"], st.session_state.theme_keywords
                    )
                    st.markdown(highlighted_text.replace("*", "**"), unsafe_allow_html=True)
                else:
                    st.write(current_row["text"])

                # Display additional metadata
                st.subheader("Metadata")
                metadata_cols = [col for col in current_row.index if col not in ["text", "is_relevant"]]
                metadata_df = pd.DataFrame([current_row[metadata_cols]]).T.reset_index()
                metadata_df.columns = ["Field", "Value"]
                st.table(metadata_df)

                # Decision buttons
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("👍 Relevant"):
                        st.session_state.decisions[original_index] = True
                        st.session_state.current_index += 1
                        st.session_state.reviewed_count += 1
                        st.experimental_rerun()

                with col2:
                    if st.button("👎 Not Relevant"):
                        st.session_state.decisions[original_index] = False
                        st.session_state.current_index += 1
                        st.session_state.reviewed_count += 1
                        st.experimental_rerun()

                with col3:
                    if st.button("⏭️ Skip"):
                        st.session_state.current_index += 1
                        st.experimental_rerun()

                # Navigation
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.session_state.current_index > 0 and st.button("⬅️ Previous"):
                        st.session_state.current_index -= 1
                        st.experimental_rerun()

                with col2:
                    if st.session_state.current_index < len(st.session_state.filtered_df) - 1 and st.button("➡️ Next"):
                        st.session_state.current_index += 1
                        st.experimental_rerun()

            else:
                st.success("You've reviewed all entries for this theme! Select another theme or save your results.")
        else:
            st.info("No entries found for the selected theme.")
    else:
        st.info("Please load data and select a theme to start the evaluation.")


if __name__ == "__main__":
    main()
