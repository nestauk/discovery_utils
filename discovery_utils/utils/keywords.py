"""
discovery_utils.utils.keywords

This module contains functions for processing and analyzing keywords in text data.
It provides functionality for keyword extraction, sentence splitting,
and label enrichment based on keyword matches.
"""

import os
import re

import numpy as np
import pandas as pd

from discovery_utils.utils import google


from typing import Dict, List, Tuple  # isort: skip


def process_keywords(keywords: List[str], separator: str = ",") -> List[List[str]]:
    """Process a list of keywords and keyword combinations

    Args:
        keywords (List[str]): list of keywords in the format ['keyword_1', 'keyword_2, keyword_3']
        separator (str, optional): Character used to separate keywords. Defaults to ",".

    Returns:
        List[List[str]]: list of lists of keywords in the format ['keyword_1', ['keyword_2', 'keyword_3']]
    """
    return [[word.strip().replace("~", " ") for word in line.split(separator)] for line in keywords]


def get_keywords(
    keyword_type: str,
) -> Dict:
    """
    Fetch keywords from a Google Sheet

    Keywords types correspond to missions and are ASF, AFS, AHL, and X.

    Args:
        keyword_type (str): Type of keywords to fetch (e.g., 'ASF', 'AFS', 'AHL', 'X').

    Returns:
        Dict[str, List[List[str]]]: Dictionary of subcategories and their associated keywords.
    """

    # Process keywords
    keywords_df = google.access_google_sheet(
        os.environ["SHEET_ID_KEYWORDS"],
        keyword_type,
    )
    return (
        keywords_df.assign(keywords_list=lambda df: process_keywords(df["Keywords"].to_list()))
        .groupby("Subcategory")
        .agg(keywords=("keywords_list", list))
        .to_dict()["keywords"]
    )


def split_sentences(texts: list, ids: list) -> Tuple[List]:
    """
    Split a list of texts into sentences and keep track of which text each sentence belongs to

    Args:
        texts (List[str]): A list of strings
        ids (List[str]): A list of identifiers for each text

    Returns:
        Tuple[List[str], List[str]]: the first list contains the sentences, and the second list contains the identifiers
    """
    # precompile the regex for efficiency
    sentence_endings = re.compile(r"(?<=[.!?]) +")
    sentences = []
    sentence_ids = []
    for i, text in enumerate(texts):
        for sentence in sentence_endings.split(text):
            sentences.append(sentence)
            sentence_ids.append(ids[i])
    return sentences, sentence_ids


def find_keyword_hits(keywords: List[List[str]], sentences: List[str]) -> List[bool]:
    """
    Check if sentences contain any of the keywords or keyword combinations.

    Args:
        keywords (List[List[str]]): List of keyword combinations to search for.
        sentences (List[str]): List of sentences to search in.

    Returns:
        List[bool]: List of boolean values indicating whether each sentence contains a keyword hit.
    """

    hits = []
    for text in sentences:
        keyword_hits = True
        for keyword in keywords:
            # Note that we are looking for exact matches
            # Make the matching case insensitive
            keyword_hits = keyword_hits and (keyword.lower() in text.lower())
        hits.append(False or keyword_hits)
    return hits


def enrich_keyword_labels(text_df: pd.DataFrame, keyword_type: str, split_sentences_flag: bool = True) -> pd.DataFrame:
    """
    Enrich text dataframe by adding topic and mission labels based on keyword hits

    Args:
        text_df (pd.DataFrame): DataFrame containing 'id' and 'text' columns.
        keyword_type (str): Type of keywords to use for enrichment.

    Returns:
        pd.DataFrame: DataFrame with added 'topic_label' and 'mission_label' columns.
    """
    # Fetch keywords
    subcategory_to_keywords = get_keywords(keyword_type)
    if split_sentences_flag:
        # Split text into sentences
        sentences, sentence_ids = split_sentences(text_df.text.to_list(), text_df.id.to_list())
        sentence_ids = np.array(sentence_ids)
    else:
        sentences = text_df.text.to_list()
        sentence_ids = np.array(text_df.id.to_list())
    # General terms
    general_term_ids = None

    # to do: first check which texts that contain any general terms

    # Then check for the specific, non-general terms
    hits_df = []
    for topic in subcategory_to_keywords:
        hits = []
        for keywords in subcategory_to_keywords[topic]:
            hits.append(find_keyword_hits(keywords, sentences))

        # Find if any of the subcategory keywords are present in the text
        hits_matrix = np.array(hits).any(axis=0)
        # Get array indices for texts with keywords
        hits_indices = np.where(hits_matrix == True)[0]  # noqa: E712
        if hits_indices.size == 0:
            continue
        # Get corresponding unique ids
        hit_ids = set(sentence_ids[hits_indices])
        # Check if these are general terms and keywords
        if "general terms" in topic:
            general_term_ids = hit_ids.copy()
        else:
            # Save to a dataframe
            hits_df.append(
                pd.DataFrame(
                    data={
                        "id": list(hit_ids),
                        "topic_label": topic,
                    }
                )
            )
    if len(hits_df) == 0:
        return pd.DataFrame(columns=["id", "topic_label", "mission_label"])
    else:
        hits_df = pd.concat(hits_df, ignore_index=True).assign(mission_label=keyword_type)
        # Filter noise using general mission topic area terms
        if general_term_ids is None:
            return hits_df
        else:
            return hits_df.query("id in @general_term_ids").reset_index(drop=True)


def transform_labels_df(
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Group the labels DataFrame by ID and aggregate the labels into sets.

    Args:
        labels_df (pd.DataFrame): DataFrame containing ids and associated mission and topic labels

    Returns:
        pd.DataFrame: DataFrame grouped by id and aggregated label columns
    """
    return (
        labels_df.groupby("id")
        .agg(
            mission_labels=("mission_label", lambda x: set(list(x))),
            topic_labels=("topic_label", lambda x: set(list(x))),
        )
        # Coverting sets to strings
        .assign(
            mission_labels=lambda df: df.mission_labels.apply(lambda x: ",".join(x) if type(x) is set else x),
            topic_labels=lambda df: df.topic_labels.apply(lambda x: ",".join(x) if type(x) is set else x),
        )
        .reset_index()
    )


def enrich_topic_labels(text_df: pd.DataFrame, split_sentences_flag: bool = True) -> pd.DataFrame:
    """
    Enrich text dataframe by adding topic and mission labels for all missions.

    Args:
        text_df (pd.DataFrame): DataFrame containing text to be labelled and ids.

    Returns:
        pd.DataFrame: DataFrame with enriched topic and mission labels.
    """

    labels_df = []
    for mission in ["ASF", "AHL", "AFS", "X", "Nesta"]:
        labels_df.append(enrich_keyword_labels(text_df, mission, split_sentences_flag=split_sentences_flag))
    return pd.concat(labels_df, ignore_index=True).pipe(transform_labels_df)


def get_keyword_hits(speech: str, keywords_dict: dict) -> pd.DataFrame:
    """Get keywords and sentences where they appear in a speech

    Args:
        speech (str): Speech text
        keywords_dict (dict): Dictionary of keywords

    Returns:
        pd.DataFrame: DataFrame with columns 'category', 'keyword', 'sentence', and 'marked_sentence'
    """

    hits_keywords = []
    hits_sentences = []
    hits_categories = []
    marked_sentences = []
    sents = split_sentences([speech], ids=[0])[0]

    # Fetch general filtering keywords
    keywords_general = None
    for cat in keywords_dict:
        if "general terms" in cat:
            keywords_general = keywords_dict[cat]
            general_cat = cat  # noqa: F841
    if keywords_general is not None:
        general_hits = np.array([find_keyword_hits(kw, sents) for kw in keywords_general]).any(axis=0)
    else:
        general_hits = [True] * len(sents)
        general_cat = "not specified"  # noqa: F841

    for cat in keywords_dict:
        for kw in keywords_dict[cat]:
            hits = find_keyword_hits(kw, sents)
            for i, hit in enumerate(hits):
                if hit and general_hits[i]:

                    hits_keywords.append(kw)
                    hits_sentences.append(sents[i])
                    hits_categories.append(cat)

                    # Add asterisks around the full words containing the matched keyword
                    marked_sentence = sents[i]
                    for keyword in kw:
                        # Regex to find substrings and expand to full words
                        pattern = r"\b(\S*" + re.escape(keyword) + r"\S*)\b"
                        marked_sentence = re.sub(pattern, r"*\1*", marked_sentence)

                    marked_sentences.append(marked_sentence)

    df = (
        pd.DataFrame(
            {
                "category": hits_categories,
                "keyword": hits_keywords,
                "sentence": hits_sentences,
                "marked_sentence": marked_sentences,
            }
        )
        .query("category != @general_cat")
        .groupby("sentence")
        # unique category and keyword for each sentence
        .agg(category=("category", list), keyword=("keyword", list), marked_sentence=("marked_sentence", "first"))
        .reset_index()
    )
    return df
