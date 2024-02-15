"""
Utility functions for dealing with keywords
"""
import re

from typing import List

import nltk
import numpy as np
import pandas as pd


def _load_tokenizer():
    """Load nltk tokenizer"""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    return tokenizer


def process_keywords(keywords: List[str], separator: str = ",") -> List[List[str]]:
    """Process a list of keywords and keyword combinations

    Args:
        keywords (List[str]): list of keywords in the format ['keyword_1', 'keyword_2, keyword_3']

    Returns:
        List[List[str]]: list of lists of keywords in the format ['keyword_1', ['keyword_2', 'keyword_3']]
    """
    return [[word.strip() for word in line.split(",")] for line in keywords]


def _deduplicate_keywords(lst: List[List[str]]) -> List[List[str]]:
    """Deduplicate a list of lists of strings"""
    deduplicated_set = set(tuple(sublist) for sublist in lst)
    deduplicated_lst = [list(item) for item in deduplicated_set]
    return deduplicated_lst


def _find_sentences_with_terms(text: str, terms: List[str], all_terms: bool = True) -> List[str]:
    """Find sentences which contain specified search terms

    Args:
        text (str): text to search
        terms (List[str]): list of terms to search for
        all_terms (bool, optional): whether to search for all terms or any term. Defaults to True.

    Returns:
        List[str]: list of sentences containing the specified terms
    """
    tokenizer = _load_tokenizer()
    # Split text into sentences
    sentences = tokenizer.tokenize(text)
    # Keep sentences with terms
    sentences_with_terms = []
    # Number of terms in the query
    n_terms = len(terms)
    for sentence in sentences:
        terms_detected = 0
        # Check all terms
        for term in terms:
            if term in sentence.lower():
                terms_detected += 1
        # Check if all terms were found
        if (all_terms and (terms_detected == n_terms)) or ((not all_terms) and (terms_detected > 0)):
            sentences_with_terms.append(sentence)
    return sentences_with_terms


def check_keyword_hits(texts: pd.Series, keywords: list) -> List[bool]:
    """Check if a text contains any of the keywords or keyword combinatons

    Args:
        texts (pd.Series): Series of texts to search
        keywords (list): list of keywords or keyword combinations

    Returns:
        List[bool]: list of boolean values (True/False) indicating whether
            the text contains any of the keywords or keyword combinations
    """
    hits = [
        texts.apply(lambda x: _find_sentences_with_terms(x.lower(), keyword)).apply(len).astype(bool)
        for keyword in keywords
    ]
    return list(np.array(hits).sum(axis=0).astype(bool))
