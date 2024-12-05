import pandas as pd
import pytest  # noqa

from discovery_utils.utils import keywords as kw


def test_enrich_keyword_labels():
    texts = [
        "This project focuses on ASF related research. Renewable energy, wind farm, heat pump",
    ]
    ids = ["id_1"]

    dummy_text_df = pd.DataFrame({"id": ids, "text": texts})

    test_df = kw.enrich_keyword_labels(dummy_text_df, "ASF")

    expected_labels = ["Heat pumps", "Renewables - General", "Wind"]

    assert test_df["topic_label"].tolist() == expected_labels


def test_split_sentences():
    texts = [
        "This project focuses on ASF related research. Insulation. Home. Renewable. energy, wind farm, heat pump",
    ]
    ids = ["id_1"]

    dummy_text_df = pd.DataFrame({"id": ids, "text": texts})

    test_df_split = kw.enrich_keyword_labels(dummy_text_df, "ASF", split_sentences_flag=True)

    assert "Renewables - General" not in test_df_split["topic_label"].tolist()

    # It would get labelled 'Energy efficiency' if 'insulation' and 'home' appeared together
    assert "Energy efficiency" not in test_df_split["topic_label"].tolist()

    # This changes if we don't care about the words appearing in the same sentence
    test_df = kw.enrich_keyword_labels(dummy_text_df, "ASF", split_sentences_flag=False)

    assert "Energy efficiency" in test_df["topic_label"].tolist()
