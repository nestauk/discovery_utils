"""
discovery_utils.discovery_utils.enrichment.crunchbase.py

Module for data enrichment utils for Crunchbase data.

Usage:
python discovery_utils/enrichment/crunchbase.py

python discovery_utils/enrichment/crunchbase.py --test
"""

import argparse
import os
import re

import dotenv
import numpy as np
import pandas as pd

from currency_converter import CurrencyConverter

from discovery_utils.utils import google


from typing import Dict, Iterator, List, Tuple  # isort: skip

dotenv.load_dotenv()

# To do: these should be defined in some config file
SMART_MONEY_TYPES = ["accelerator", "entrepreneurship_program", "incubator", "startup_competition"]
# Criteria for investment opportunities
MIN_INVESTMENT = 1e3
MAX_INVESTMENT = 10e3
MAX_EMPLOYEE_COUNT = 100
COUNTRIES_SCOPE = [
    "GBR",
    "DEU",
    "ESP",
    "FRA",
    "ROM",
    "NLD",
    "AUT",
    "NOR",
    "GRC",
    "HRV",
    "SWE",
    "CZE",
    "IRL",
    "LTU",
    "HUN",
    "ITA",
    "BGR",
    "POL",
    "SVK",
    "EST",
    "FIN",
    "CHE",
    "DNK",
    "LUX",
    "ISL",
    "SVN",
    "BIH",
    "PRT",
    "GIB",
    "CYP",
    "LVA",
    "USA",
    "CAN",
    "AUS",
    "NZL",
]


# To do: This should probably go to more general utils
def convert_currency(
    funding_df: pd.DataFrame,
    date_column: str,
    amount_column: str,
    currency_column: str,
    converted_column: str = None,
    target_currency: str = "GBP",
) -> pd.DataFrame:
    """
    Convert amount in any currency to a target currency using CurrencyConverter package.

    Deal dates should be provided in the datetime.date format
    NB: Rate conversion for dates before year 2000 is not reliable and hence
    is not carried out (the function returns nulls instead)

    Args:
        funding_df: A dataframe which must have a column for a date and amount to be converted
        date_column: Name of column with deal dates
        amount_column: Name of column with the amounts in the original currency
        currency_column: Name of column with the currency codes (eg, 'USD', 'EUR' etc)
        converted_column: Name for new column with the converted amounts

    Returns:
        Same dataframe with an extra column for the converted amount

    """
    # Column name
    converted_column = f"{amount_column}_{target_currency}" if converted_column is None else converted_column
    # Check if there is anything to convert
    rounds_with_funding = len(funding_df[-funding_df[amount_column].isnull()])
    df = funding_df.copy()
    if rounds_with_funding > 0:
        # Set up the currency converter
        Converter = CurrencyConverter(
            fallback_on_missing_rate=True,
            fallback_on_missing_rate_method="linear_interpolation",
            # If the date is out of bounds (eg, too recent)
            # then use the closest date available
            fallback_on_wrong_date=True,
        )
        # Convert currencies
        converted_amounts = []
        for _, row in df.iterrows():
            # Only convert deals after year 1999
            if (row[date_column].year >= 2000) and (row[currency_column] in Converter.currencies):
                converted_amounts.append(
                    Converter.convert(
                        row[amount_column],
                        row[currency_column],
                        target_currency,
                        date=row[date_column],
                    )
                )
            else:
                converted_amounts.append(np.nan)
        df[converted_column] = converted_amounts
        # For deals that were originally in the target currency, use the database values
        deals_in_target_currency = df[currency_column] == target_currency
        df.loc[deals_in_target_currency, converted_column] = df.loc[deals_in_target_currency, amount_column].copy()
    else:
        # If nothing to convert, copy the values and return
        df[converted_column] = df[amount_column].copy()
    return df


def get_org_funding_rounds(
    organisations: pd.DataFrame, funding_rounds: pd.DataFrame, org_id_column: str = "id"
) -> pd.DataFrame:
    """
    Get the funding rounds for a specific organisation

    Args:
        org_id: Crunchbase organisation identifier
        funding_rounds: Dataframe with funding rounds
        investments: Dataframe with investments
        investors: Dataframe with investors

    Returns:
        Dataframe with the funding rounds for the specified organisation
    """
    # Get the funding rounds for the organisation
    return (
        organisations[[org_id_column, "name"]]
        .rename(columns={org_id_column: "org_id"})
        .merge(funding_rounds, how="left", on="org_id")
    )


def get_funding_round_investors(
    funding_rounds: pd.DataFrame, investments: pd.DataFrame, investors: pd.DataFrame
) -> pd.DataFrame:
    """
    Get the investors involved in the specified funding rounds

    Args:
        funding_rounds: Dataframe with funding rounds
        investments: Dataframe with investments
        investors: Dataframe with investors

    Returns:
        Dataframe with the investors for the specified funding rounds
    """
    investments_cols = [
        "funding_round_id",
        "investor_id",
        "investor_name",
        "investor_type",
        "is_lead_investor",
    ]
    investor_cols = [
        "id",
        "investor_types",
        "cb_url",
    ]
    return funding_rounds.merge(investments[investments_cols], on="funding_round_id", how="left",).merge(
        investors[investor_cols].rename(columns={"id": "investor_id", "cb_url": "investor_url"}),
        on="investor_id",
        how="left",
    )


def _process_keywords(keywords: List[str], separator: str = ",") -> List[List[str]]:
    """Process a list of keywords and keyword combinations

    Args:
        keywords (List[str]): list of keywords in the format ['keyword_1', 'keyword_2, keyword_3']

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
    """
    # Process keywords
    keywords_df = google.access_google_sheet(
        os.environ["SHEET_ID_KEYWORDS"],
        keyword_type,
    )
    return (
        keywords_df.assign(keywords_list=lambda df: _process_keywords(df["Keywords"].to_list()))
        .groupby("Subcategory")
        .agg(keywords=("keywords_list", list))
        .to_dict()["keywords"]
    )


def _enrich_funding_smart_money(
    funding_rounds_enriched: pd.DataFrame,
    investments: pd.DataFrame,
    investors: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich the funding rounds with smart money information"""
    # Fetch the manually curated smart money table
    smart_money_manual_df = google.access_google_sheet(os.environ["SHEET_ID_INVESTORS"], "investors")
    return funding_rounds_enriched.pipe(
        get_funding_round_investors, investments=investments, investors=investors
    ).assign(
        # Check if investment was made by one of the smart money types
        smart_money_auto=lambda df: df.investor_types.str.contains("|".join(SMART_MONEY_TYPES)),
        # Check if investment was made by one of the manually curated smart money investors
        smart_money_manual=lambda df: df.investor_url.isin(smart_money_manual_df["Crunchbase URL"]),
        # Combine the two smart money columns
        smart_money=lambda df: df.smart_money_auto | df.smart_money_manual,
    )


def enrich_funding_rounds(
    funding_rounds: pd.DataFrame,
    investments: pd.DataFrame,
    investors: pd.DataFrame,
    funding_round_ids: Iterator[str] = None,
    cutoff_year: int = 2000,
) -> pd.DataFrame:
    """Enrich the funding rounds with additional information"""
    # If no IDs are specified, assume all funding rounds are needed
    if funding_round_ids is None:
        funding_round_ids = funding_rounds.id.unique()

    return (
        funding_rounds.query("id in @funding_round_ids")
        # More informative column names
        .rename(
            columns={
                "id": "funding_round_id",
                "name": "funding_round_name",
            }
        )
        # Remove really old funding rounds
        .query(f"announced_on > '{cutoff_year}'")
        .sort_values("announced_on")
        .assign(
            # Convert investment amounts to thousands
            raised_amount=lambda df: df["raised_amount"] / 1e3,
            raised_amount_usd=lambda df: df["raised_amount_usd"] / 1e3,
            # Convert date strings to datetimes
            announced_on_date=lambda df: pd.to_datetime(df["announced_on"]),
        )
        # Get years from dates
        .assign(year=lambda df: df["announced_on_date"].apply(lambda x: x.year))
        # Convert USD currency to GBP
        .pipe(
            convert_currency,
            date_column="announced_on_date",
            amount_column="raised_amount",
            currency_column="raised_amount_currency_code",
            converted_column="raised_amount_gbp",
            target_currency="GBP",
        )
        .pipe(
            _enrich_funding_smart_money,
            investments=investments,
            investors=investors,
        )
    )


def _process_max_employees(count_range: str) -> int:
    """
    Convert the employee count range to a single (highest) number

    Args:
        count_range: A string with the employee count range, eg, '51-100', '10000+'

    Returns:
        A single integer with the maximum number of employees
    """
    if count_range is None:
        return 0
    elif type(count_range) == str:
        return int(count_range.split("-")[-1].replace("+", ""))
    else:
        raise ValueError(f"Unexpected value for employee count: {count_range}")


def _enrich_org_total_funding_gbp(
    organisations: pd.DataFrame,
    funding_rounds_enriched: pd.DataFrame,
    date_start: int = None,
    date_end: str = None,
) -> pd.DataFrame:
    """
    Get the total funding in GBP for a set of organisations

    Args:
        funding_rounds_enriched_df: Dataframe with enriched funding rounds
        date_start: Earliest date for the funding rounds to include, in the format 'YYYY-MM-DD'
        date_end: Latest date for the funding rounds to include, in the format 'YYYY-MM-DD'

    Returns:
        Dataframe with columns 'org_id' and 'total_funding_gbp'. Organisations that have no funding
        in the specified time period will not be included in the dataframe
    """
    # If no IDs are specified, assume all organisations are needed
    organisation_ids = funding_rounds_enriched.org_id.unique()

    # Determine the dates to include
    date_start = date_start or funding_rounds_enriched.announced_on.min()
    date_end = date_end or funding_rounds_enriched.announced_on.max()

    # Total funding company has received minus grants
    funding_rounds_df = (
        funding_rounds_enriched.loc[
            funding_rounds_enriched["announced_on"].between(date_start, date_end, inclusive="both")
        ]
        .query("org_id in @organisation_ids")
        .query("investment_type != 'grant'")
        .groupby("org_id")
        .agg(
            investment_funding_gbp=("raised_amount_gbp", "sum"),
            num_investment_rounds=("funding_round_id", "count"),
        )
        .reindex(organisation_ids)
        .reset_index()
        .fillna(0)
    )
    # Grant funding
    grant_funding_rounds_df = (
        funding_rounds_enriched.loc[
            funding_rounds_enriched["announced_on"].between(date_start, date_end, inclusive="both")
        ]
        .query("org_id in @organisation_ids")
        .query("investment_type == 'grant'")
        .groupby("org_id")
        .agg(
            grant_funding_gbp=("raised_amount_gbp", "sum"),
            num_grants=("funding_round_id", "count"),
        )
        .reindex(organisation_ids)
        .reset_index()
        .fillna(0)
    )

    return (
        organisations.merge(funding_rounds_df, how="left", left_on="id", right_on="org_id")
        .drop(columns="org_id")
        .merge(grant_funding_rounds_df, how="left", left_on="id", right_on="org_id")
        .drop(columns="org_id")
        .assign(total_funding_gbp=lambda df: df["investment_funding_gbp"] + df["grant_funding_gbp"])
    )


def _step_function_decay(y: float, X: float) -> float:
    """
    Step function with exponential decay

    Args:
        y: input value (investment)
        X: threshold for maximum investment
    """
    k = -np.log(0.1) / X
    return np.where(y <= X, 1, np.exp(-k * (y - X)))


def _enrich_org_investment_opportunity(organisations_enriched: pd.DataFrame) -> pd.DataFrame:
    """Enrich the organisations with investment opportunity tags"""
    # Check for valid employee count and country
    valid_count = organisations_enriched.employee_count_max <= MAX_EMPLOYEE_COUNT
    in_uk = organisations_enriched.country_code == "GBR"
    in_countries_scope = organisations_enriched.country_code.isin(COUNTRIES_SCOPE)
    # Investment opportunities tags
    investment_opp_metric = (
        organisations_enriched.total_funding_gbp.fillna(0)
        .apply(lambda x: _step_function_decay(x, MAX_INVESTMENT))
        .round(3)
    )
    potential_investment_opp = (
        valid_count & in_uk & (organisations_enriched.total_funding_gbp < MIN_INVESTMENT)
    ).astype(int)
    investment_opp = (valid_count & in_uk & (organisations_enriched.total_funding_gbp >= MIN_INVESTMENT)).astype(
        int
    ) * investment_opp_metric
    investment_foreign_opp = (
        valid_count
        & in_countries_scope
        # for foreign companies, have at least one funding round or grant
        & ((organisations_enriched.num_investment_rounds > 0) | (organisations_enriched.num_grants > 0))
    ).astype(int) * investment_opp_metric
    return organisations_enriched.assign(
        potential_investment_opp=potential_investment_opp,
        investment_opp=investment_opp,
        interesting_foreign_opp=investment_foreign_opp,
        investment_opp_metric=investment_opp_metric,
    )


def _enrich_org_has_smart_money(
    organisations: pd.DataFrame,
    funding_rounds_enriched: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich the organisations with smart money tag"""
    # find orgs that have received smart money
    smart_money_orgs = funding_rounds_enriched.query("smart_money").org_id.unique()
    return organisations.assign(smart_money=lambda df: df.id.isin(smart_money_orgs))


def _enrich_org_is_smart_money(
    organisations_enriched: pd.DataFrame,
    funding_rounds_enriched: pd.DataFrame,
    filter_mission_relevant: bool = True,
) -> pd.DataFrame:
    """Enrich the organisations with investor smart money tag"""
    # Organisations relevant to one of our missions
    if filter_mission_relevant:
        relevant_org_ids = organisations_enriched[
            organisations_enriched.mission_labels.fillna("").str.contains("|".join(["ASF", "AHL", "AFS"]))
        ].id.to_list()
    else:
        relevant_org_ids = organisations_enriched.id.unique()
    # Investors manually set as smart money (keep all)
    manual_ids = funding_rounds_enriched.query("smart_money_manual == True").investor_id.drop_duplicates().to_list()
    # Automatically detected smart money investors (keep only relevant ones)
    auto_ids = set(
        funding_rounds_enriched.query("smart_money_auto == True").investor_id.drop_duplicates()
    ).intersection(set(relevant_org_ids))
    smart_money_df = pd.DataFrame(data={"id": list(set(manual_ids).union(auto_ids)), "smart_money_investor": True})
    return (
        organisations_enriched.merge(smart_money_df, how="left", on="id")
        .astype({"smart_money_investor": "boolean"})
        .fillna({"smart_money_investor": False})
    )


def _split_sentences(texts: list, ids: list) -> Tuple[List]:
    """
    Split a list of texts into sentences and keep track of which text each sentence belongs to

    Args:
        texts: A list of strings
        ids: A list of identifiers for each text

    Returns:
        A tuple with two lists: the first list contains the sentences, and the second list contains the identifiers
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


def _find_keyword_hits(keywords: List[List[str]], sentences: List[str]) -> List[bool]:
    """Check if a text contains any of the keywords or keyword combinatons"""
    hits = []
    for text in sentences:
        keyword_hits = True
        for keyword in keywords:
            # Note that we are looking for exact matches
            keyword_hits = keyword_hits and (keyword in text)
        hits.append(False or keyword_hits)
    return hits


def get_organisation_descriptions(
    organisations: pd.DataFrame,
    organisation_descriptions: pd.DataFrame,
) -> pd.DataFrame:
    """Create a full text description from the short description and the description columns"""
    return (
        organisations[["id", "short_description"]]
        .merge(organisation_descriptions[["id", "description"]], on="id", how="left")
        .dropna(subset=["short_description", "description"], how="all")
        .fillna("")
        .assign(text=lambda df: df["short_description"] + ". " + df["description"])
        .drop(columns=["short_description", "description"])
    )


def _enrich_keyword_labels(
    text_df: pd.DataFrame,
    keyword_type: str,
) -> pd.DataFrame:
    """Enrich organisation data by adding topic and mission labels based on keyword hits"""
    # Fetch keywords
    subcategory_to_keywords = get_keywords(keyword_type)
    # Split text into sentences
    sentences, sentence_ids = _split_sentences(text_df.text.to_list(), text_df.id.to_list())
    sentence_ids = np.array(sentence_ids)
    # General terms
    general_term_ids = None

    # to do: first check which texts that contain any general terms

    # Then check for the specific, non-general terms
    hits_df = []
    for topic in subcategory_to_keywords:
        hits = []
        for keywords in subcategory_to_keywords[topic]:
            hits.append(_find_keyword_hits(keywords, sentences))

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


def _transform_labels_df(
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Group the labels dataframe by organisation ID and aggregate the labels into sets"""
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


def enrich_organisations(
    organisations: pd.DataFrame,
    funding_rounds_enriched: pd.DataFrame,
    organisation_descriptions: pd.DataFrame,
    organisation_ids: Iterator[str] = None,
    enrich_labels: bool = True,
) -> pd.DataFrame:
    """Enrich organisation data"""
    # If no IDs are specified, assume all organisations are needed
    if organisation_ids is None:
        organisation_ids = organisations.id.unique()

    organisations_enriched = (
        organisations.query("id in @organisation_ids")
        .assign(employee_count_max=lambda df: df["employee_count"].apply(_process_max_employees))
        .pipe(_enrich_org_total_funding_gbp, funding_rounds_enriched=funding_rounds_enriched)
        .pipe(_enrich_org_investment_opportunity)
        .pipe(_enrich_org_has_smart_money, funding_rounds_enriched=funding_rounds_enriched)
    )

    if enrich_labels:
        topic_labels = _enrich_topic_labels(organisations_enriched, organisation_descriptions)
        return organisations_enriched.merge(topic_labels, how="left", on="id").pipe(
            _enrich_org_is_smart_money,
            funding_rounds_enriched=funding_rounds_enriched,
            filter_mission_relevant=True,
        )
    else:
        return organisations_enriched.pipe(
            _enrich_org_is_smart_money, funding_rounds_enriched=funding_rounds_enriched, filter_mission_relevant=False
        )


def _enrich_topic_labels(
    organisations: pd.DataFrame,
    organisation_descriptions: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich organisation data by adding topic and mission labels for all missions"""
    text_df = get_organisation_descriptions(organisations, organisation_descriptions)
    labels_df = []
    for mission in ["ASF", "AHL", "AFS", "X"]:
        labels_df.append(_enrich_keyword_labels(text_df, mission))
    return pd.concat(labels_df, ignore_index=True).pipe(_transform_labels_df)


if __name__ == "__main__":

    # Use argparase to get the test flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Use a subset of the data for testing")
    args = parser.parse_args()
    test = args.test

    # To do: For prototyping purposes, loading local files instead of using getters here. Should change to using getters.
    from discovery_utils import PROJECT_DIR

    DATA_DIR = PROJECT_DIR / "discovery_utils/enrichment"
    organisations = pd.read_parquet(DATA_DIR / "organizations.parquet")
    funding_rounds = pd.read_parquet(DATA_DIR / "funding_rounds.parquet")
    investments = pd.read_parquet(DATA_DIR / "investments.parquet")
    investors = pd.read_parquet(DATA_DIR / "investors.parquet")
    organisation_descriptions = pd.read_parquet(DATA_DIR / "organization_descriptions.parquet")

    if test:
        organisation_ids = organisations.head(10).id.unique()
    else:
        organisation_ids = None
    print(f"Enriching data for {len(organisation_ids)} organisations")  # noqa: T001

    funding_rounds_enriched = enrich_funding_rounds(
        funding_rounds,
        investments,
        investors,
    )
    organisations_enriched = enrich_organisations(
        organisations,
        funding_rounds_enriched,
        organisation_descriptions,
        organisation_ids=organisation_ids,
    )

    # Export the results (To do: save to S3)
    funding_rounds_enriched.to_parquet("funding_rounds_enriched.parquet")
    organisations_enriched.to_parquet("organisations_enriched.parquet")
