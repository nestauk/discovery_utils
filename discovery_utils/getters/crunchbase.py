"""
discovery_utils.getters.crunchbase.py

Getters for Crunchbase data
"""

import logging
import os
import re

from pathlib import Path
from typing import Dict
from typing import List
from typing import Literal

import pandas as pd

from numpy import dot

from discovery_utils.utils import embeddings
from discovery_utils.utils import s3
from discovery_utils.utils.io import remap_dict


S3_BUCKET = os.environ["S3_BUCKET"]
S3_PREFIX = "data/crunchbase/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CrunchbaseGetter:
    """Class to get Crunchbase data from S3"""

    def __init__(self, use_latest_version: bool = True, data_version: str = None, vector_db_path: Path = None) -> None:
        """Initialise CrunchbaseGetter

        Args:
            use_latest_version (bool, optional): Use the latest version of the data. Defaults to True.
            data_version (str, optional): Version of data to use, should follow format "Crunchbase_YYYY-MM-DD".
        """
        self.s3_client = s3.s3_client()
        self.bucket = S3_BUCKET
        self.s3_prefix = S3_PREFIX
        self.use_latest_version = use_latest_version
        if (data_version is None) and self.use_latest_version:
            self.data_version = self._get_latest_data_version()
        else:
            self.data_version = data_version
        self._organisations = None
        self._organisations_enriched = None
        self._descriptions = None
        self._org_parents = None
        self._funding_rounds = None
        self._funding_rounds_enriched = None
        self._funds = None
        self._acquisitions = None
        self._event_appearances = None
        self._events = None
        self._investors = None
        self._investments = None
        self._ipos = None
        self._jobs = None
        self._people = None
        self._people_descriptions = None
        self._degrees = None
        self._organisation_categories = None
        self._organisation_nesta_categories = None
        self._category_groups = None
        self._group_to_categories = None
        self._embedding_model = None
        self._category_vectors = None
        self._group_vectors = None
        self._latest_grants = None
        self._latest_funding_rounds = None
        self._latest_startups = None
        self._latest_smart_money_investors = None
        # Vector DB
        self.VectorDB = embeddings.VectorDB(
            db_path=vector_db_path,
            db_name="crunchbase-lancedb",
            table_name="company_embeddings",
            model="all-MiniLM-L6-v2",
        )

    def _get_latest_data_version(self) -> str:
        """Find the latest Crunchbase version based on S3 folder timestamps."""
        try:
            logger.info(f"Checking for latest version of data in S3 bucket: {self.bucket}")
            response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=S3_PREFIX, Delimiter="/")
            folders = [content["Prefix"] for content in response.get("CommonPrefixes", [])]

            # Extract the date part from the folder names using regex
            date_folders = []
            date_pattern = re.compile(r"Crunchbase_(\d{4}-\d{2}-\d{2})")

            for folder in folders:
                match = date_pattern.search(folder)
                if match:
                    date_folders.append(match.group(1))  # Extract just the date part

            # Sort the folders by date and get the latest one
            if date_folders:
                latest_date = sorted(date_folders)[-1]
                latest_version = f"Crunchbase_{latest_date}"
                logger.info(f"Latest Crunchbase version found: {latest_version}")
                return latest_version
            else:
                raise ValueError("No valid Crunchbase folders found.")
        except Exception as e:
            logger.error(f"Error fetching or processing folder names from S3: {str(e)}")
            raise

    def _get_table(self, key: str) -> pd.DataFrame:
        """Download parquet table from S3"""
        logger.info(f"Downloading parquet file: {key}")
        try:
            response = s3._download_obj(
                self.s3_client,
                self.bucket,
                key,
                download_as="dataframe",
            )
            logger.info(f"Successfully downloaded and read parquet file: {key}")
            return response
        except Exception as e:
            logger.error(f"Error downloading parquet file {key}: {str(e)}")
            raise

    def _get_cb_table(self, table: str) -> pd.DataFrame:
        """Get Crunchbase data from S3

        Args:
            table (str): Table name to download
            data_version (str, optional): Version of data to download.

        Returns:
            pd.DataFrame: Crunchbase data
        """
        key = f"{S3_PREFIX}{self.data_version}/{table}.parquet"
        return self._get_table(key)

    def _get_enriched_organizations(self) -> pd.DataFrame:
        """Get enriched companies data"""
        key = f"{S3_PREFIX}enriched/organizations_full.parquet"
        return self._get_table(key)

    def _get_enriched_funding_rounds(self) -> pd.DataFrame:
        """Get enriched funding rounds data"""
        key = f"{S3_PREFIX}enriched/funding_rounds_full.parquet"
        return self._get_table(key)

    def _get_enriched_latest_grants(self) -> pd.DataFrame:
        """Get the latest Crunchbase grants data"""
        key = f"{S3_PREFIX}enriched/grants_new_only.parquet"
        return self._get_table(key)

    def _get_enriched_latest_funding_rounds(self) -> pd.DataFrame:
        """Get the latest Crunchbase funding rounds data"""
        key = f"{S3_PREFIX}enriched/funding_rounds_for_slack.parquet"
        return self._get_table(key)

    def _get_enriched_latest_startups(self) -> pd.DataFrame:
        """Get the latest Crunchbase startups data"""
        key = f"{S3_PREFIX}enriched/orgs_for_slack.parquet"
        return self._get_table(key)

    def _get_enriched_latest_smart_money_investors(self) -> pd.DataFrame:
        """Get the latest Crunchbase smart money investors data"""
        key = f"{S3_PREFIX}enriched/smart_money_for_slack.parquet"
        return self._get_table(key)

    @property
    def organisations(self) -> pd.DataFrame:
        """Get Crunchbase organisations data"""
        if self._organisations is None:
            self._organisations = self._get_cb_table("organizations")
        return self._organisations

    @property
    def organisations_enriched(self) -> pd.DataFrame:
        """Get enriched Crunchbase organisations data"""
        if self._organisations_enriched is None:
            self._organisations_enriched = self._get_enriched_organizations()
        return self._organisations_enriched

    @property
    def descriptions(self) -> pd.DataFrame:
        """Get Crunchbase long descriptions"""
        if self._descriptions is None:
            self._descriptions = self._get_cb_table("organization_descriptions")
        return self._descriptions

    @property
    def org_parents(self) -> pd.DataFrame:
        """Get Crunchbase organisation parents"""
        if self._org_parents is None:
            self._org_parents = self._get_cb_table("org_parents")
        return self._org_parents

    @property
    def funding_rounds(self) -> pd.DataFrame:
        """Get Crunchbase funding rounds"""
        if self._funding_rounds is None:
            self._funding_rounds = self._get_cb_table("funding_rounds")
        return self._funding_rounds

    @property
    def funds(self) -> pd.DataFrame:
        """Get Crunchbase funds"""
        if self._funds is None:
            self._funds = self._get_cb_table("funds")
        return self._funds

    @property
    def funding_rounds_enriched(self) -> pd.DataFrame:
        """Get enriched Crunchbase funding rounds"""
        if self._funding_rounds_enriched is None:
            self._funding_rounds_enriched = self._get_enriched_funding_rounds()
        return self._funding_rounds_enriched

    @property
    def acquisitions(self) -> pd.DataFrame:
        """Get Crunchbase acquisitions"""
        if self._acquisitions is None:
            self._acquisitions = self._get_cb_table("acquisitions")
        return self._acquisitions

    @property
    def events(self) -> pd.DataFrame:
        """Get Crunchbase events"""
        if self._events is None:
            self._events = self._get_cb_table("events")
        return self._events

    @property
    def event_appearances(self) -> pd.DataFrame:
        """Get Crunchbase event appearances"""
        if self._event_appearances is None:
            self._event_appearances = self._get_cb_table("event_appearances")
        return self._event_appearances

    @property
    def investors(self) -> pd.DataFrame:
        """Get Crunchbase investors"""
        if self._investors is None:
            self._investors = self._get_cb_table("investors")
        return self._investors

    @property
    def investments(self) -> pd.DataFrame:
        """Get Crunchbase investments"""
        if self._investments is None:
            self._investments = self._get_cb_table("investments")
        return self._investments

    @property
    def ipos(self) -> pd.DataFrame:
        """Get Crunchbase IPOs"""
        if self._ipos is None:
            self._ipos = self._get_cb_table("ipos")
        return self._ipos

    @property
    def jobs(self) -> pd.DataFrame:
        """Get Crunchbase jobs"""
        if self._jobs is None:
            self._jobs = self._get_cb_table("jobs")
        return self._jobs

    @property
    def people(self) -> pd.DataFrame:
        """Get Crunchbase people"""
        if self._people is None:
            self._people = self._get_cb_table("people")
        return self._people

    @property
    def people_descriptions(self) -> pd.DataFrame:
        """Get Crunchbase people descriptions"""
        if self._people_descriptions is None:
            self._people_descriptions = self._get_cb_table("people_descriptions")
        return self._people_descriptions

    @property
    def degrees(self) -> pd.DataFrame:
        """Get Crunchbase degrees"""
        if self._degrees is None:
            self._degrees = self._get_cb_table("degrees")
        return self._degrees

    @property
    def latest_grants(self) -> pd.DataFrame:
        """Get the latest Crunchbase grants data"""
        if self._latest_grants is None:
            self._latest_grants = self._get_enriched_latest_grants()
        return self._latest_grants

    @property
    def latest_funding_rounds(self) -> pd.DataFrame:
        """Get the latest Crunchbase funding rounds data

        Note that that there is a row for each unique company and investor pair.
        This means that one funding round will be represented by multiple rows.
        When aggregating funding data, need to deduplicate by funding_round_id column.
        """
        if self._latest_funding_rounds is None:
            self._latest_funding_rounds = self._get_enriched_latest_funding_rounds()
        return self._latest_funding_rounds

    @property
    def latest_startups(self) -> pd.DataFrame:
        """Get the latest Crunchbase startups data"""
        if self._latest_startups is None:
            self._latest_startups = self._get_enriched_latest_startups()
        return self._latest_startups

    @property
    def latest_smart_money_investors(self) -> pd.DataFrame:
        """Get the latest Crunchbase smart money investors data"""
        if self._latest_smart_money_investors is None:
            self._latest_smart_money_investors = self._get_enriched_latest_smart_money_investors()
        return self._latest_smart_money_investors

    @property
    def unique_funding_round_types(self) -> List[str]:
        """Get unique funding round types"""
        return list(sorted(self.funding_rounds_enriched.investment_type.unique().tolist()))

    @staticmethod
    def _split_list(text_list: str, delimiter: str = ",") -> list:
        """Split a string into a list"""
        if text_list is None:
            return []
        else:
            return [text.strip() for text in text_list.split(delimiter)]

    @property
    def organisation_categories(self) -> pd.DataFrame:
        """Get the mapping between Crunchbase organisations and categories"""
        if self._organisation_categories is None:
            cats = self.organisations_enriched.category_list.to_list()
            cats = [self._split_list(_cats) for _cats in cats]
            self._organisation_categories = pd.DataFrame(
                data={"id": self.organisations_enriched.id.to_list(), "category_list": cats}
            )
        return self._organisation_categories

    @property
    def organisation_nesta_categories(self) -> pd.DataFrame:
        """Get the mapping between Crunchbase organisations and Nesta categories"""
        if self._organisation_nesta_categories is None:
            self._organisation_nesta_categories = (
                self.organisations_enriched[["id", "mission_labels", "topic_labels"]]
                .assign(
                    mission_labels=lambda df: df.mission_labels.apply(
                        lambda x: x.split(",") if (type(x) is str) else []
                    )
                )
                .assign(
                    topic_labels=lambda df: df.topic_labels.apply(lambda x: x.split(",") if (type(x) is str) else [])
                )
            )
        return self._organisation_nesta_categories

    @property
    def category_groups(self) -> pd.DataFrame:
        """Get Crunchbase categories and the groups they belong to"""
        if self._category_groups is None:
            self._category_groups = self._get_cb_table("category_groups")
        return self._category_groups

    @property
    def group_to_categories(self) -> pd.DataFrame:
        """Get Crunchbase group to categories mapping"""
        if self._group_to_categories is None:
            self._group_to_categories = (
                self.category_groups.assign(
                    category_groups_list=lambda df: df.category_groups_list.apply(
                        lambda x: [y.strip() for y in x.split(",")]
                    )
                )
                .explode("category_groups_list")
                .drop_duplicates(["name", "category_groups_list"])
                .rename(columns={"name": "category", "category_groups_list": "group"})
                .sort_values(["group", "category"])
            )[["group", "category"]]
        return self._group_to_categories

    def get_organisations_people(self, organisations_df: pd.DataFrame) -> pd.DataFrame:
        """Get people associated with provided organisations"""
        return (
            organisations_df.merge(
                self.people,
                how="left",
                left_on="id",
                right_on="featured_job_organization_id",
                suffixes=("", "_person"),
            )
            .merge(
                self.people_descriptions[["id", "cb_url", "description"]],
                how="left",
                left_on="id_person",
                right_on="id",
                suffixes=("", "_person_description"),
            )
            .dropna(subset=["name_person"])
        )[
            [
                "id",
                "id_person",
                "name",
                "cb_url",
                "name_person",
                "first_name",
                "last_name",
                "gender",
                "cb_url_person_description",
                "linkedin_url_person",
                "featured_job_title",
                "description",
            ]
        ]

    def get_aggregated_people(self, organisations_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate people associated with provided organisations"""
        org_people_df = self.get_organisations_people(organisations_df).assign(
            job_company=lambda x: x["featured_job_title"].str.cat(x["name"], sep=" - ")
        )

        return org_people_df.groupby(["id_person", "name_person"]).agg(
            n_companies=("id", "count"),
            name=("name", list),
            featured_job_title=("featured_job_title", list),
            job_company=("job_company", list),
        )

    def get_companies_in_categories(
        self, categories: List[str], category_type: Literal["narrow", "broad"] = "narrow"
    ) -> pd.DataFrame:
        """Get all companies belonging to the provided categories

        Note, this is equivalent to an OR operation on the categories.

        Args:
            categories (List[str]): List of categories to filter by
            category_type (Literal["narrow", "broad"], optional): Type of category to filter by. Defaults to "narrow".
                narrow = Crunchbase categories; broad = Crunchbase category groups.
                Use self.category_groups to see the mapping between categories and groups.

        Returns:
            pd.DataFrame: Subset of self.organisations_enriched containing companies in the provided categories
        """
        _orgs_to_narrow_categories_df = self.organisation_categories.explode("category_list")
        if category_type == "narrow":
            matching_ids = set(
                _orgs_to_narrow_categories_df.query("category_list in @categories").id.to_list()
            )  # noqa
        elif category_type == "broad":
            matching_ids = set(  # noqa
                _orgs_to_narrow_categories_df.merge(
                    self.group_to_categories, left_on="category_list", right_on="category"
                )
                .query("group in @categories")
                .id.to_list()
            )
        elif category_type not in ["narrow", "broad"]:
            raise ValueError(f"category_type must be one of ['narrow', 'broad'], not {category_type}.")
        return self.organisations_enriched.query("id in @matching_ids").drop_duplicates(subset="id")

    def get_companies_in_nesta_categories(
        self,
        category_type: Literal["mission_labels", "topic_labels"],
        categories: List[str],
    ) -> pd.DataFrame:
        """Get all companies belonging to the provided categories"""
        matching_ids = (  # noqa
            self.organisation_nesta_categories.explode(category_type)
            .query(f"{category_type} in @categories")
            .id.to_list()
        )
        return self.organisations_enriched.query("id in @matching_ids").drop_duplicates(subset="id")

    def select_funding_rounds(
        self,
        org_ids: List[str] = None,
        funding_round_types: List[str] = None,
        deduplicate: bool = True,
    ) -> pd.DataFrame:
        """Select funding rounds for organisations

        Args:
            org_ids (List[str], optional): List of organisation IDs to filter by. Defaults to None.
            funding_round_types (List[str], optional): List of funding round types to filter by. Defaults to None.
            deduplicate (bool, optional): Deduplicate funding rounds. If False, returns also all investors.
        """
        # Filter by organisation ids
        if org_ids is not None:
            funding_rounds_df = self.funding_rounds_enriched.query("org_id in @org_ids")
        else:
            funding_rounds_df = self.funding_rounds_enriched
        # Filter by funding round types
        if funding_round_types is not None:
            funding_rounds_df = funding_rounds_df.query("investment_type in @funding_round_types")
        # Deduplicate funding rounds
        if deduplicate:
            funding_rounds_df = funding_rounds_df.drop_duplicates(subset=["funding_round_id"])
        return funding_rounds_df

    @property
    def embedding_model(self) -> embeddings.SentenceTransformer:
        """Get the sentence transformer model"""
        if self._embedding_model is None:
            self._embedding_model = embeddings.SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedding_model

    @property
    def category_vectors(self) -> pd.DataFrame:
        """Get the Crunchbase category vectors"""
        if self._category_vectors is None:
            unique_categories = self.group_to_categories.category.unique()
            vectors = self.embedding_model.encode(unique_categories)
            self._category_vectors = pd.DataFrame(data={"category": unique_categories, "vector": list(vectors)})
        return self._category_vectors

    @property
    def group_vectors(self) -> pd.DataFrame:
        """Get the Crunchbase group vectors"""
        if self._group_vectors is None:
            unique_groups = self.group_to_categories.group.unique()
            vectors = self.embedding_model.encode(unique_groups)
            self._group_vectors = pd.DataFrame(data={"group": unique_groups, "vector": list(vectors)})
        return self._group_vectors

    def find_similar_categories(
        self, query: str, n_results: int = 10, category_type: Literal["narrow", "broad"] = "narrow"
    ) -> pd.DataFrame:
        """Find similar categories to the query

        Args:
            query (str): Query to find similar categories to
            n_results (int, optional): Number of results to return. Defaults to 10.
            category_type (Literal["narrow", "broad"], optional): Type of category to search for. Defaults to "narrow".
        """
        query_embedding = self.embedding_model.encode([query])[0]
        if category_type == "narrow":
            vectors_df = self.category_vectors
        elif category_type == "broad":
            vectors_df = self.group_vectors
        elif category_type not in ["narrow", "broad"]:
            raise ValueError(f"category_type must be one of ['narrow', 'broad'], not {category_type}.")
        # calculate similarity
        return (
            vectors_df.assign(similarity=vectors_df.vector.apply(lambda x: dot(query_embedding, x)))
            .sort_values("similarity", ascending=False)
            .drop(columns="vector")
            .head(n_results)
        )

    # def get_organisation_text(self) -> pd.DataFrame:
    #     """Get full available text data for projects"""
    #     text_fields = self._default_text_fields
    #     boilerplate_empty_text = "Abstracts are not currently available in GtR for all funded research. \
    #         This is normally because the abstract was not required at the time of proposal submission, \
    #         but may be because it included sensitive information such as personal details"

    #     columns = ["id"] + text_fields

    #     return (
    #         self.projects[columns]
    #         .copy()
    #         .fillna("")
    #         .astype({field: str for field in text_fields})
    #         .assign(text=lambda df: df[text_fields].apply(lambda x: " ".join(x), axis=1))
    #         .assign(text=lambda df: df.text.str.strip())
    #         .assign(text=lambda df: df.text.apply(lambda x: re.sub(boilerplate_empty_text, "", x)))
    #         .drop(columns=text_fields)
    #     )

    @property
    def vector_db(self) -> embeddings.LanceDBConnection:
        """Get the LanceDB connection"""
        return self.VectorDB.vector_db

    def text_search(self, query: str, n_results: int = 10) -> pd.DataFrame:
        """Search the LanceDB for the query"""
        return self.VectorDB.text_search(query, n_results)

    def vector_search(self, query: str, n_results: int = 10) -> pd.DataFrame:
        """Search the LanceDB for the query"""
        return self.VectorDB.vector_search(query, n_results)


REGION_TO_COUNTRIES = {
    "North America + Australia": ["USA", "CAN", "AUS", "NZL"],
    "South + Central America": [
        "VEN",
        "ARG",
        "BRA",
        "CHL",
        "COL",
        "PER",
        "URY",
        "PRY",
        "ECU",
        "BOL",
        "GUY",
        "SUR",
        "MEX",
        "CRI",
        "SLV",
        "GTM",
        "HND",
        "PAN",
        "NIC",
    ],
    "Europe": [
        "IRL",
        "LUX",
        "CHE",
        "ESP",
        "DEU",
        "FRA",
        "FIN",
        "SWE",
        "NLD",
        "BEL",
        "DNK",
        "CZE",
        "POL",
        "EST",
        "AUT",
        "ITA",
        "ROU",
        "CYP",
        "NOR",
        "PRT",
        "BGR",
        "BLR",
        "SVN",
        "ARM",
        "HUN",
        "ISL",
        "LVA",
        "LTU",
        "HRV",
        "MKD",
        "BIH",
        "SRB",
        "SVK",
        "GEO",
        "MDA",
        "ALB",
        "SMR",
        "AND",
        "GIB",
        "FRO",
        "LIE",
        "IMN",
        "GGY",
        "JEY",
        "ALA",
    ],
    "UK": ["GBR"],
    "Asia": [
        "IND",
        "HKG",
        "ISR",
        "RUS",
        "KOR",
        "SGP",
        "JPN",
        "ARE",
        "CHN",
        "PHL",
        "IDN",
        "THA",
        "TUR",
        "MYS",
        "TWN",
        "PAK",
        "LBN",
        "ARM",
        "BGD",
        "KWT",
        "VNM",
        "MDV",
        "JOR",
        "LKA",
        "IRN",
        "SYR",
        "KAZ",
        "UZB",
        "IRQ",
        "OMN",
        "PSE",
        "TJK",
        "BTN",
        "TLS",
        "MAC",
        "MMR",
        "MNG",
        "KHM",
        "LAO",
        "BRN",
    ],
    "Africa": [
        "ZAF",
        "MUS",
        "EGY",
        "GHA",
        "KEN",
        "NGA",
        "MAR",
        "CIV",
        "ETH",
        "TUN",
        "MOZ",
        "UGA",
        "SEN",
        "ZWE",
        "RWA",
        "SDN",
    ],
    "Middle East": ["SAU", "ARE", "KWT", "QAT", "OMN", "IRQ", "IRN", "SYR", "JOR", "LBN", "ISR", "YEM"],
    "Rest of the World": [None, "BMU", "TTO", "GLP", "CYM", "IMN"],
}

INVESTMENT_STAGES = {
    "early_stage": [
        "pre_seed",
        "seed",
        "angel",
        "series_a",
        "series_b",
        "convertible_note",
        "equity_crowdfunding",
        "product_crowdfunding",
        "grant",
        "non_equity_assistance",
        "initial_coin_offering",
    ],
    "growth_stage": ["series_c", "series_d", "series_e", "series_f", "series_g", "series_h", "series_i", "series_j"],
    "late_stage": ["private_equity", "post_ipo_equity", "post_ipo_debt", "post_ipo_secondary", "secondary_market"],
    "other": ["corporate_round", "debt_financing"],
    "uncategorized": ["series_unknown", "undisclosed"],
}


def country_to_region() -> Dict[str, str]:
    """Get the mapping from countries to regions."""
    return remap_dict(REGION_TO_COUNTRIES)


def investment_type_to_stage() -> dict:
    """Get the mapping from investments to stages"""
    return remap_dict(INVESTMENT_STAGES)
