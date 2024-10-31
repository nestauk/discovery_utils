"""
discovery_utils.getters.crunchbase.py

Getters for Crunchbase data
"""
import logging
import os
import re

import pandas as pd

from discovery_utils.utils import s3


S3_BUCKET = os.environ["S3_BUCKET"]
S3_PREFIX = "data/crunchbase/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CrunchbaseGetter:
    """Class to get Crunchbase data from S3"""

    def __init__(self, use_latest_version: bool = True, data_version: str = None) -> None:
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
        self._investors = None
        self._ipos = None
        self._jobs = None
        self._people = None
        self._people_descriptions = None
        self._degrees = None
        self._organisation_categories = None
        self._category_groups = None
        self._group_to_categories = None

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
