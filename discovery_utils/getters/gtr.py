"""
discovery_utils.getters.gtr.py

Getters for Gateway to Research data
"""
import datetime
import logging
import os
import re

from pathlib import Path
from typing import Dict
from typing import List

import pandas as pd

from discovery_utils.utils import embeddings
from discovery_utils.utils import s3


S3_BUCKET = os.environ["S3_BUCKET"]
S3_PREFIX = "data/GtR/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GtrGetter:
    """Class to get Gateway to Research data from S3"""

    def __init__(self, use_latest_version: bool = True, data_version: str = None, vector_db_path: Path = None) -> None:
        """Initialise GtrGetter

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
        self._projects = None
        self._organisations = None
        self._persons = None
        self._funds = None
        self._projects_funds = None
        self._projects_persons = None
        self._projects_organisations = None
        self._persons_organisations = None
        self._default_text_fields = ["title", "abstractText", "techAbstractText", "potentialImpact"]
        # Vector DB
        self.VectorDB = embeddings.VectorDB(
            db_path=vector_db_path, db_name="gtr-lancedb", table_name="project_embeddings", model="all-MiniLM-L6-v2"
        )

    def _get_latest_data_version(self) -> str:
        """Find the latest version based on S3 folder timestamps."""
        try:
            logger.info(f"Checking for latest version of data in S3 bucket: {self.bucket}")
            response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=S3_PREFIX, Delimiter="/")
            folders = [content["Prefix"] for content in response.get("CommonPrefixes", [])]

            # Extract the date part from the folder names using regex
            date_folders = []
            date_pattern = re.compile(r"GtR_(\d{4}\d{2}\d{2})")

            for folder in folders:
                match = date_pattern.search(folder)
                if match:
                    date_folders.append(match.group(1))  # Extract just the date part

            # Sort the folders by date and get the latest one
            if date_folders:
                latest_date = sorted(date_folders)[-1]
                latest_version = f"GtR_{latest_date}"
                logger.info(f"Latest version found: {latest_version}")
                return latest_version
            else:
                raise ValueError("No valid folders found.")
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

    def _get_gtr_table(self, table: str) -> pd.DataFrame:
        """Get Gateway to Research data from S3

        Args:
            table (str): Table name to download

        Returns:
            pd.DataFrame
        """
        key = f"{S3_PREFIX}{self.data_version}/{table}.parquet"
        return self._get_table(key)

    @property
    def projects(self) -> pd.DataFrame:
        """Get projects data"""
        if self._projects is None:
            self._projects = self._get_gtr_table("projects")
        return self._projects

    @property
    def organisations(self) -> pd.DataFrame:
        """Get organisations data"""
        if self._organisations is None:
            self._organisations = self._get_gtr_table("organisations")
        return self._organisations

    @property
    def persons(self) -> pd.DataFrame:
        """Get persons data"""
        if self._persons is None:
            self._persons = self._get_gtr_table("persons")
        return self._persons

    @property
    def funds(self) -> pd.DataFrame:
        """Get funds data"""
        if self._funds is None:
            self._funds = self._get_gtr_table("funds")
        return self._funds

    @property
    def projects_funds(self) -> pd.DataFrame:
        """Get projects data enriched with start and end dates, and funding amounts"""
        if self._projects_funds is None:
            self._projects_funds = self._link_projects_to_dates_and_funds()
        return self._projects_funds

    @property
    def projects_persons(self) -> pd.DataFrame:
        """Get projects data linked to persons"""
        if self._projects_persons is None:
            self._link_projects_to_persons_and_organisations()
        return self._projects_persons

    @property
    def projects_organisations(self) -> pd.DataFrame:
        """Get projects data linked to organisations"""
        if self._projects_organisations is None:
            self._link_projects_to_persons_and_organisations()
        return self._projects_organisations

    @property
    def persons_organisations(self) -> pd.DataFrame:
        """Get persons data linked to organisations"""
        if self._persons_organisations is None:
            self._persons_organisations = self._link_persons_to_organisations()
        return self._persons_organisations

    @property
    def projects_enriched(self) -> pd.DataFrame:
        """Get projects data enriched with funds information and urls"""
        return self.projects_funds.merge(self.get_projects_urls(), on="id", how="left")

    @staticmethod
    def _get_links_project_dates_and_funds(links: Dict) -> Dict:
        """Get the start and end dates for a project"""
        start_dates = []
        end_dates = []
        funds_id = []
        # Find the link that's pertinent to the project's funding
        for link in links["link"]:
            if link["rel"] == "FUND":
                start_dates.append(link["start"])
                end_dates.append(link["end"])
                funds_id.append(link["href"].split("/")[-1])

        if len(start_dates) > 1:
            # Unlikely to have multiple funds, but just in case
            logging.warning(f"Multiple funds for one project: {funds_id}")

        if len(start_dates) > 0:
            start_date = min(start_dates)
            start_date = datetime.datetime.fromtimestamp(start_date / 1e3).strftime("%Y-%m-%d")
            end_date = max(end_dates)
            if end_date is not None:
                end_date = datetime.datetime.fromtimestamp(end_date / 1e3).strftime("%Y-%m-%d")
            else:
                end_date = None
            funds_id = funds_id[0]
        else:
            start_date = None
            end_date = None
            funds_id = None
        return {"start": start_date, "end": end_date, "funds_id": funds_id}

    def _link_projects_to_dates_and_funds(self) -> pd.DataFrame:
        """Link projects to their start and end dates, and funding ids"""
        projects_df = (
            # Go through projects and get their start and end dates, and funding ids
            pd.concat(
                [
                    self.projects.drop(columns=["links", "start", "end"]),
                    pd.json_normalize(self.projects["links"].apply(self._get_links_project_dates_and_funds)),
                ],
                axis=1,
            )
            # Get the amount of funding for each project
            .merge(
                self.funds[["id", "valuePounds", "category"]],
                left_on="funds_id",
                right_on="id",
                how="left",
                suffixes=("", "_funds"),
            ).drop(columns=["id_funds"])
        )
        # Normalise the funding data
        projects_df = pd.concat(
            [projects_df.drop(columns=["valuePounds"]), pd.json_normalize(projects_df["valuePounds"])], axis=1
        )
        # Just in case, check that only one currency is used
        return projects_df.rename(columns={"value": "amount", "category": "funds_category"})

    @staticmethod
    def _get_links(links: Dict, endpoint: str, url_position: int = -2) -> pd.DataFrame:
        """Link persons to their organisations

        Args:
            links (Dict): Links dictionary
            endpoint (str): The endpoint for which to find links
            url_position (int, optional): Position of the endpoint in the URL. Defaults to -2.
        """
        extracted_links = []
        for link in links["link"]:
            split_link = link["href"].split("/")
            if split_link[url_position] == endpoint:
                extracted_links.append(
                    {f"{endpoint}_rel": link["rel"], f"{endpoint}_id": split_link[-1], "endpoint": split_link[-2]}
                )
        return extracted_links

    def _get_links_project_organisations_and_persons(self, links: Dict) -> List[Dict]:
        """Get the start and end dates for a project

        Args:
            links (Dict): Links for a project

        Returns:
            Tuple[List[Dict], List[Dict]]: Organisations and persons for the project, in the format:
                ([{"rel": "ORG", "organisation_id": "123"}], [{"rel": "PER", "person_id": "456"}])
                The possible values of "rel" are provided in repo's documentation.

        """
        return self._get_links(links, "organisations", -2) + self._get_links(links, "persons", -2)

    def _link_projects_to_persons_and_organisations(self) -> None:
        """Link projects to their organisations and persons"""
        df = (
            self.projects.assign(
                orgs_persons=self.projects["links"].apply(self._get_links_project_organisations_and_persons)
            )
            .explode("orgs_persons")
            .reset_index(drop=True)
        )
        df = pd.concat([df.drop(columns=["orgs_persons"]), pd.json_normalize(df["orgs_persons"])], axis=1)
        self._projects_persons = (
            df
            # Select only links to persons
            .query("endpoint == 'persons'")[["id", "title", "persons_rel", "persons_id"]]
            # Merge with persons data
            .merge(self.persons, left_on="persons_id", right_on="id", how="left", suffixes=("", "_persons"))
            # Drop unnecessary columns
            .drop(columns=["id_persons"])
            # to do: add organisations for each person
        )
        self._projects_organisations = (
            df
            # Select only links to organisations
            .query("endpoint == 'organisations'")[["id", "title", "organisations_rel", "organisations_id"]]
            # Merge with organisations data
            .merge(
                self.organisations,
                left_on="organisations_id",
                right_on="id",
                how="left",
                suffixes=("", "_organisations"),
            )
            # Drop unnecessary columns
            .drop(columns=["id_organisations"])
        )

    def _link_persons_to_organisations(self) -> None:
        """Link persons to their organisations"""
        df = (
            self.persons.assign(orgs=lambda df: df["links"].apply(lambda x: self._get_links(x, "organisations", -2)))
            .explode("orgs")
            .reset_index(drop=True)
        )
        df = pd.concat([df.drop(columns=["orgs"]), pd.json_normalize(df["orgs"])], axis=1)
        return df.merge(
            self.organisations, left_on="organisations_id", right_on="id", how="left", suffixes=("", "_organisations")
        ).drop(columns=["id_organisations"])

    def get_projects_urls(self) -> pd.DataFrame:
        """Get URLs for projects"""
        return (
            self.projects[["id", "identifiers"]]
            .copy()
            .assign(refs=lambda df: df.identifiers.apply(lambda x: x["identifier"][0]["value"]))
            .assign(url=lambda df: "https://gtr.ukri.org/projects?ref=" + df.refs)
            .drop(columns=["identifiers", "refs"])
        )

    def get_projects_text(self) -> pd.DataFrame:
        """Get full available text data for projects"""
        text_fields = self._default_text_fields
        boilerplate_empty_text = "Abstracts are not currently available in GtR for all funded research. \
            This is normally because the abstract was not required at the time of proposal submission, \
            but may be because it included sensitive information such as personal details"

        columns = ["id"] + text_fields

        return (
            self.projects[columns]
            .copy()
            .fillna("")
            .astype({field: str for field in text_fields})
            .assign(text=lambda df: df[text_fields].apply(lambda x: " ".join(x), axis=1))
            .assign(text=lambda df: df.text.str.strip())
            .assign(text=lambda df: df.text.apply(lambda x: re.sub(boilerplate_empty_text, "", x)))
            .drop(columns=text_fields)
        )

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
