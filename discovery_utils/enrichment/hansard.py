"""
This module provides functionality to process and enrich Hansard debate data.
It includes classes for parsing XML files, extracting speech data, and enriching
the data with keyword and ML labels.
"""

import argparse
import logging
import os
import re

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd

from botocore.exceptions import ClientError
from dotenv import load_dotenv
from lxml import etree

import horizon_scout.inference_bert as hs

from discovery_utils.getters.hansard import HansardGetter
from discovery_utils.utils import s3
from discovery_utils.utils.keywords import enrich_topic_labels


logger = logging.getLogger(__name__)

load_dotenv()
S3_BUCKET = os.getenv("S3_BUCKET")


class HansardProcessor:
    """
    A class to handle processing of raw Hansard debate XML files.
    """

    def __init__(self):
        """
        Initialize the HansardProcessor.
        """
        self.parser = etree.XMLParser(dtd_validation=False)

    def process_debates(self, xml_files: List[str]) -> pd.DataFrame:
        """
        Process multiple debate XML files and return a structured DataFrame.

        Args:
            xml_files (List[str]): List of paths to XML files to process.

        Returns:
            pd.DataFrame: Structured data from all processed debate files.
        """
        logger.info(f"Processing {len(xml_files)} debate files")
        all_speeches = []
        for file in xml_files:
            try:
                speeches = list(self.process_debate_file(file))
                all_speeches.extend(speeches)
            except Exception as e:
                logger.error(f"Error processing file {file}: {str(e)}")

        df = pd.DataFrame(all_speeches)
        logger.info(f"Processed {len(df)} speeches from {len(xml_files)} files")
        return df

    def process_debate_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single debate XML file and yield structured speech data.

        Args:
            file_path (str): Path to the XML file to process.

        Yields:
            Dict[str, Any]: Structured data for each speech in the debate.
        """
        logger.debug(f"Processing debate file: {file_path}")
        try:
            tree = etree.parse(file_path, self.parser)
            root = tree.getroot()

            date = self.extract_date_from_filename(file_path)
            current_major_heading = None
            current_minor_heading = None

            for elem in root.findall("./"):
                if elem.tag == "major-heading":
                    current_major_heading = self.clean_text(elem.text)
                    current_minor_heading = None
                elif elem.tag == "minor-heading":
                    current_minor_heading = self.clean_text(elem.text)
                elif elem.tag == "speech":
                    speech_data = self.process_speech(elem, date)
                    speech_data["major_heading"] = current_major_heading
                    speech_data["minor_heading"] = current_minor_heading
                    yield speech_data
        except Exception as e:
            logger.error(f"Error parsing XML file {file_path}: {str(e)}")

    def process_speech(self, elem: etree.Element, date: str) -> Dict[str, Any]:
        """
        Process a single speech element from the XML.

        Args:
            elem (etree.Element): The XML element representing a speech.
            date (str): The date of the debate.

        Returns:
            Dict[str, Any]: Structured data for the speech.
        """
        return {
            "speech_id": elem.get("id", "NA"),
            "speakername": elem.get("speakername", "NA"),
            "speaker_id": elem.get("speakerid", "NA"),
            "person_id": elem.get("person_id", "NA"),
            "speech": self.clean_text(self.extract_text(elem)),
            "date": date,
            "year": date[:4] if date else None,
        }

    @staticmethod
    def extract_text(elem: etree.Element) -> str:
        """
        Extract text from an XML element, handling line breaks.

        Args:
            elem (etree.Element): The XML element to extract text from.

        Returns:
            str: The extracted text.
        """
        for br in elem.xpath(".//br"):
            br.tail = " " + (br.tail or "")
        return "".join(elem.xpath(".//text()"))

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text by removing extra whitespace.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        return " ".join(line.strip() for line in text.splitlines() if line.strip())

    @staticmethod
    def extract_date_from_filename(file_path: str) -> str:
        """
        Extract the debate date from the filename.

        Args:
            file_path (str): The path to the XML file.

        Returns:
            str: The extracted date in YYYY-MM-DD format, or None if not found.
        """
        filename = os.path.basename(file_path)
        match = re.search(r"debates(\d{4}-\d{2}-\d{2})", filename)
        if match:
            return match.group(1)
        logger.warning(f"Could not extract date from filename: {filename}")
        return None


class HansardEnricher:
    """
    A class to handle enrichment of Hansard debate data with keyword and ML labels.
    """

    def __init__(self, test_flag: bool = False):
        """
        Initialize the HansardEnricher with S3 bucket and prefix information.

        Args:
            bucket (str): The name of the S3 bucket.
            prefix (str): The prefix for all S3 keys.
            test_flag (bool): If True, use CPU instead of GPU for ML models.
        """
        self.bucket = S3_BUCKET
        hs.s3.BUCKET_NAME_RAW = self.bucket
        self.prefix = self.s3_prefix = "data/policy_scanning_data"
        self.s3_client = s3.s3_client()
        self.test_flag = test_flag

        self.bert_models = None
        self.tokenizer = None
        self.initialize_ml_models()

    def _get_s3_key(self, key: str) -> str:
        """Combine prefix and key for S3 operations."""
        return f"{self.prefix}/{key}"

    def initialize_ml_models(self):
        """Initialize BERT models and tokenizer."""
        logger.info("Initializing ML models and tokenizer")
        self.bert_models = hs.load_models(hs.MODEL_PATHS, hs.MISSIONS, gpu=(not self.test_flag))
        self.tokenizer = hs.AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    def prepare_text_df(self, debates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare text DataFrame for enrichment.

        Args:
            debates_df (pd.DataFrame): The debates DataFrame.

        Returns:
            pd.DataFrame: Prepared text DataFrame with 'id' and 'text' columns.
        """
        return debates_df[["speech_id", "speech"]].rename({"speech_id": "id", "speech": "text"}, axis=1)

    def apply_keyword_labeling(self, text_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply keyword labeling to the text DataFrame.

        Args:
            text_df (pd.DataFrame): The prepared text DataFrame.

        Returns:
            pd.DataFrame: DataFrame with keyword labels.
        """
        logger.info("Running keyword labeling")
        return enrich_topic_labels(text_df).merge(text_df, on="id")

    def apply_ml_labeling(self, text_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ML labeling to the text DataFrame.

        Args:
            text_df (pd.DataFrame): The prepared text DataFrame.

        Returns:
            pd.DataFrame: DataFrame with ML labels.
        """
        logger.info("Running ML labeling")
        if self.bert_models is None or self.tokenizer is None:
            self.initialize_ml_models()
        return hs.bert_add_predictions(text_df, "text", self.bert_models, self.tokenizer, hs.MISSIONS, 10)

    def merge_labelstores(self, existing_store: pd.DataFrame, new_labels: pd.DataFrame) -> pd.DataFrame:
        """
        Merge existing labelstore with new labels.

        Args:
            existing_store (pd.DataFrame): Existing labelstore.
            new_labels (pd.DataFrame): New labels to be added.

        Returns:
            pd.DataFrame: Merged labelstore.
        """
        combined = pd.concat([existing_store, new_labels], ignore_index=True)
        return combined.drop_duplicates(subset="id", keep="last").reset_index(drop=True)

    def upload_labelstore(self, labelstore: pd.DataFrame, is_keyword: bool):
        """
        Upload labelstore to S3.

        Args:
            labelstore (pd.DataFrame): The labelstore to upload.
            is_keyword (bool): If True, upload keyword labelstore. Otherwise, upload ML labelstore.

        Raises:
            ClientError: If there's an error uploading to S3.
        """
        file_name = "HansardDebates_LabelStore_keywords.csv" if is_keyword else "HansardDebates_LabelStore_ml.csv"
        s3_key = self._get_s3_key(f"enriched/{file_name}")
        try:
            s3.upload_obj(labelstore, self.bucket, s3_key)
            logger.info(f"Successfully uploaded labelstore: {s3_key}")
        except ClientError as e:
            logger.error(f"Error uploading labelstore to S3: {e}")
            raise

    def append_and_upload_debates(
        self, existing_debates: pd.DataFrame, new_debates: pd.DataFrame, reset: bool = False
    ) -> None:
        """
        Append new debates to the existing debates dataset and upload to S3.

        Args:
            existing_debates (pd.DataFrame): The existing Hansard debates dataset.
            new_debates (pd.DataFrame): The new debates to be appended.
            reset (bool): If true uploads only debates processed during current run.

        Raises:
            ClientError: If there's an error uploading to S3.
        """
        logger.info("Appending new debates to existing dataset and uploading to S3")
        try:
            if not reset:
                # Append new debates to existing debates
                combined_debates = pd.concat([existing_debates, new_debates], ignore_index=True)

                # Remove duplicates if any, based on speech_id
                combined_debates = combined_debates.drop_duplicates(subset=["speech_id"], keep="last").reset_index(
                    drop=True
                )
            else:
                combined_debates = new_debates

            if not self.test_flag:
                # Upload the combined dataset to S3
                s3_key = self._get_s3_key("enriched/HansardDebates.parquet")
                s3.upload_obj(combined_debates, self.bucket, s3_key)
                logger.info(f"Successfully uploaded appended debates to S3: {s3_key}")
            else:
                logger.info("Test mode: Skipping debate upload")

            return combined_debates
        except ClientError as e:
            logger.error(f"Error uploading appended debates to S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in appending and uploading debates: {e}")
            raise

    def enrich_data(
        self,
        hansard_debates: pd.DataFrame,
        keyword_labelstore: pd.DataFrame,
        ml_labelstore: pd.DataFrame,
        latest_debates: pd.DataFrame,
        incremental: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Enrich Hansard debate data with keyword and ML labels.

        Args:
            hansard_debates (pd.DataFrame): Existing Hansard debates.
            keyword_labelstore (pd.DataFrame): Existing keyword labels.
            ml_labelstore (pd.DataFrame): Existing ML labels.
            latest_debates (pd.DataFrame): Latest debate data to be processed.
            incremental (bool): If True, only process the latest debates.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Updated keyword and ML labelstores.

        Raises:
            Exception: If there's an error in the enrichment process.
        """
        logger.info("Starting data enrichment process")
        try:
            if self.test_flag:
                logger.info("Test mode: Processing 10 speeches")
                if len(latest_debates) > 10:
                    latest_debates = latest_debates.sample(n=10)
                    hansard_debates = hansard_debates.sample(n=10)

            text_df = self.prepare_text_df(latest_debates if incremental else hansard_debates)

            new_keyword_labels = self.apply_keyword_labeling(text_df)
            new_ml_labels = self.apply_ml_labeling(text_df)

            if incremental:
                updated_keyword_labelstore = self.merge_labelstores(keyword_labelstore, new_keyword_labels)
                updated_ml_labelstore = self.merge_labelstores(ml_labelstore, new_ml_labels)
            else:
                logger.info("Full run: Replacing existing labelstores")
                updated_keyword_labelstore = new_keyword_labels
                updated_ml_labelstore = new_ml_labels

            if not self.test_flag:
                self.upload_labelstore(updated_keyword_labelstore, is_keyword=True)
                self.upload_labelstore(updated_ml_labelstore, is_keyword=False)
            else:
                logger.info("Test mode: Skipping labelstore upload")

            logger.info("Data enrichment process completed successfully")
            return new_keyword_labels, new_ml_labels
        except Exception as e:
            logger.error(f"Error in data enrichment process: {str(e)}")
            raise

    def generate_enrichment_report(self, keyword_labelstore: pd.DataFrame, ml_labelstore: pd.DataFrame) -> dict:
        """
        Generate a report summarising the enrichment process.

        Args:
            keyword_labelstore (pd.DataFrame): The keyword labelstore.
            ml_labelstore (pd.DataFrame): The ML labelstore.

        Returns:
            dict: A dictionary containing summary statistics of the enrichment process.
        """
        return {
            "total_debates": len(keyword_labelstore),
            "keyword_label_distribution": keyword_labelstore["topic_labels"].value_counts().to_dict(),
            "ml_label_distribution": {
                "AFS": ml_labelstore["afs_bert_preds"].sum(),
                "ASF": ml_labelstore["asf_bert_preds"].sum(),
                "AHL": ml_labelstore["ahl_bert_preds"].sum(),
            },
        }


def run_hansard_enrichment(incremental: bool = True, test: bool = False, reset: bool = False):
    """
    Run the Hansard enrichment pipeline.

    Args:
        incremental (bool): If True, only process new debates. If False, process all debates.
    """
    logger.info(f"Starting Hansard enrichment pipeline in {'incremental' if incremental else 'full'} mode")
    logger.info(f"Test mode: {'ON' if test else 'OFF'}")

    try:

        # Initialize components
        getter = HansardGetter()
        processor = HansardProcessor()
        enricher = HansardEnricher(test_flag=test)

        if reset:
            getter.delete_labelstore(keywords=True)
            getter.delete_labelstore(keywords=False)
            logger.info("Labelstores have been reset. Proceeding with full run.")
            incremental = False

        # Retrieve data
        logger.info("Retrieving data from S3")
        existing_debates = getter.get_debates_parquet()
        keyword_labelstore = getter.get_labelstore(keywords=True)
        ml_labelstore = getter.get_labelstore(keywords=False)

        # Switch to full mode if labelstores do not exist
        if keyword_labelstore.empty and ml_labelstore.empty:
            incremental = False

        if incremental:
            xml_files = getter.get_latest_debates()
        else:
            xml_files = getter.get_all_debates()

        if not xml_files:
            logger.info("No new debates to process. Exiting.")
            return

        # Process debates
        logger.info("Processing debate files")
        processed_debates = processor.process_debates(xml_files)

        # Enrich data
        logger.info("Enriching debate data")
        new_keyword_labels, new_ml_labels = enricher.enrich_data(
            existing_debates, keyword_labelstore, ml_labelstore, processed_debates, incremental
        )

        # Append new debates to existing dataset and upload
        enricher.append_and_upload_debates(existing_debates, processed_debates, reset)

        # Generate and log enrichment report
        report = enricher.generate_enrichment_report(new_keyword_labels, new_ml_labels)
        logger.info(f"Enrichment report: {report}")

        logger.info("Hansard enrichment pipeline completed successfully")

    except ValueError as ve:
        logger.error(f"Configuration error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error in Hansard enrichment pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Parse pipeline mode, default is incremental
    parser = argparse.ArgumentParser(description="Run Hansard enrichment pipeline")
    parser.add_argument("--full", action="store_true", help="Run full enrichment instead of incremental")
    parser.add_argument("--test", action="store_true", help="Use a subset of the data for local testing")
    parser.add_argument("--reset", action="store_true", help="Reset labelstores and run full enrichment")
    args = parser.parse_args()

    run_hansard_enrichment(not args.full, args.test, args.reset)
