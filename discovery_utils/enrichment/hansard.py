import logging

from io import BytesIO
from io import StringIO

import pandas as pd

import horizon_scout.inference_bert as hs

from discovery_utils import PROJECT_DIR
from discovery_utils.utils.keywords import enrich_topic_labels


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import json
import logging
import os
import re

from datetime import date
from datetime import datetime
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Union

import pandas as pd

from lxml import etree

from discovery_utils import PROJECT_DIR
from discovery_utils.utils import google


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HansardProcessor:
    def __init__(self):
        self.membership_dict: Dict[str, str] = {}
        self.person_id_dict: Dict[str, List[Dict[str, Any]]] = {}
        self.load_member_data()

    def load_member_data(self) -> None:
        """Load and process member data from the people.json file."""
        logger.info("Loading member data...")
        path = PROJECT_DIR / "tmp/people.json"
        try:
            with open(path, "r") as file:
                data = json.load(file)
                memberships = data["memberships"]

            filtered_meta_data = self._clean_meta_data(memberships)
            self.membership_dict, self.person_id_dict = self._create_member_dicts(filtered_meta_data)
            logger.info("Member data loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading member data: {e}")
            raise

    @staticmethod
    def _clean_meta_data(meta_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean the people.json file metadata.
        Include members that are still sitting (no end date) and
        members whose term ended after the year 2000.
        """
        logger.info("Cleaning metadata...")
        cleaned_data = []
        current_year = datetime.now().year

        for item in meta_data:
            try:
                if all(["start_date" in item, "on_behalf_of_id" in item]):
                    start_date = int(str(item["start_date"])[:4])  # Get year from start_date

                    if "end_date" in item:
                        end_date = int(str(item["end_date"])[:4])  # Get year from end_date
                        if end_date >= 2000:
                            cleaned_data.append(item)
                    else:
                        # No end_date means the member is still sitting
                        cleaned_data.append(item)

                    if start_date > current_year:
                        logger.warning(f"Found start_date in the future: {item}")

            except Exception as e:
                logger.warning(f"Error processing item in meta_data: {e}")
                logger.debug(f"Problematic item: {item}")

        logger.info(f"Cleaned {len(meta_data) - len(cleaned_data)} invalid entries from metadata.")
        logger.info(f"Retained {len(cleaned_data)} valid entries.")
        return cleaned_data

    @staticmethod
    def _create_member_dicts(data: List[Dict[str, Any]]) -> tuple[Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
        """Create dictionaries for faster access to member party information."""
        logger.info("Creating member dictionaries...")
        member_dict = {item["id"]: item["on_behalf_of_id"] for item in data}

        person_dict: Dict[str, List[Dict[str, Any]]] = {}
        for item in data:
            person_id = item["person_id"]
            if person_id in person_dict:
                person_dict[person_id].append(item)
            else:
                person_dict[person_id] = [item]

        current_date = date.today()
        for value in person_dict.values():
            for item in value:
                item["start_date"] = pd.to_datetime(item["start_date"]).date()
                if "end_date" in item and item["end_date"]:
                    item["end_date"] = pd.to_datetime(item["end_date"]).date()
                else:
                    # If end_date is missing or None, set it to current date
                    item["end_date"] = current_date
                    logger.debug(f"Missing end_date for item: {item}. Set to current date.")

        logger.info(f"Created dictionaries with {len(member_dict)} members and {len(person_dict)} persons.")
        return member_dict, person_dict

    def find_party(self, member_id: str, person_id: str, debate_date: str) -> Union[str, None]:
        """Select the party of the MP who gave the speech."""
        debate_date_dt = pd.to_datetime(debate_date).date()

        if member_id != "NA":
            return self.membership_dict.get(member_id)
        elif person_id != "NA":
            for term in self.person_id_dict.get(person_id, []):
                if term["start_date"] <= debate_date_dt <= term["end_date"]:
                    return term["on_behalf_of_id"]
        # logger.warning(f"Could not find party for member_id: {member_id}, person_id: {person_id}, date: {debate_date}")
        return None

    @staticmethod
    def get_debate_files(directory: str) -> List[str]:
        """Get a list of debate file paths."""
        logger.info(f"Scanning for debate files in {directory}")
        file_list = [os.path.join(path, file) for path, _, files in os.walk(directory) for file in files]
        logger.info(f"Found {len(file_list)} debate files.")
        return file_list

    @staticmethod
    def select_debates_per_year(file_list: List[str], year: int) -> List[str]:
        """Select debate files for a specific year."""
        year_debates = [file for file in file_list if str(year) in file]
        logger.info(f"Selected {len(year_debates)} debate files for year {year}")
        return year_debates

    @staticmethod
    def extract_text(elem: etree.Element) -> str:
        """Extract text from an XML element."""
        for br in elem.xpath(".//br"):
            br.tail = " " + (br.tail or "")
        return "".join(elem.xpath(".//text()"))

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text."""
        cleaned = " ".join(line.strip() for line in text.splitlines() if line.strip())
        return " ".join(cleaned.split())

    def process_speech(self, elem: etree.Element, date: str, year: str) -> Dict[str, Any]:
        """Process a single speech element."""
        speech_id = elem.get("id", "NA")
        speaker_name = elem.get("speakername", "NA")
        speaker_id = elem.get("speakerid", "NA")
        person_id = elem.get("person_id", "NA")
        party = self.find_party(speaker_id, person_id, date)

        speech_text = self.extract_text(elem)
        clean_speech = self.clean_text(speech_text)

        return {
            "speech_id": speech_id,
            "speakername": speaker_name,
            "speaker_id": speaker_id,
            "person_id": person_id,
            "party_speaker": party,
            "speech": clean_speech,
            "year": year,
            "date": date,
            "major_heading": None,
            "minor_heading": None,
        }

    def process_debate_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """Process a single debate file and yield speech data."""
        # logger.info(f"Processing debate file: {file_path}")
        parser = etree.XMLParser(dtd_validation=False)
        try:
            tree = etree.parse(file_path, parser)
            root = tree.getroot()
        except Exception as e:
            logger.error(f"Error parsing XML file {file_path}: {e}")
            return

        # Extract date from filename
        filename = os.path.basename(file_path)
        date_match = re.search(r"debates(\d{4}-\d{2}-\d{2})", filename)
        if date_match:
            date_str = date_match.group(1)
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                date = date_obj.strftime("%Y-%m-%d")
                year = date_obj.year
            except ValueError:
                logger.error(f"Invalid date format in filename: {filename}")
                return
        else:
            logger.error(f"Could not extract date from filename: {filename}")
            return

        current_major_heading = None
        current_minor_heading = None

        speech_count = 0
        for elem in root.findall("./"):
            if elem.tag == "major-heading":
                current_major_heading = self.clean_text(elem.text)
                current_minor_heading = None
            elif elem.tag == "minor-heading":
                current_minor_heading = self.clean_text(elem.text)
            elif elem.tag == "speech":
                speech_data = self.process_speech(elem, date, year)
                speech_data["major_heading"] = current_major_heading
                speech_data["minor_heading"] = current_minor_heading
                yield speech_data
                speech_count += 1

        # logger.info(f"Processed {speech_count} speeches from file {file_path}")

    def process_debates(self, year: int) -> pd.DataFrame:
        """Process all debate files for a given year."""
        logger.info(f"Processing debates for year {year}")
        data_path = PROJECT_DIR / "tmp/debates"
        debate_files = self.get_debate_files(data_path)
        year_debates = self.select_debates_per_year(debate_files, year)

        all_speeches = []
        for file in year_debates:
            all_speeches.extend(list(self.process_debate_file(file)))

        df = pd.DataFrame(all_speeches)
        logger.info(f"Processed {len(df)} speeches for year {year}")
        return df


class HansardEnricher:
    def __init__(self, s3_client, bucket_name):
        self.s3 = s3_client
        self.bucket = bucket_name
        logger.info(f"Initialized HansardEnricher with bucket: {bucket_name}")
        hs.s3.BUCKET_NAME_RAW = "discovery-iss"

    def process_new_xml(self, xml_files):
        logger.info(f"Processing {len(xml_files)} new XML files")
        return pd.DataFrame(columns=["id", "text"])

    def append_and_upload_parquet(self, existing_df, new_df, key):
        logger.info(f"Appending and uploading parquet file: {key}")
        try:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            buffer = BytesIO()
            combined_df.to_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue())
            logger.info(f"Successfully appended and uploaded parquet file: {key}")
        except Exception as e:
            logger.error(f"Error appending and uploading parquet file {key}: {str(e)}")
            raise

    def append_and_upload_csv(self, existing_df, new_df, key):
        logger.info(f"Appending and uploading CSV file: {key}")
        try:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            csv_buffer = StringIO()
            combined_df.to_csv(csv_buffer, index=False)
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=csv_buffer.getvalue())
            logger.info(f"Successfully appended and uploaded CSV file: {key}")
        except Exception as e:
            logger.error(f"Error appending and uploading CSV file {key}: {str(e)}")
            raise

    def enrich_data(self, hansard_debates, keyword_labels, ml_labels, new_xml_files):
        logger.info("Starting data enrichment process")
        try:
            new_debates_df = self.process_new_xml(new_xml_files)

            logger.info("Appending and uploading Hansard debates")
            self.append_and_upload_parquet(hansard_debates, new_debates_df, "HansardDebates.parquet")

            logger.info("Running keyword labeling")
            new_keyword_labels = enrich_topic_labels(new_debates_df)
            self.append_and_upload_csv(keyword_labels, new_keyword_labels, "keyword_label_store.csv")

            logger.info("Running ML labeling")
            # Load models and tokenizer
            bert_models = hs.load_models(hs.MODEL_PATHS, hs.MISSIONS, gpu=False)
            tokenizer = hs.AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
            new_ml_labels = hs.bert_add_predictions(
                hansard_debates.to_pandas(), "text", bert_models, tokenizer, hs.MISSIONS, 10
            )
            self.append_and_upload_csv(ml_labels, new_ml_labels, "ml_label_store.csv")

            logger.info("Data enrichment process completed successfully")
        except Exception as e:
            logger.error(f"Error in data enrichment process: {str(e)}")
            raise
