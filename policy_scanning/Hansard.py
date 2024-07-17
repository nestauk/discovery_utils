#!/usr/bin/env python

# download the data from TheyWorkForYou by using:
# rsync -az --exclude='*s19*'  --progress --exclude '.svn' --exclude 'tmp/'
# --relative  data.theyworkforyou.com::parldata/scrapedxml/debates/  .

import json

# load libraries
import os
import re
import sys

from typing import Dict
from typing import List

import pandas as pd

from lxml import etree  # nosec B3410

from discovery_utils import PROJECT_DIR


# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# global variables
global membership_dict
global person_id_dict

# --------------------------------------------------------------------------------------------------------------------
# Define functions:


def FilePaths(directory: str) -> List[str]:
    """
    Create a list that stores the path files to the debates

    Args:
        directory (str): The filepath that leads you to where you have stored the debate transcripts.


    Returns:
        list: list of filepaths to the debates.
    """

    filenamelist = []
    for path, _, filenames in os.walk(directory):
        for file in filenames:
            filenamelist.append(os.path.join(path, file))
    return filenamelist


def select_debates_per_year(filelist: List[str], current_year: int) -> List[str]:
    """
    From the list that stores the path files to the debates, select the ones that occurred in the year you are processing.

    Args:
        filelist (list): the list that has all the debatefiles stored
        current_year (int): current_year is the year of which you want to retrieve the debates

    Returns:
        year_list list[str]: list of the debate files for the current_year.
    """
    year_list = []
    for i in filelist:
        match = re.search(str(current_year), i)
        if match:
            year_list.append(i)
    return year_list


def extract_text(elem: object) -> str:
    """
    Retrieve the text of the speeches from the lxml boject.

    Args:
        elem (object): xml element with elem.tag 'speech'

    Returns:
      speech_text (str): The text of the speech.

    """
    speech_text = ""
    # Replace <br> tags with newlines
    for br in elem.xpath(".//br"):
        br.tail = " " + (br.tail if br.tail else "")
    # Extract the full text content
    speech_text = "".join(elem.xpath(".//text()"))
    return speech_text


def clean_text(text: str) -> str:
    """
    Clean the speeches by removing white spaces and white lines.

    Args:
        text (str): the text of the speech

    Returns:
        connect_speech (str): the cleaned version of the speech
    """
    cleaned_text = text.strip()  # removes spaces, tabs, newlines
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = cleaned_text.replace("\t", "")
    lines = cleaned_text.splitlines()  # split the text into seperate lines
    non_empty_lines = [line for line in lines if line.strip()]  # remove all the blank lines in the speech
    connect_speech = " ".join(non_empty_lines)  # concatenate all the non-blank lines from the speech.
    return connect_speech


def find_party(member_id, person_id, current_date):
    party = None
    debate_date = pd.to_datetime(current_date, format="%Y-%m-%d").date()

    if member_id != "NA":
        party = membership_dict.get(member_id)
        return party
    elif person_id != "NA":
        list_possibilities = person_id_dict.get(person_id, [])
        for possibility in list_possibilities:
            if possibility["start_date"] <= debate_date <= possibility["end_date"]:
                party = possibility["on_behalf_of_id"]
                return party
    return party


def clean_meta_data(meta_data: List[Dict]) -> List[Dict]:
    only_2000 = []
    start_date = None

    for item in meta_data:
        # Check if 'start_date' and 'end_date' are present
        if "start_date" in item:
            start_date = str(item["start_date"])
            if len(start_date) > 4:
                if "end_date" in item:
                    if item["end_date"][:-8] == "20":
                        if "on_behalf_of_id" in item:
                            only_2000.append(item)

    filtered_meta_data = only_2000
    return filtered_meta_data


# Create a function that changes the list into dictionaries for faster acces
def create_dict(data, old_key):
    dict = {}
    for item in data:
        new_key = item[old_key]
        if new_key in dict:
            dict[new_key].append(item)
        else:
            dict[new_key] = [item]

    for key, value in dict.items():  # change this because you are not using key
        for item in value:
            item["end_date"] = pd.to_datetime(item["end_date"], format="%Y-%m-%d").date()
            item["start_date"] = pd.to_datetime(item["start_date"], format="%Y-%m-%d").date()

    return dict


# Retrieve meta-data from MPs to assign each speaker to their party
def open_file(file_path: str) -> list[dict]:
    """
    Open the people.json file from Twfy to retrieve meta-data for the members of parliament

    Args:
        file_path(str): The path to the people.json file

    Returns:
        memberships (list): A list of dictionaries. Each dictionary contains the information related to one MP.
    """

    global memberships  # make global so that we only have to open the file once
    with open(file_path, "r") as file:
        data = json.load(file)
        memberships = data["memberships"]
    return memberships


def get_speeches(temp_list: list, current_year: int) -> List[dict]:
    """
    Parse the debatefile and retrieve the following information form the file:

        - Major_heading: Major debate heading
        - Minor_heading:  sub title of a debate
        - Speeches: The text of the speech
        - speakername = The name of the MP who gave the speech
        - speaker_id = The member_id given to the MP by Twfy (mysociety)
        - party_speaker = The party the MP who gave the speech belongs to
        - person_id = The person_id given to the MP by Twfy (mysociety)
        - speech_type = The type of speech given.


    Args:
        temp_list (list): list that stores the debates to analyse.
        current_year (int): passes the year that is being processed

    Returns:
        data (list): stores the dictionaries that contain the information of every speech.
                    Every dictionary contains the information of one speech.
    """

    data = []
    for item in temp_list:
        parser = etree.XMLParser(dtd_validation=False)
        tree = etree.parse(item, parser)  # nosec B320
        condition = bool(tree.xpath("//minor-heading"))

        if condition:
            root = tree.getroot()
            context = root.findall("./")

            debate_year = str(current_year)
            if len(item) == 98:
                date_current = item[-15:-5]
            else:
                date_current = item[-14:-4]

            current_major_heading = None
            current_minor_heading = None

            # Iterate over the elements
            for elem in context:
                if elem.tag == "major-heading":
                    store_text = clean_text(elem.text)
                    # Update major heading and reset minor heading if it's a new major heading
                    if store_text != current_major_heading:
                        current_major_heading = store_text
                        current_minor_heading = None
                elif elem.tag == "minor-heading":
                    current_minor_heading = clean_text(elem.text)
                elif elem.tag == "speech":
                    # get attributes
                    sp_id = elem.get("id")
                    speakername = elem.get("speakername", "NA")
                    speaker_id = elem.get("speakerid", "NA")
                    person_id = elem.get("person_id", "NA")
                    party_speaker = find_party(speaker_id, person_id, date_current)
                    person_id = elem.get("person_id", "NA")
                    speech_type = elem.get("type", "NA")

                    # get actual speech
                    speechtext = extract_text(elem)
                    clean_speech = clean_text(speechtext)
                    data.append(
                        {
                            "major_heading": current_major_heading,
                            "minor_heading": current_minor_heading,
                            "speech": clean_speech,
                            "speech_id": sp_id,
                            "speakername": speakername,
                            "speaker_id": speaker_id,
                            "person_id": person_id,
                            "party_speaker": party_speaker,
                            "speech_type": speech_type,
                            "year": debate_year,
                            "date": date_current,
                        }
                    )

    return data


# getthedata
if __name__ == "__main__":
    # Retrieve meta-data from MPs to assign each speaker to their party.
    path = PROJECT_DIR / "policy_scanning/data/parlparse/members/people.json"
    meta_data = open_file(path)

    # clean meta_data for easier access
    filtered_meta_data = clean_meta_data(meta_data)

    # Create dictionaries for easy search access
    membership_dict = {item["id"]: item["on_behalf_of_id"] for item in filtered_meta_data}
    person_id_dict = create_dict(filtered_meta_data, "person_id")

    # Select the paths to the debate files
    data_path = PROJECT_DIR / "policy_scanning/data/scrapedxml/debates"
    list_of_datafiles = sorted(FilePaths(data_path))

    list_of_df_debates = []  # Every dataframe in this list represents the debates held in one year.

    for year in range(2005, 2025):
        print("processing:", year)  # noqa:<T001>

        # Get all the debates that belong to one year
        one_year_list = select_debates_per_year(list_of_datafiles, year)

        # Get the speeches
        debates = get_speeches(one_year_list, year)
        df_one_year = pd.DataFrame(debates)
        list_of_df_debates.append(df_one_year)

    # Concatenate all DataFrames in the list
    combined_df = pd.concat(list_of_df_debates, ignore_index=True)

    folder_path = PROJECT_DIR / "policy_scanning/data"
    os.makedirs(folder_path, exist_ok=True)
    file_parquet = os.path.join(folder_path, "HansardDebates.parquet")
    # parquet_file
    combined_df.to_parquet(file_parquet, engine="pyarrow")
