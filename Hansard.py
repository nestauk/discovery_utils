#!/usr/bin/env python

# download the data from TheyWorkForYou by using:
# rsync -az --exclude='*s19*'  --progress --exclude '.svn' --exclude 'tmp/'
# --relative  data.theyworkforyou.com::parldata/scrapedxml/debates/  .

# load libraries
import os
import re

import pandas as pd

from defusedxml import ElementTree as etree


# --------------------------------------------------------------------------------------------------------------------
# Define functions:
def FilePaths(directory: str) -> list:
    """
    Create a list of all the debate files from where you have stored them.

    Args:
        arg1 (str): The filepath of where the debates are stored.

    Returns:
        list: list of all the debatefiles.
    """

    filenamelist = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            filenamelist.append(os.path.join(dirpath, f))
    return filenamelist


def select_debates_per_year(filelist: list, current_year: str, path: str) -> list:
    """
    Retrieve all the debates from one year

    Args:
        arg1 (list): the list that has all the debatefiles stored
        arg2 (str): current_year used to only select the files that belong to one year

    Returns:
        list: list of all the debatefiles.
    """
    year_list = []
    for i in filelist:
        string_to_match = path + str(current_year)
        match = re.search(string_to_match, i)
        if match:
            year_list.append(i)
    return year_list


def extract_text(elem: object) -> str:
    """
    Extract the text from the speeches

    Args:
        arg1 (object): lmxl object

    Returns:
        text that belong to lmxl object
    """
    speech_text = ""
    for t in elem:
        # Process text node of t if it exists and t has no children
        if len(list(t)) == 0 and t.text:
            speech_text += t.text
        else:
            # Iterate over child elements of t
            for p in t:
                # Recursively process child elements of p
                speech_text += extract_text([p])
                # Process text node of p if it exists
                if p.text:
                    speech_text += p.text
                # Process tail content of p if it exists
                if p.tail:
                    speech_text += p.tail
                else:
                    speech_text += " "
    return speech_text


def clean_text(text: str) -> str:
    """
    Clean the text

    Args:
        arg1 (str): the speech

    Returns:
        the cleaned version of the speech
    """
    cleaned_text = text.strip()
    cleaned_text = cleaned_text.replace("\t", "")
    lines = cleaned_text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    connect_speech = " ".join(non_empty_lines)
    return connect_speech


def create_dataframe(data: list) -> list:
    """
    Create dataframes for each year

    Args:
        arg1 (list): list that stores all the debates per year

    Returns:
        list of dataframes. Each dataframes has the information of one year
    """
    list_of_df = []
    for item in data:
        df = pd.DataFrame(item)
        list_of_df.append(df)
    return list_of_df


def get_speeches(temp_list: list, current_year: str) -> list:
    """
    Create dataframes for each year

    Args:
        arg1 (list): list that stores the debates of only one year
        arg2 (str): passes the year that is being processed

    Returns:
        list called data, which stores the dictionaries that contain the information of each speech
    """

    data = []
    for item in temp_list:
        # print(item)
        parser = etree.XMLParser()
        tree = etree.parse(item, parser)
        # nodes_latest = tree.findall("//publicwhip")
        # print(type(nodes_latest))
        # print(len(nodes_latest))
        # latest_values = [node.get("latest", "NA") for node in nodes_latest]
        # print(type(latest_values))

        # condition1 = latest_values[0] == "yes"
        condition2 = bool(tree.findall(".//minor-heading"))

        if condition2:
            root = tree.getroot()
            context = root.findall("./")

            debate_year = current_year
            date_current = item[-15:-5]

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
                    # sp_nospeaker = elem.get("nospeaker", "NA")
                    # sp_colnum = elem.get("colnum", "NA")
                    # sp_time = elem.get("time", "NA")
                    # sp_url = elem.get("url", "NA")
                    speakername = elem.get("speakername", "NA")
                    speaker_id = elem.get("speakerid", "NA")
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
                            "speech_type": speech_type,
                            "year": debate_year,
                            "date": date_current,
                        }
                    )
    return data


# getthedata
data_path = "/Users/rinskejongma/tryRcode/scrapedxml/debates"  # this is the path where the data is stored
list_of_datafiles = sorted(FilePaths(data_path))

debates_per_year = []  # list that stores 25 each item in the lists has all the debates for one year.
for year in range(2000, 2001):
    # print("processing:", year)

    # Get all the debates that belong to one year
    one_year_list = select_debates_per_year(list_of_datafiles, str(year), "/debates/debates")

    # Get the speeches
    debates = get_speeches(one_year_list, str(year))
    debates_per_year.append(debates)

# Store the data into dataframes
all_debates = create_dataframe(debates_per_year)
df_ = all_debates[0]
