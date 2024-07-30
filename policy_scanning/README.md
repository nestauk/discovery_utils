# Script to collect data from Hansard using the TheyWorkForYou website

You can use Hansard.py to retrieve transcripts from debates held in the UK House Of Commons.
The folder should contain only the xml files downloaded from theyworkforyou, otherwise there might be errors.

STEP 1: navigate to the folder where you wish to store the data:

    for example: cd policy_scanning/data

STEP 2: run the following command in the terminal to download the data:

    rsync -az --exclude='*s19*'  --progress --exclude '.svn' --exclude 'tmp/' --relative  data.theyworkforyou.com::parldata/scrapedxml/debates/  .

    This will download all debates from 2000+

    You can change this command of you only which to download debates from, for example, one year or one day.
    For example: rsync -az --progress --exclude '.svn' --exclude 'tmp/' --relative data.theyworkforyou.com::parldata/scrapedxml/debates/'debates2024-*' .

    Notes:
        rsync compares the files from the source directory with the files that are on the destination directory (your directory).
        That means rsync is desigend to synchronize files between a source and a destination by transferring only the differences between the source and destination directories. That means you can rerun the above code to download new files without redownloading all the data.
        It also means that if you want to redownload a debate because it has been updated, you need to first remove that specific file within your folder.

        Do not forget the parantheses if you only wish the download debates from one specific year or day.

        Running the Rsync code above will create the path scrapedxml/debates.
        Hence the file_path where the transcripts are stored as follows: ./scrapedxml/debates"
        To see more data that can be downloaded from TheyWorkForYou.com you can checkout: https://www.theyworkforyou.com/pwdata/

STEP 3: download the people.json file by cloning the parlparse github into your own data folder. This file is used to retrieve
        the affiliated party from MPs.

    git clone https://github.com/mysociety/parlparse

STEP 4: Then run:
    python path/to/the/file/Hansard.py

    This will create a file in data/HansardDebates.parquet with the following columns:
        - Major_heading: Major debate heading
        - Minor_heading:  sub title of a debate
        - Speeches: The text of the speech
        - speakername = The name of the MP who gave the speech
        - speaker_id = The member_id given to the MP by Twfy (mysociety)
        - person_id = The person_id given to the MP by Twfy (mysociety)
        - party_speaker = The party the MP who gave the speech belongs to
