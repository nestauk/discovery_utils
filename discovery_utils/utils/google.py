"""
This module contains functions for accessing Google Sheets

Usage:
import discovery_child_development.utils.google_utils as google_utils

# access data from Google Sheets
data = google_utils.access_google_sheet(<sheet_id>, <sheet_name>)
"""

import os

from os import environ
from os import path
from pathlib import PosixPath

import dotenv
import gspread
import gspread_formatting as gsf

from df2gspread import df2gspread as d2g
from df2gspread import gspread2df as g2d
from googleapiclient.discovery import Resource
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from gspread.exceptions import WorksheetNotFound
from oauth2client.service_account import ServiceAccountCredentials
from pandas import DataFrame

from discovery_utils import PROJECT_DIR
from discovery_utils import S3_BUCKET
from discovery_utils import logging
from discovery_utils.utils.s3 import s3_client


dotenv.load_dotenv()


def find_credentials(credentials_env_var: str) -> PosixPath:
    """Find credentials file

    For accessing some Google resources, we need credentials stored in a JSON file in `.credentials/`.
    This function takes the name of an environment variable as input and checks whether the corresponding
    credentials file exists. If not, it downloads the file from S3.

    Args:
        credentials_env_var (str): Name of the env var eg "GOOGLE_SHEETS_CREDENTIALS".
        Your .env file should have paths to Google credentials files stored like
        "GOOGLE_SHEETS_CREDENTIALS=<path-to-credentials-file>".

    Raises:
        EnvironmentError: If this env var is not recorded in `.env`
        Exception: If the function can neither find the credentials file nor download it from S3

    Returns:
        PosixPath: Path to the credentials file
    """
    # Check if the environment variable is set
    if credentials_env_var not in environ:
        raise EnvironmentError("The environment variable is not set.")

    credentials_json = PROJECT_DIR / environ.get(credentials_env_var)

    if not path.isfile(credentials_json):
        logging.info("Credentials not found. Downloading from S3...")
        credentials_json.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3_client().download_file(
                S3_BUCKET,
                f"credentials/{credentials_json.name}",
                str(credentials_json),
            )
        except Exception as e:
            raise Exception(f"Error downloading credentials from S3: {e}")

    return credentials_json


# Google Drive


def get_drive_service() -> Resource:
    """Initialise Google Drive API service."""
    credentials = load_gsheet_credentials()
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    return service


def upload_image_to_drive(service: Resource, file_path: str) -> tuple[str, str]:
    """Upload a file to Google Drive and makes it public."""

    file_metadata = {"name": os.path.basename(file_path), "mimeType": "image/png"}
    media = MediaFileUpload(file_path, mimetype="image/png")

    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()

    # Make the file publicly accessible
    service.permissions().create(fileId=uploaded_file["id"], body={"role": "reader", "type": "anyone"}).execute()

    # Get the public URL
    file_id = uploaded_file["id"]
    image_url = f"https://drive.google.com/uc?id={file_id}"

    logging.info(f"Uploaded image available at: {image_url}")

    return file_id, image_url


def delete_file_from_drive(service: Resource, file_id: str) -> None:
    """Delete a file from Google Drive using its file ID."""
    try:
        service.files().delete(fileId=file_id).execute()
        logging.info(f"Image with file ID '{file_id}' has been deleted from Google Drive.")
    except Exception as e:
        logging.error(f"Error deleting file: {e}")


# Google Sheets


def load_gsheet_credentials() -> ServiceAccountCredentials:
    """Get credentials for accessing Google Sheets"""
    google_credentials_json = find_credentials("GOOGLE_SHEETS_CREDENTIALS")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    return ServiceAccountCredentials.from_json_keyfile_name(google_credentials_json, scope)


def access_google_sheet(sheet_id: str, sheet_name: str) -> DataFrame:
    """
    Access a specified Google Sheet and return its contents as a pandas DataFrame.

    This function authenticates using service account credentials, defines the scope
    for the Google Sheets API, and downloads the sheet contents. The sheet is accessed
    by its unique identifier and a specific sheet name within the spreadsheet.

    Args:
        sheet_id (str): The unique identifier for the Google Sheets file.
        sheet_name (str): The name of the individual sheet within the Google Sheets file.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the specified Google Sheet.

    Raises:
        GoogleAuthError: If authentication with Google Sheets API fails.
        DownloadError: If there is an issue downloading the sheet contents.

    Notes:
    - The GOOGLE_SHEETS_CREDENTIALS environment variable must be set with the path to
      the credentials JSON file ie `.credentials/xxxxx.json`.
    - The service account must have the necessary permissions to access the Google Sheet.
    - The function assumes the first row and column of the sheet contain the header and
      index names, respectively.
    """
    # Load the credentials for use with Google Sheets
    credentials = load_gsheet_credentials()
    # Load the data into a pandas DataFrame
    data = g2d.download(sheet_id, sheet_name, credentials=credentials, col_names=True, row_names=True)
    return data


def connect_to_gsheet(sheet_id: str) -> gspread.Spreadsheet:
    """
    Connect to an existing Google Sheet by its unique identifier.

    Args:
        sheet_id (str): The unique identifier for the Google Sheet.

    Returns:
        The Google Sheet object.
    """
    credentials = load_gsheet_credentials()
    client = gspread.authorize(credentials)
    # Open the existing Google Sheet by ID
    spreadsheet = client.open_by_key(sheet_id)
    logging.info(f"Connected to Google Sheet: {spreadsheet.title}")
    return spreadsheet


def upload_data_to_gsheet(sheet_id: str, dataframes: dict) -> None:
    """
    Upload multiple DataFrames to an existing Google Sheet as separate sheets.

    Args:
        sheet_id (str): The Google Sheet ID where data should be uploaded.
        dataframes (dict): A dictionary where keys are sheet names and values are pandas DataFrames.
    """
    credentials = load_gsheet_credentials()
    # Upload each DataFrame to the corresponding sheet
    for sheet_name, df in dataframes.items():
        try:
            spreadsheet = connect_to_gsheet(sheet_id)
            worksheet = spreadsheet.worksheet(sheet_name)
        except WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows="100", cols="20")
        # Hack to avoid IncorrectCellLabel
        worksheet.update("A1", [["1"]])
        # Upload data
        logging.info(f"Uploading DataFrame to sheet: {sheet_name}")
        d2g.upload(df, sheet_id, sheet_name, credentials=credentials, row_names=True)
        # Delete the first column (index)
        worksheet.delete_columns(1)
    logging.info("Upload completed successfully.")


def format_gsheet(sheet_id: str, sheet_name: str, freeze_cols: int = 0) -> None:
    """Apply formatting to a Google Sheet

    Specifically: freeze the header row, apply background colour to the header row,
    and add filters.

    Args:
        sheet_id (str): The unique identifier for the Google Sheets file.
        sheet_name (str): The name of the individual sheet within the Google Sheets file.
        freeze_cols (int): The number of columns to freeze. Default is 0.
    """
    spreadsheet = connect_to_gsheet(sheet_id)
    worksheet = spreadsheet.worksheet(sheet_name)

    # Freeze the header row
    worksheet.freeze(rows=1, cols=freeze_cols)

    # Apply background colour to header row
    header_format = gsf.CellFormat(
        backgroundColor=gsf.Color(red=1, green=1, blue=0.33),
        textFormat=gsf.TextFormat(bold=True, fontSize=10),
        horizontalAlignment="LEFT",
    )
    gsf.format_cell_range(worksheet, "1:1", header_format)

    # Add filters
    worksheet.set_basic_filter("A1:AZ")
