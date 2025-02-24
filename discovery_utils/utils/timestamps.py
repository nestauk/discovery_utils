import re

from datetime import datetime
from datetime import timedelta


def get_weekly_start_date(end_date: str, weeks: int = 1) -> str:
    """Get the start date for a weekly period ending at the specified end_date

    Args:
        end_date: The end date of the period, in the format "YYYY-MM-DD"
        weeks: The number of weeks to go back

    Returns:
        The start date of the period, in the format "YYYY-MM-DD"
    """
    data_end_date = datetime.strptime(end_date, "%Y-%m-%d")
    weeks_ago = data_end_date - timedelta(weeks=weeks)
    return weeks_ago.strftime("%Y-%m-%d")


def _extract_timestamp(directory: str, format: str = r"\d{8}_\d{6}") -> str:
    """Use regular expression to extract the timestamp from a directory."""
    match = re.search(format, directory)
    if match:
        timestamp_str = match.group()
        return timestamp_str


def _timestamp_list(directories: list) -> list:
    """Return a list of timestamps from a list of directories"""
    # Get a list of timestamps
    timestamps = []
    for directory in directories:
        timestamps.append(_extract_timestamp(directory))

    # Sort the timestamps in descending order
    return sorted(timestamps, reverse=True)


def _directory(directories: list, timestamp: str) -> str:
    """Find a directory with a specific timestamp"""

    # Return the most recent directory
    for directory in directories:
        if timestamp in directory:
            return directory
