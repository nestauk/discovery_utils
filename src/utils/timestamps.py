import re

def _extract_timestamp(directory: str, format: str = r'\d{8}_\d{6}') -> str:
    """Use regular expression to extract the timestamp from a directory."""
    match = re.search(format, directory)
    if match:
        timestamp_str = match.group()
        return timestamp_str
    
def _timestamp_list(directories: list) -> list:
    """Returns a list of timestamps from a list of directories"""
    # Get a list of timestamps
    timestamps = []
    for directory in directories:
        timestamps.append(_extract_timestamp(directory))
    
    # Sort the timestamps in descending order
    return sorted(timestamps, reverse=True)
    
def _directory(directories: list, timestamp: str) -> str:
    """Finds a directory with a specific timestamp"""
    
    # Return the most recent directory
    for directory in directories:
        if timestamp in directory:
            return directory