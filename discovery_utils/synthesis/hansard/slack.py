import re

import os


from typing import List, Dict

def mission_header(mission: str) -> Dict:
    """Construct mission header block"""
    if mission == "ASF":
        mission_header = ":potted_plant: *A Sustainable Future*"
    elif mission == "AFS":
        mission_header = ":hatched_chick: *A Fairer Start*"
    elif mission == "AHL":
        mission_header = ":mending_heart: *A Healthier Life*"
    else:
        raise ValueError(f"Invalid mission: {mission}")

    return {"type": "section", "text": {"type": "mrkdwn", "text": mission_header}}

def message_header(message_date:str, data_start_date:str, data_end_date:str) -> List[Dict]:
    """Construct message header block
    
    Args:
        message_date (str): Date when posting the message, in format DD-MM-YYYY
        data_start_date (str): Start date of the data, in format DD-MM-YYYY
        data_end_date (str): End date of the data, in format DD-MM-YYYY

    Returns:
        List[Dict]: List of blocks including a header and a context block, where 
            context block indicates data sources and date range
    """
    header = {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"Policy update {message_date}",
        }
    }
    context = {
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"House of Commons debates ({data_start_date} - {data_end_date})",
            }
        ]
    }
    return [header, context]
    

def divider() -> Dict:
    """Construct a divider block"""
    return {"type": "divider"}

def _bullet_point_string(points: List[str]) -> str:
    """Construct a string of bullet points from a list of strings"""
    return "\n".join([f"• {point}" for point in points])     

def debate_summary(debate: Dict) -> Dict:
    """Construct a block for a single debate summary
    
    Args:
        debate (Dict): Dictionary with keys "title", "summary", "positives", "negatives", and "next_steps".
            For example: {
                "title": "Title of the debate",
                "purpose": "Summary of the debate",
                "positives": ["Positive point 1", "Positive point 2"],
                "negatives": ["Negative point 1", "Negative point 2"],
                "next_steps": ["Next step 1", "Next step 2"],
            }
    """
    summary = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"<{debate['url']}|*{debate['title']}*> ({debate['date']})\n{debate['purpose']}"
        }
    }
    positives = {
        "type": "section",
        "text":{
             "type": "mrkdwn",
             "text": f"*Positives*\n{_bullet_point_string(debate['positives'])}"
        }
    }
    negatives = {
        "type": "section",
        "text":{
             "type": "mrkdwn",
             "text": f"*Negatives*\n{_bullet_point_string(debate['negatives'])}"
        }
    }
    next_steps = {
        "type": "section",
        "text":{
             "type": "mrkdwn",
             "text": f"*Next Steps*\n{_bullet_point_string(debate['next_steps'])}"
        }
    }
    return [summary, positives, negatives, next_steps, divider()]

def quote_block(quote: Dict) -> Dict:
    """Construct a block with a quote
    
    Args:
        quote (Dict): Dictionary with keys "name", "party", "category", "debate", and "text".
    """
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*{quote['name']}* ({quote['party']}) mentioned *{quote['category']}* in *{quote['debate']}*\n\n> {quote['text']}"
        }
    }
