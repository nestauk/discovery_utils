import os
import re

from datetime import datetime

import boto3
import pandas as pd

from dotenv import load_dotenv
from slack_sdk.webhook import WebhookClient

from discovery_utils.getters.crunchbase import CrunchbaseGetter
from discovery_utils.getters.gtr import GtrGetter
from discovery_utils.synthesis.investment_signals.investment_signals import create_investment_signals_message
from discovery_utils.synthesis.investment_signals.investment_signals import create_research_grants_message


load_dotenv()
SLACK_URL = os.environ["SLACK_WEBHOOK_URL_TESTING"]


def deduplicate(lst):
    seen = set()
    deduped_list = []
    for item in lst:
        if item not in seen:
            deduped_list.append(item)
            seen.add(item)
    return deduped_list


def format_list(lst):
    """Format each list into a string with ", " and "&" as separators"""
    if len(lst) == 1:
        return lst[0]
    elif len(lst) == 2:
        return f"{lst[0]} & {lst[1]}"
    else:
        return f"{', '.join(lst[:-1])} & {lst[-1]}"


def load_data():
    gtr = GtrGetter()
    crunchbase = CrunchbaseGetter()

    gtr_df = gtr.latest_enriched_grants

    crunchbase_grants = crunchbase.latest_grants
    funding_rounds = crunchbase.latest_funding_rounds
    orgs = crunchbase.latest_startups
    smart_money = crunchbase.latest_smart_money_investors

    return gtr_df, crunchbase_grants, funding_rounds, orgs, smart_money


def format_gtr_data(gtr_df):
    gtr_df_filtered = gtr_df[
        gtr_df["mission_labels"].notna()
        & (gtr_df["mission_labels"] != "X")
        & (gtr_df["status"] == "Active")
        & (gtr_df["amount"] > 0)
    ]

    gtr_df_filtered = gtr_df_filtered.assign(
        name_deduped=gtr_df_filtered["name"].apply(deduplicate),
        name_string=lambda df: df["name_deduped"].apply(format_list),
        country_alpha2="GB",
        country_code="GBR",
        source="GtR",
        leadFunder=gtr_df_filtered["leadFunder"].fillna("Unknown"),
    )

    return gtr_df_filtered


def concat_grants(gtr_df_filtered, crunchbase_grants):
    crunchbase_grants["source"] = "crunchbase"
    crunchbase_grants["leadFunder"] = crunchbase_grants["investor_name"]

    cb_grants = crunchbase_grants[
        [
            "country_alpha2",
            "topic_labels",
            "mission_labels",
            "source",
            "short_description",
            "org_name",
            "investor_name",
            "raised_amount_gbp",
            "announced_on",
            "country_code",
            "cb_url",
            "leadFunder",
        ]
    ].rename(
        columns={
            "announced_on": "announced_on_date",
        }
    )

    gtr_grants = gtr_df_filtered[
        [
            "country_alpha2",
            "country_code",
            "amount",
            "leadFunder",
            "name_string",
            "title",
            "start",
            "end",
            "abstractText",
            "mission_labels",
            "topic_labels",
            "url",
            "source",
        ]
    ].rename(
        columns={
            "name_string": "org_name",
            "amount": "raised_amount_gbp",
            # "abstractText": "short_description",
        }
    )

    grants = pd.concat([cb_grants, gtr_grants], ignore_index=True)

    return grants


def send_to_slack(slack_webhook=WebhookClient(SLACK_URL)):

    gtr_df, crunchbase_grants, funding_rounds, orgs, smart_money = load_data()

    gtr_df_filtered = format_gtr_data(gtr_df)

    grants = concat_grants(gtr_df_filtered, crunchbase_grants)

    uk_alpha3 = ["GBR"]

    eu_countries_alpha3 = [
        "AUT",
        "BEL",
        "BGR",
        "HRV",
        "CYP",
        "CZE",
        "DNK",
        "EST",
        "FIN",
        "FRA",
        "DEU",
        "GRC",
        "HUN",
        "IRL",
        "ITA",
        "LVA",
        "LTU",
        "LUX",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "ESP",
        "SWE",
    ]

    # Send UK messages
    # 1. UK Investment signals
    uk_investment_block = create_investment_signals_message(
        region="uk",
        org_diffs=orgs.query("country_code in @uk_alpha3"),
        funding_rounds_diffs=funding_rounds.query("country_code in @uk_alpha3"),
        grants=grants.query("country_code in @uk_alpha3"),
        smart_money_investors=smart_money,
        limit=None,
    )
    slack_webhook.send(
        blocks=uk_investment_block,
        unfurl_links=False,
        unfurl_media=False,
    )

    # Send EU investment signals message
    eu_investment_block = create_investment_signals_message(
        region="eu",
        org_diffs=orgs.query("country_code in @eu_countries_alpha3"),
        funding_rounds_diffs=funding_rounds.query("country_code in @eu_countries_alpha3"),
        grants=grants.query("country_code in @eu_countries_alpha3"),
        smart_money_investors=smart_money,
        limit=None,
    )
    slack_webhook.send(
        blocks=eu_investment_block,
        unfurl_links=False,
        unfurl_media=False,
    )

    # 2. UK Research grants
    uk_research_block = create_research_grants_message(
        region="uk",
        grants=grants.query("country_code in @uk_alpha3"),
        limit=None,
    )
    slack_webhook.send(
        blocks=uk_research_block,
        unfurl_links=False,
        unfurl_media=False,
    )


import json


def _preview_slack_message(message_payload):
    """
    Preview how a Slack message will look by printing a formatted version
    of the message content.
    """
    print("\n=== SLACK MESSAGE PREVIEW ===\n")

    if isinstance(message_payload, str):
        print(message_payload)
        return

    if "text" in message_payload:
        print(message_payload["text"])

    if "blocks" in message_payload:
        for block in message_payload["blocks"]:
            if block["type"] == "section":
                if "text" in block:
                    text = block["text"]["text"] if isinstance(block["text"], dict) else block["text"]
                    print(text)
            elif block["type"] == "divider":
                print("-" * 40)

    print("\n=== JSON PAYLOAD ===")
    print(json.dumps(message_payload, indent=2))

    with open("slack_message.json", "w") as f:
        json.dump(message_payload, f, indent=2)


if __name__ == "__main__":
    send_to_slack(slack_webhook=WebhookClient(SLACK_URL))
