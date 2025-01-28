from datetime import date
from typing import Dict
from typing import List
from typing import Literal
from typing import TypedDict
from typing import Union

import pandas as pd

from numpy import datetime64


MISSIONS = {
    "ASF": {"emoji": ":potted_plant:", "title": "A Sustainable Future"},
    "AFS": {"emoji": ":hatched_chick:", "title": "A Fairer Start"},
    "AHL": {"emoji": ":mending_heart:", "title": "A Healthier Life"},
    "X": {"emoji": ":rocket:", "title": "Cross-cutting"},
}
REGIONS = {
    "uk": {"title": "United Kingdom", "emoji": ":flag-gb:"},
    "eu": {"title": "Europe", "emoji": ":flag-eu:"},
    "world": {"title": "Rest of the World", "emoji": ":world_map:"},
}
DEFAULT_LIMIT = 1000000
DATE_THRESHOLD = pd.DateOffset(months=3)  # Cutoff for funding rounds
FUNDING_AMOUNT_THRESHOLD = 30000  # Maximum amount for funding rounds in thousands


class SlackHeaderBlock(TypedDict):
    type: Literal["header"]
    text: Dict[str, str]


class RichTextElement(TypedDict):
    type: Literal["text", "link", "emoji"]
    text: str
    url: str | None
    style: Dict[str, bool] | None
    name: str | None


class RichTextSection(TypedDict):
    type: Literal["rich_text_section"]
    elements: List[RichTextElement]


class RichTextListBlock(TypedDict):
    type: Literal["rich_text_list"]
    style: Literal["bullet"]
    elements: List[RichTextSection]


class RichTextBlock(TypedDict):
    type: Literal["rich_text"]
    elements: List[Union[RichTextSection, RichTextListBlock]]


class SectionBlock(TypedDict):
    type: Literal["section"]
    text: Dict[str, str]


class DividerBlock(TypedDict):
    type: Literal["divider"]


def message_title_block(region: Literal["uk", "eu", "world"]) -> SlackHeaderBlock:
    region_info = REGIONS[region]
    return {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"{region_info['emoji']} {region_info['title']} - New investment signals {date.today().strftime('%d-%m-%Y')}",
            "emoji": True,
        },
    }


divider_block = [{"type": "divider"}]


# converts each row of a pandas dataframe to a richtext item
def _to_richtext_item(
    row: pd.Series, section: Literal["startups", "grants", "funding_rounds", "smart_money"]
) -> RichTextSection:
    country_alpha2: str = row["country_alpha2"]
    if section == "startups":
        description: str = row["short_description"]
        topic_labels: str = row["topic_labels"]
        topic_labels: str = topic_labels.replace(",", ", ")
        main_text: str = f": {description} [{topic_labels}]"

        elements = [
            {"type": "link", "text": "<text>", "url": "<url>", "style": {"bold": True}},
            {"type": "text", "text": "<text>"},
        ]

        elements[0]["text"] = row["name"]
        elements[0]["url"] = row["cb_url"]
        elements[1]["text"] = main_text

        if country_alpha2 != "None":  # Upstream conversion changed None to str(None), i.e. "None"
            elements.insert(0, {"type": "text", "text": " "})
            elements.insert(0, {"type": "emoji", "name": f"flag-{country_alpha2}"})

    elif section == "grants":
        topic_labels: str = row["topic_labels"]
        topic_labels: str = topic_labels.replace(",", ", ")
        if row["source"] == "crunchbase":
            description: str = row["short_description"]
            name: str = row["org_name"]
            raised_amount: float = row["raised_amount_gbp"] * 1000  # value is in thousands
            announced_on: datetime64 = row["announced_on_date"].strftime("%b %d, %Y")
            investor: str = row["investor_name"]
            if investor == "Unknown":
                main_text: str = f"{name} [{topic_labels}] was awarded £{raised_amount:0,.2f} on {announced_on} ("
            else:
                main_text: str = (
                    f"{name} [{topic_labels}] was awarded £{raised_amount:0,.2f} by {investor} on {announced_on} ("
                )

            elements = [
                {"type": "text", "text": "<text>"},
                {"type": "link", "text": "link", "url": "<url>", "style": {"bold": True}},
                {"type": "text", "text": f"). {description}"},
            ]

            elements[0]["text"] = main_text
            elements[1]["url"] = row["cb_url"]
        else:
            description: str = row["abstractText"]
            raised_amount: float = row["raised_amount_gbp"]
            start: datetime64 = row["start"]  # .strftime("%b %d, %Y")
            end: datetime64 = row["end"]  # .strftime("%b %d, %Y")
            funder: str = row["leadFunder"]
            org: str = row["org_name"]
            title: str = row["title"]
            main_text: str = f"[{topic_labels}] Researchers at {org} were awarded £{raised_amount:0,.2f} by {funder} for the project '{title}'. The project runs from {start} to {end} ("

            elements = [
                {"type": "text", "text": "<text>"},
                {"type": "link", "text": "link", "url": "<url>", "style": {"bold": True}},
                {"type": "text", "text": f")."},
                # {"type": "text", "text": f"). Abstract:"},
                # {"type": "text", "text": f" {description}", "style": {"italic": True}},
            ]

            elements[0]["text"] = main_text
            elements[1]["url"] = row["url"]

        if country_alpha2 != "None":  # Upstream conversion changed None to str(None), i.e. "None"
            elements.insert(0, {"type": "text", "text": " "})
            elements.insert(0, {"type": "emoji", "name": f"flag-{country_alpha2}"})
    elif section == "funding_rounds":
        description: str = row["short_description"]
        name: str = row["org_name"]
        topic_labels: str = row["topic_labels"]
        topic_labels: str = topic_labels.replace(",", ", ")
        raised_amount: float = row["raised_amount_gbp"] * 1000  # value is in thousands
        announced_on: datetime64 = row["announced_on_date"].strftime("%b %d, %Y")
        main_text: str = f"{name} [{topic_labels}] raised £{raised_amount:0,.2f} on {announced_on} ("

        elements = [
            {"type": "text", "text": "<text>"},
            {"type": "link", "text": "link", "url": "<url>", "style": {"bold": True}},
            {"type": "text", "text": f"). {description}"},
        ]

        elements[0]["text"] = main_text
        elements[1]["url"] = row["cb_url"]

        if country_alpha2 != "None":  # Upstream conversion changed None to str(None), i.e. "None"
            elements.insert(0, {"type": "text", "text": " "})
            elements.insert(0, {"type": "emoji", "name": f"flag-{country_alpha2}"})

    elif section == "smart_money":
        description: str = row["short_description"]
        investor_name: str = row["investor_name"]
        org_name: str = row["org_name"]
        topic_labels: str = row["topic_labels"]
        topic_labels: str = topic_labels.replace(",", ", ")
        announced_on: datetime64 = row["announced_on_date"].strftime("%b %d, %Y")

        elements = [
            {"type": "text", "text": "<text>"},
            {"type": "link", "text": "<text>", "url": "<url>", "style": {"bold": True}},
            {"type": "text", "text": f"). {description}"},
        ]

        elements[1]["text"] = row["investment_type"]
        elements[1]["url"] = row["cb_url"]

        if country_alpha2 == "None":  # Upstream conversion changed None to str(None), i.e. "None"
            main_text: str = f"{investor_name} invested in {org_name} [{topic_labels}] on {announced_on} ("
            elements[0]["text"] = main_text
        else:
            investor_text: str = f"{investor_name} invested in "
            elements[0]["text"] = investor_text
            elements.insert(1, {"type": "text", "text": " "})  # insert space after investor_text, at position 1
            elements.insert(
                1, {"type": "emoji", "name": f"flag-{country_alpha2}"}
            )  # insert emoji after investor_text, at position 1, space is position 2

            org_text: str = f"{org_name} [{topic_labels}] on {announced_on} ("
            elements.insert(
                3, {"type": "text", "text": org_text}
            )  # insert org_text at position 3, after emoji(pos 1) and space(pos 2)
    else:
        raise NotImplementedError

    # TODO add smart money template

    return {"type": "rich_text_section", "elements": elements}


def gen_startups_block(org_data: pd.DataFrame) -> RichTextBlock:

    # filter startups not founded this year, founded_on. or entry created this year, entry is created_at
    this_year = date.today().year
    org_data["founded_on"] = pd.to_datetime(org_data["founded_on"], format="ISO8601", errors="coerce")
    org_data = org_data.query("founded_on.dt.year >= @this_year")

    if startups_item_blocks := [_to_richtext_item(row, section="startups") for _, row in org_data.iterrows()]:
        startups_list_block = {"type": "rich_text_list", "style": "bullet", "elements": startups_item_blocks}

        startups_block = {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": [{"type": "text", "text": "New startups", "style": {"bold": True}}],
                },
                startups_list_block,
            ],
        }
    else:
        startups_block = {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": [{"type": "text", "text": "No new startups", "style": {"bold": True}}],
                }
            ],
        }

    return startups_block


def gen_funding_rounds_block(
    funding_rounds_data: pd.DataFrame,
) -> RichTextBlock:

    # filter funding rounds data older than 3 months, using announced_on_date
    # also filter out deals > £30million, raised_amount_gbp is in 1000s
    date_cutoff = pd.Timestamp(date.today()) - DATE_THRESHOLD
    funding_rounds_data = funding_rounds_data.query("announced_on_date >= @date_cutoff").query(
        f"raised_amount_gbp <= {FUNDING_AMOUNT_THRESHOLD}"
    )

    if funding_rounds_item_blocks := [
        _to_richtext_item(row, section="funding_rounds") for _, row in funding_rounds_data.iterrows()
    ]:
        funding_rounds_list_block = {
            "type": "rich_text_list",
            "style": "bullet",
            "elements": funding_rounds_item_blocks,
        }

        funding_rounds_block = {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": [{"type": "text", "text": "New funding rounds", "style": {"bold": True}}],
                },
                funding_rounds_list_block,
            ],
        }
    else:
        funding_rounds_block = {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": [{"type": "text", "text": "No new funding rounds", "style": {"bold": True}}],
                }
            ],
        }

    return funding_rounds_block


def gen_grants_block(
    grants_data: pd.DataFrame,
) -> RichTextBlock:

    # filter funding rounds data older than 3 months, using announced_on_date
    # also filter out deals > £30million, raised_amount_gbp is in 1000s
    date_cutoff = pd.Timestamp(date.today()) - pd.DateOffset(months=3)
    if grants_data["source"].all() == "crunchbase":
        grants_data = grants_data.query("announced_on_date >= @date_cutoff").query("raised_amount_gbp <= 30000")

    if grants_item_blocks := [_to_richtext_item(row, section="grants") for _, row in grants_data.iterrows()]:
        grants_list_block = {
            "type": "rich_text_list",
            "style": "bullet",
            "elements": grants_item_blocks,
        }

        grants_block = {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": [{"type": "text", "text": "New grants", "style": {"bold": True}}],
                },
                grants_list_block,
            ],
        }
    else:
        grants_block = {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": [{"type": "text", "text": "No new grants", "style": {"bold": True}}],
                }
            ],
        }

    return grants_block


def gen_mission_block(
    org_diffs: pd.DataFrame,
    funding_rounds_diffs: pd.DataFrame,
    grants: pd.DataFrame,
    mission: Literal["ASF", "AFS", "AHL", "X"],
    limit: int = DEFAULT_LIMIT,
) -> List[Union[SectionBlock, RichTextBlock]]:

    # Filter/ Select relevant data
    org_res = (
        org_diffs.query("~mission_labels.isna() and mission_labels.str.contains(@mission)")
        .drop_duplicates(subset=["id"])
        .sort_values(by=["investment_opp", "potential_investment_opp", "interesting_foreign_opp"], ascending=False)
    )

    funding_res = (
        funding_rounds_diffs
        # each funding round has the same raised_amount
        .drop_duplicates(subset=["funding_round_id"]).query("mission_labels.str.contains(@mission)")
    )

    grants = grants.query("mission_labels.str.contains(@mission)")

    # Construct mission block
    if mission not in MISSIONS:
        raise ValueError(f"Invalid mission: {mission}")

    mission_info = MISSIONS[mission]
    mission_header = f"{mission_info['emoji']} *{mission_info['title']}*"
    mission_header_block = {"type": "section", "text": {"type": "mrkdwn", "text": mission_header}}

    startups_block = gen_startups_block(org_res.head(limit))
    funding_rounds_block = gen_funding_rounds_block(funding_res.head(limit))
    grants_block = gen_grants_block(grants.head(limit))

    return [mission_header_block, startups_block, funding_rounds_block, grants_block]


def gen_smart_money(
    smart_money_investors: pd.DataFrame,
    funding_rounds_diffs: pd.DataFrame,
    limit: int = DEFAULT_LIMIT,
) -> List[Union[SectionBlock, RichTextBlock]]:

    funding_res = (
        funding_rounds_diffs.merge(
            smart_money_investors[["id", "smart_money_investor"]],
            left_on="investor_id",
            right_on="id",
            how="left",
        )
        # no investor would raise same amount more than once in a funding_round
        # each funding_round has the same raised_amount, but can have multiple investors
        .drop_duplicates(subset=["funding_round_id", "investor_id", "raised_amount_gbp"])
        .query("smart_money_investor == True")  # filter for smart money investors
        .query(
            "mission_labels.str.contains('ASF') or mission_labels.str.contains('AFS') or mission_labels.str.contains('AHL')"
        )
        .head(limit)
    )

    smart_money_header_block = {
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*:money_with_wings: Follow the (smart) money*"},
    }

    if smart_money_item_blocks := [_to_richtext_item(row, section="smart_money") for _, row in funding_res.iterrows()]:
        smart_money_list_block = {"type": "rich_text_list", "style": "bullet", "elements": smart_money_item_blocks}

        smart_money_block = {"type": "rich_text", "elements": [smart_money_list_block]}
    else:
        smart_money_block = {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": [{"type": "text", "text": "No new smart money signals", "style": {"bold": True}}],
                }
            ],
        }

    return [smart_money_header_block, smart_money_block]


def to_slack_signal_msg(
    region: Literal["uk", "eu", "world"],
    org_diffs: pd.DataFrame,
    funding_rounds_diffs: pd.DataFrame,
    grants: pd.DataFrame,
    smart_money_investors: pd.DataFrame,
    limit: int = DEFAULT_LIMIT,  # lift limits for now
) -> List[Union[SlackHeaderBlock, SectionBlock, RichTextBlock, DividerBlock]]:

    if region.lower() not in REGIONS:
        raise ValueError(f"Invalid region: {region}")

    message_blocks = [message_title_block(region)]

    asf_signals = gen_mission_block(
        org_diffs=org_diffs,
        funding_rounds_diffs=funding_rounds_diffs,
        grants=grants,
        mission="ASF",
        limit=limit,
    )

    afs_signals = gen_mission_block(
        org_diffs=org_diffs,
        funding_rounds_diffs=funding_rounds_diffs,
        grants=grants,
        mission="AFS",
        limit=limit,
    )

    ahl_signals = gen_mission_block(
        org_diffs=org_diffs,
        funding_rounds_diffs=funding_rounds_diffs,
        grants=grants,
        mission="AHL",
        limit=limit,
    )

    # x_signals = gen_mission_block(
    #    org_diffs=org_diffs,
    #    funding_rounds_diffs=funding_rounds_diffs,
    #    smart_money_investors=smart_money_investors,
    #    mission="X",
    #    limit=limit,
    # ) # noisy, remove for now

    smart_money_signals = gen_smart_money(
        smart_money_investors=smart_money_investors,
        funding_rounds_diffs=funding_rounds_diffs,
        limit=limit,
    )

    message_blocks.extend(asf_signals)
    message_blocks.extend(divider_block)
    message_blocks.extend(afs_signals)
    message_blocks.extend(divider_block)
    message_blocks.extend(ahl_signals)
    message_blocks.extend(divider_block)
    # message_blocks.extend(x_signals) # noisy, remove for now
    # message_blocks.extend(divider_block) # noisy, remove for now
    message_blocks.extend(smart_money_signals)
    message_blocks.extend(divider_block)

    return message_blocks


message_design = """
{
    "blocks": [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "New investment signals 26-03-2024",
                "emoji": true
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ":potted_plant: *A Sustainable Future*"
            }
        },
        {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": [
                        {
                            "type": "text",
                            "text": "New startups",
                            "style": {
                                "bold": true
                            }
                        }
                    ]
                },
                {
                    "type": "rich_text_list",
                    "style": "bullet",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "link",
                                    "url": "https://slack.com/",
                                    "text": "Airex",
                                    "style": {
                                        "bold": true
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": " is a smart ventilation control that builds intelligent air ventilation units using atmospheric sensors and cloud algorithms [energy efficiency, built environment]"
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 2: "
                                },
                                {
                                    "type": "text",
                                    "text": "this is a list item"
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 3: "
                                },
                                {
                                    "type": "link",
                                    "url": "https://slack.com/",
                                    "text": "with a link",
                                    "style": {
                                        "bold": true
                                    }
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 4: "
                                },
                                {
                                    "type": "text",
                                    "text": "we are near the end"
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 5: "
                                },
                                {
                                    "type": "text",
                                    "text": "this is the end"
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": [
                        {
                            "type": "text",
                            "text": "New funding rounds",
                            "style": {
                                "bold": true
                            }
                        }
                    ]
                },
                {
                    "type": "rich_text_list",
                    "style": "bullet",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 1: "
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 2: "
                                },
                                {
                                    "type": "text",
                                    "text": "this is a list item"
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 3: "
                                },
                                {
                                    "type": "link",
                                    "url": "https://slack.com/",
                                    "text": "with a link",
                                    "style": {
                                        "bold": true
                                    }
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 4: "
                                },
                                {
                                    "type": "text",
                                    "text": "we are near the end"
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 5: "
                                },
                                {
                                    "type": "text",
                                    "text": "this is the end"
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "type": "divider"
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*:hatched_chick: A Fairer Start*"
            }
        },
        {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": [
                        {
                            "type": "text",
                            "text": "New startups",
                            "style": {
                                "bold": true
                            }
                        }
                    ]
                },
                {
                    "type": "rich_text_list",
                    "style": "bullet",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "link",
                                    "url": "https://slack.com/",
                                    "text": "Airex",
                                    "style": {
                                        "bold": true
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": " is a smart ventilation control that builds intelligent air ventilation units using atmospheric sensors and cloud algorithms [energy efficiency, built environment]"
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 2: "
                                },
                                {
                                    "type": "text",
                                    "text": "this is a list item"
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 3: "
                                },
                                {
                                    "type": "link",
                                    "url": "https://slack.com/",
                                    "text": "with a link",
                                    "style": {
                                        "bold": true
                                    }
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 4: "
                                },
                                {
                                    "type": "text",
                                    "text": "we are near the end"
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 5: "
                                },
                                {
                                    "type": "text",
                                    "text": "this is the end"
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": [
                        {
                            "type": "text",
                            "text": "New funding rounds",
                            "style": {
                                "bold": true
                            }
                        }
                    ]
                },
                {
                    "type": "rich_text_list",
                    "style": "bullet",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 1: "
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 2: "
                                },
                                {
                                    "type": "text",
                                    "text": "this is a list item"
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 3: "
                                },
                                {
                                    "type": "link",
                                    "url": "https://slack.com/",
                                    "text": "with a link",
                                    "style": {
                                        "bold": true
                                    }
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 4: "
                                },
                                {
                                    "type": "text",
                                    "text": "we are near the end"
                                }
                            ]
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "item 5: "
                                },
                                {
                                    "type": "text",
                                    "text": "this is the end"
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "type": "divider"
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*:money_with_wings: Follow the (smart) money*\nClean Growth Fund invested in HutanBio on Jan 18, 2024 (Seed round, <link|https://example.com>)"
            }
        }
    ]
}
"""
