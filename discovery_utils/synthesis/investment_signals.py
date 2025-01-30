"""Investment signals message builder for Slack.

This module handles the creation of investment signals messages for different regions,
including startup, funding round, grant, and smart money information organised by mission.
"""

from datetime import date
from typing import Dict
from typing import List
from typing import Literal
from typing import TypedDict

import pandas as pd

from slack_utils import SlackBlock
from slack_utils import SlackElement
from slack_utils import SlackMessage
from slack_utils import format_date


# Constants
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
DATE_THRESHOLD = pd.DateOffset(months=3)
FUNDING_AMOUNT_THRESHOLD = 30000  # in thousands
SMART_MONEY_MISSION_QUERY = (
    "mission_labels.str.contains('ASF') or "
    "mission_labels.str.contains('AFS') or "
    "mission_labels.str.contains('AHL')"
)


class InvestmentData(TypedDict):
    """Type definition for investment data."""

    region: Literal["uk", "eu", "world"]
    org_data: pd.DataFrame
    funding_data: pd.DataFrame
    grants_data: pd.DataFrame
    smart_money_data: pd.DataFrame


def _calculate_duration_years(start_date: str, end_date: str) -> str:
    """Calculate project duration in years."""
    if not start_date or not end_date:
        return ""

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    years = (end - start).days / 365.25
    if years <= 1.5:
        return f"({round(years)} year)"
    else:
        return f"({round(years)} years)"


def _format_amount(amount: float) -> str:
    """Format currency amount into K/M format."""
    if amount >= 1_000_000:
        return f"£{amount/1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"£{amount/1_000:.0f}K"
    else:
        return f"£{amount:.0f}"


def _format_investment_type(investment_type: str) -> str:
    """Format investment type to be more readable."""
    # Convert snake_case to Title Case
    return " ".join(word.capitalize() for word in investment_type.split("_"))


# Data filtering functions
def filter_startups(df: pd.DataFrame, mission: str, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    """Filter startups based on mission and investment opportunity.

    Args:
        df: DataFrame containing startup data
        mission: Mission identifier to filter by
        limit: Maximum number of records to return

    Returns:
        Filtered DataFrame of startups
    """
    this_year = date.today().year

    # Convert founding date
    df = df.assign(founded_on=pd.to_datetime(df["founded_on"], format="ISO8601", errors="coerce"))

    return (
        df
        # Filter by mission and founded date
        .query("~mission_labels.isna() and mission_labels.str.contains(@mission)")
        .query("founded_on.dt.year >= @this_year")
        # Remove duplicates
        .drop_duplicates(subset=["id"])
        # Sort by investment opportunities
        .sort_values(by=["investment_opp", "potential_investment_opp", "interesting_foreign_opp"], ascending=False)
        .head(limit)
    )


def filter_funding_rounds(df: pd.DataFrame, mission: str, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    """Filter funding rounds based on mission, date, and amount.

    Args:
        df: DataFrame containing funding round data
        mission: Mission identifier to filter by
        limit: Maximum number of records to return

    Returns:
        Filtered DataFrame of funding rounds
    """
    date_cutoff = pd.Timestamp(date.today()) - DATE_THRESHOLD

    return (
        df
        # Filter by mission
        .query("mission_labels.str.contains(@mission)")
        # Filter by date and amount
        .query("announced_on_date >= @date_cutoff")
        .query(f"raised_amount_gbp <= {FUNDING_AMOUNT_THRESHOLD}")
        # Remove duplicates (each round has same amount)
        .drop_duplicates(subset=["funding_round_id"])
        .head(limit)
    )


def filter_grants_by_type(
    df: pd.DataFrame, mission: str, grant_type: Literal["research", "commercial"], limit: int = DEFAULT_LIMIT
) -> pd.DataFrame:
    """Filter grants based on mission, source-specific criteria, and grant type.

    Args:
        df: DataFrame containing grant data
        mission: Mission identifier to filter by
        grant_type: Type of grant ("research" or "commercial")
        limit: Maximum number of records to return

    Returns:
        Filtered DataFrame of grants
    """
    if df.empty:
        return df

    filtered = df.query("mission_labels.str.contains(@mission)")

    # Apply type-specific filters
    if grant_type == "commercial":
        filtered = filtered.query("leadFunder == 'Innovate UK'")
    else:  # research
        filtered = filtered.query("leadFunder != 'Innovate UK'")

    # Apply additional filters for crunchbase grants
    if filtered["source"].all() == "crunchbase":
        date_cutoff = pd.Timestamp(date.today()) - DATE_THRESHOLD
        filtered = filtered.query("announced_on_date >= @date_cutoff").query(
            f"raised_amount_gbp <= {FUNDING_AMOUNT_THRESHOLD}"
        )

    return filtered.head(limit)


def filter_smart_money(
    funding_rounds: pd.DataFrame, smart_money_investors: pd.DataFrame, limit: int = DEFAULT_LIMIT
) -> pd.DataFrame:
    """Filter and combine funding rounds with smart money investors.

    Args:
        funding_rounds: DataFrame containing funding round data
        smart_money_investors: DataFrame containing smart money investor data
        limit: Maximum number of records to return

    Returns:
        Filtered DataFrame of smart money investments
    """
    return (
        funding_rounds.merge(
            smart_money_investors[["id", "smart_money_investor"]],
            left_on="investor_id",
            right_on="id",
            how="left",
        )
        # Remove duplicate investments
        .drop_duplicates(subset=["funding_round_id", "investor_id", "raised_amount_gbp"])
        # Filter for smart money and relevant missions
        .query("smart_money_investor == True")
        .query(SMART_MONEY_MISSION_QUERY)
        .head(limit)
    )


class InvestmentSignalsMessage(SlackMessage):
    """Builds Slack messages for investment signals."""

    def __init__(self, region: Literal["uk", "eu", "world"]):
        """Initialise the message builder.

        Args:
            region: Region identifier for the message

        Raises:
            ValueError: If the region is invalid
        """
        super().__init__()
        if region.lower() not in REGIONS:
            raise ValueError(f"Invalid region: {region}")
        self.region = region.lower()
        self.block_builder = SlackBlock()

    def add_title(self) -> "InvestmentSignalsMessage":
        """Add the title block for investment signals."""
        region_info = REGIONS[self.region]
        title = (
            f"{region_info['emoji']} {region_info['title']} - " f"New investment signals {format_date(date.today())}"
        )
        return self.add_header(title)

    def _create_country_flag(self, country_alpha2: str) -> List[Dict]:
        """Create country flag elements if country code exists."""
        if country_alpha2 != "None":
            return [SlackElement(f"flag-{country_alpha2}").as_emoji(), SlackElement(" ").as_text()]
        return []

    def _create_grant_section(self, grant: pd.Series) -> Dict:
        """Create a rich text section for a grant."""
        elements = self._create_country_flag(grant["country_alpha2"])

        # Format varies based on grant source
        if grant["source"] == "crunchbase":
            amount = _format_amount(grant["raised_amount_gbp"] * 1000)
            date_str = pd.to_datetime(grant["announced_on_date"]).strftime("%b %Y")
            investor_name = "Unknown" if grant["investor_name"] == "Unknown" else grant["investor_name"]
            funding_info = [
                SlackElement("moneybag").as_emoji(),
                SlackElement(f" {amount} • ").as_text(),
                SlackElement("bank").as_emoji(),
                SlackElement(f" {investor_name} • ").as_text(),
                SlackElement("clock1").as_emoji(),
                SlackElement(f" {date_str}").as_text(),
            ]
        else:
            amount = _format_amount(grant["raised_amount_gbp"])
            duration = _calculate_duration_years(grant["start"], grant["end"])
            date_str = f"{pd.to_datetime(grant['start']).strftime('%b %Y')} - {pd.to_datetime(grant['end']).strftime('%b %Y')}"
            funding_info = [
                SlackElement("moneybag").as_emoji(),
                SlackElement(f" {amount} • ").as_text(),
                SlackElement("bank").as_emoji(),
                SlackElement(f" {grant['leadFunder']} • ").as_text(),
                SlackElement("clock1").as_emoji(),
                SlackElement(f" {date_str} {duration}").as_text(),
            ]

        elements.extend(
            [
                # Title line
                SlackElement(grant["title"]).bold().as_text(),
                SlackElement("\n").as_text(),
                # Organisation line
                SlackElement("round_pushpin").as_emoji(),
                SlackElement(" ").as_text(),
                SlackElement(grant["org_name"]).as_text(),
                SlackElement("\n").as_text(),
                # Funding info line
                *funding_info,
                SlackElement("\n").as_text(),
                # Topics line
                SlackElement("label").as_emoji(),
                SlackElement(" ").as_text(),
                SlackElement(grant["topic_labels"].replace(",", ", ")).as_text(),
                SlackElement("\n").as_text(),
                # Link line
                SlackElement("link").as_emoji(),
                SlackElement(" ").as_text(),
                SlackElement("View project details").as_link(
                    grant["cb_url"] if grant["source"] == "crunchbase" else grant["url"]
                ),
                SlackElement("\n ").as_text(),
            ]
        )

        return self.block_builder.create_rich_text_section(elements)

    def _create_funding_section(self, funding: pd.Series) -> Dict:
        """Create a rich text section for a funding round."""
        elements = self._create_country_flag(funding["country_alpha2"])

        # Format the amount and date
        amount = _format_amount(funding["raised_amount_gbp"] * 1000)
        date_str = pd.to_datetime(funding["announced_on_date"]).strftime("%b %d, %Y")

        elements.extend(
            [
                # Company name line
                SlackElement(funding["org_name"]).bold().as_text(),
                SlackElement("\n").as_text(),
                # Description line
                SlackElement("information_source").as_emoji(),
                SlackElement(" ").as_text(),
                SlackElement(funding["short_description"]).as_text(),
                SlackElement("\n").as_text(),
                # Funding info line
                SlackElement("moneybag").as_emoji(),
                SlackElement(f" {amount} • ").as_text(),
                SlackElement("calendar").as_emoji(),
                SlackElement(f" {date_str} ").as_text(),
                SlackElement("\n").as_text(),
                # Topics line
                SlackElement("label").as_emoji(),
                SlackElement(" ").as_text(),
                SlackElement(funding["topic_labels"].replace(",", ", ")).as_text(),
                SlackElement("\n").as_text(),
                # Link line
                SlackElement("link").as_emoji(),
                SlackElement(" ").as_text(),
                SlackElement("View company details").as_link(funding["cb_url"]),
                SlackElement("\n ").as_text(),
            ]
        )

        return self.block_builder.create_rich_text_section(elements)

    def _create_smart_money_section(self, investment: pd.Series) -> Dict:
        """Create a rich text section for a smart money investment."""
        elements = self._create_country_flag(investment["country_alpha2"])

        # Format the date
        date_str = pd.to_datetime(investment["announced_on_date"]).strftime("%b %d, %Y")
        investment_type = _format_investment_type(investment["investment_type"])

        elements.extend(
            [
                # Company name line
                SlackElement(investment["org_name"]).bold().as_text(),
                SlackElement("\n").as_text(),
                # Description line
                SlackElement("information_source").as_emoji(),
                SlackElement(" ").as_text(),
                SlackElement(investment["short_description"]).as_text(),
                SlackElement("\n").as_text(),
                # Investment info line
                SlackElement("moneybag").as_emoji(),
                SlackElement(f" {investment_type} • ").as_text(),
                SlackElement("bank").as_emoji(),
                SlackElement(f" {investment['investor_name']} • ").as_text(),
                SlackElement("calendar").as_emoji(),
                SlackElement(f" {date_str}").as_text(),
                SlackElement("\n").as_text(),
                # Topics line
                SlackElement("label").as_emoji(),
                SlackElement(" ").as_text(),
                SlackElement(investment["topic_labels"].replace(",", ", ")).as_text(),
                SlackElement("\n").as_text(),
                # Link line
                SlackElement("link").as_emoji(),
                SlackElement(" ").as_text(),
                SlackElement("View investment details").as_link(investment["cb_url"]),
                SlackElement("\n ").as_text(),
            ]
        )

        return self.block_builder.create_rich_text_section(elements)

    def _create_startup_section(self, startup: pd.Series) -> Dict:
        """Create a rich text section for a startup."""
        elements = self._create_country_flag(startup["country_alpha2"])

        # Format the founding date
        founded_date = pd.to_datetime(startup["founded_on"]).strftime("%b %Y")

        elements.extend(
            [
                # Company name line
                SlackElement(startup["name"]).bold().as_text(),
                SlackElement("\n").as_text(),
                # Description line
                SlackElement("information_source").as_emoji(),
                SlackElement(" ").as_text(),
                SlackElement(startup["short_description"]).as_text(),
                SlackElement("\n").as_text(),
                # Founded date line
                SlackElement("calendar").as_emoji(),
                SlackElement(f" Founded {founded_date}").as_text(),
                SlackElement("\n").as_text(),
                # Topics line
                SlackElement("label").as_emoji(),
                SlackElement(" ").as_text(),
                SlackElement(startup["topic_labels"].replace(",", ", ")).as_text(),
                SlackElement("\n").as_text(),
                # Link line
                SlackElement("link").as_emoji(),
                SlackElement(" ").as_text(),
                SlackElement("View company details").as_link(startup["cb_url"]),
                SlackElement("\n ").as_text(),
            ]
        )

        return self.block_builder.create_rich_text_section(elements)

    def add_content_section(
        self, title: str, items: List[pd.Series], section_type: Literal["startup", "funding", "grant", "smart_money"]
    ) -> "InvestmentSignalsMessage":
        """Add a content section with title and items."""
        # Add section title
        title_section = self.block_builder.create_rich_text_section([SlackElement(title).bold().as_text()])

        if not items:
            # Add "No items" message if empty
            self.add_rich_text(
                [self.block_builder.create_rich_text_section([SlackElement(f"No {title.lower()}").bold().as_text()])]
            )
            return self

        # Create item sections based on type
        section_creators = {
            "startup": self._create_startup_section,
            "funding": self._create_funding_section,
            "grant": self._create_grant_section,
            "smart_money": self._create_smart_money_section,
        }
        creator = section_creators[section_type]
        item_sections = [creator(item) for item in items]

        # Add the complete rich text block
        self.add_rich_text([title_section, self.block_builder.create_rich_text_list(item_sections)])
        return self

    def add_mission_section(
        self,
        mission: Literal["ASF", "AFS", "AHL", "X"],
        org_data: pd.DataFrame,
        funding_data: pd.DataFrame,
        grants_data: pd.DataFrame,
        limit: int = DEFAULT_LIMIT,
    ) -> "InvestmentSignalsMessage":
        """Add a complete mission section."""
        if mission not in MISSIONS:
            raise ValueError(f"Invalid mission: {mission}")

        # Add mission header
        mission_info = MISSIONS[mission]
        self.add_section(f"{mission_info['emoji']} *{mission_info['title']}*", markdown=True)

        # Filter data using dedicated filtering functions
        startups = filter_startups(org_data, mission, limit)
        funding_rounds = filter_funding_rounds(funding_data, mission, limit)
        # Split grants into research and commercial
        research_grants = filter_grants_by_type(grants_data, mission, "research", limit)
        commercial_grants = filter_grants_by_type(grants_data, mission, "commercial", limit)

        # Add each section
        self.add_content_section("New Startups", startups.to_dict("records"), "startup")
        self.add_content_section("New Funding Rounds", funding_rounds.to_dict("records"), "funding")
        self.add_content_section("New Commercial Grants", commercial_grants.to_dict("records"), "grant")
        # Uncomment to add research grants
        # self.add_content_section(
        #    "New Research Grants",
        #    research_grants.to_dict('records'),
        #    "grant"
        # )

        return self

    def add_smart_money_section(
        self,
        smart_money_data: pd.DataFrame,
    ) -> "InvestmentSignalsMessage":
        """Add the smart money section to the message."""
        self.add_section("*:money_with_wings: Follow the (smart) money*", markdown=True)

        self.add_content_section("Smart Money Investments", smart_money_data.to_dict("records"), "smart_money")

        return self


def create_investment_signals_message(
    region: Literal["uk", "eu", "world"],
    org_diffs: pd.DataFrame,
    funding_rounds_diffs: pd.DataFrame,
    grants: pd.DataFrame,
    smart_money_investors: pd.DataFrame,
    limit: int = DEFAULT_LIMIT,
) -> List[Dict]:
    """Create a complete investment signals message.

    Args:
        region: Region identifier for the message
        org_diffs: DataFrame containing organisation data
        funding_rounds_diffs: DataFrame containing funding round data
        grants: DataFrame containing grant data
        smart_money_investors: DataFrame containing smart money investor data
        limit: Maximum number of records per section

    Returns:
        List of Slack blocks forming the complete message
    """
    message = InvestmentSignalsMessage(region).add_title()

    # Add mission sections
    for mission in ["ASF", "AFS", "AHL"]:
        message.add_mission_section(mission, org_diffs, funding_rounds_diffs, grants, limit).add_divider()

    # Add smart money section
    smart_money_data = filter_smart_money(funding_rounds_diffs, smart_money_investors, limit)
    message.add_smart_money_section(smart_money_data).add_divider()

    return message.build()


# Example usage:
# message_blocks = create_investment_signals_message(
#     region="uk",
#     org_diffs=org_data,
#     funding_rounds_diffs=funding_data,
#     grants=grants_data,
#     smart_money_investors=smart_money_data
# )
#
# webhook.send(blocks=message_blocks)
