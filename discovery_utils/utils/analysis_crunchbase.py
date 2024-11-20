"""
discovery_utils.utils.analysis_crunchbase

Analyses utils for Crunchbase data
"""

from typing import List
from typing import Tuple

import altair as alt
import pandas as pd

from discovery_utils.utils.analysis import check_valid
from discovery_utils.utils.analysis import impute_empty_periods
from discovery_utils.utils.charts import configure_plots
from discovery_utils.utils.charts import configure_titles


DEAL_ORDER = ["n/a", "£0-5M", "£5-20M", "£20-100M", "£100M+"]
DEAL_COLOURS = ["#757575", "#0000FF", "#FDB633", "#18A48C", "#F6ADBD"]


def orgs_founded_per_period(cb_orgs: pd.DataFrame, period: str, min_year: int, max_year: int) -> pd.DataFrame:
    """
    Calculate the number of Crunchbase organisations founded per period

    Args:
        cb_orgs: A dataframe with columns for 'id' and 'founded_on' among other data
        period: Time period the data is grouped by, 'M', 'Q' or 'Y'
        min_year: Earliest year to impute values for
        max_year: Last year to impute values for

    Returns:
        A dataframe with the following columns:
            'time_period',
            'no_of_orgs_founded' - number of new organisations founded in a given time period
    """
    check_valid(period, ["year", "month", "quarter"])
    period = period[0].capitalize()
    # Remove orgs that don't have year when they were founded
    cb_orgs = cb_orgs.query("founded_on.notnull()").query("founded_on >= '1980-01-01'")
    cb_orgs = cb_orgs[-cb_orgs.founded_on.isnull()].copy().assign(time_period=lambda x: pd.to_datetime(x.founded_on))
    # Group by time period
    grouped = cb_orgs.groupby(cb_orgs["time_period"].dt.to_period(period)).agg(n_orgs_founded=("id", "count"))
    grouped.index = grouped.index.astype("datetime64[ns]")
    return impute_empty_periods(grouped.reset_index(), "time_period", period, min_year, max_year).astype(
        {"n_orgs_founded": "int"}
    )


def investments_per_period(funding_rounds_df: pd.DataFrame, period: str, min_year: int, max_year: int) -> pd.DataFrame:
    """
    Aggregate the raised investment amount and number of deals across all orgs

    Args:
        cb_funding_rounds: A dataframe with columns for 'funding_round_id', 'raised_amount_usd'
            'raised_amount_gbp' and 'announced_on' among other data
        period: Time period the data is grouped by, 'M', 'Q' or 'Y'
        min_year: Earliest year to impute values for
        max_year: Last year to impute values for

    Returns:
        A dataframe with the following columns:
            'time_period',
            'no_of_rounds' - number of funding rounds (deals) in a given year
            'raised_amount_usd_total' - total raised investment (USD) in a given year, in millions
            'raised_amount_gbp_total' - total raised investment (GBP) in a given year, in millions
    """
    check_valid(period, ["year", "month", "quarter"])
    period = period[0].capitalize()
    # Create time period column
    funding_rounds_df = funding_rounds_df.query("announced_on.notnull()").query("announced_on >= '1980-01-01'")
    funding_rounds_df["time_period"] = pd.to_datetime(funding_rounds_df.announced_on)
    # Group by time period
    grouped = funding_rounds_df.groupby(funding_rounds_df["time_period"].dt.to_period(period)).agg(
        n_rounds=("funding_round_id", "count"),
        raised_amount_usd_total=("raised_amount_usd", "sum"),
        raised_amount_gbp_total=("raised_amount_gbp", "sum"),
    )
    grouped.index = grouped.index.astype("datetime64[ns]")
    return (
        impute_empty_periods(grouped.reset_index(), "time_period", period, min_year, max_year)
        .assign(year=lambda df: df.time_period.dt.year)
        # Convert to millions
        .assign(raised_amount_usd_total=lambda df: df.raised_amount_usd_total / 1e3)
        .assign(raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1e3)
    )


def get_timeseries(
    cb_orgs: pd.DataFrame,
    cb_funding_rounds: pd.DataFrame,
    period: str,
    min_year: int,
    max_year: int,
) -> pd.DataFrame:
    """
    Produce time series data of investment amounts, number of deals and number of new organisations founded

    Args:
        cb_orgs: A dataframe with columns for 'id' and 'founded_on' among other data
        cb_funding_rounds: A dataframe with columns for 'funding_round_id', 'raised_amount_usd'
            'raised_amount_gbp' and 'announced_on' among other data
        period: Time period to group the data by, 'month', 'quarter' or 'year'
        min_year: Earliest year to impute values for
        max_year: Last year to impute values for

    Returns:
        A dataframe with the following columns:
            'time_period',
            'year',
            'n_rounds' - number of funding rounds (deals) in a given year
            'raised_amount_usd_total' - total raised investment (USD) in a given year, in millions
            'raised_amount_gbp_total' - total raised investment (GBP) in a given year, in millions
            'n_orgs_founded' - number of new organisations founded in a given year
    """
    # Number of new companies per year
    time_series_orgs_founded = orgs_founded_per_period(cb_orgs, period, min_year, max_year)
    # Amount of raised investment per year
    time_series_investment = investments_per_period(cb_funding_rounds, period, min_year, max_year)
    # Join up both tables
    time_series_investment["n_orgs_founded"] = time_series_orgs_founded["n_orgs_founded"]
    return time_series_investment[
        ["time_period", "year", "n_rounds", "raised_amount_usd_total", "raised_amount_gbp_total", "n_orgs_founded"]
    ]


def aggregate_by_funding_round_types(funding_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate investment data by funding round type and year

    Returns a dataframe with the following columns:
        - 'year': Year of the funding round
        - 'investment_type': Type of the funding round
        - 'raised_amount_gbp': Total amount raised in GBP
        - 'counts': Number of funding rounds
    """
    return (
        funding_df.drop_duplicates(["funding_round_id"])
        .groupby(["year", "investment_type"])
        .agg(raised_amount_gbp=("raised_amount_gbp", "sum"), counts=("funding_round_id", "count"))
        .reset_index()
    )


def convert_deal_amount_to_range(
    amount: float,
    currency: str = "£",
) -> str:
    """
    Convert amounts to range in millions

    Args:
        amount: Investment amount (in thousands)
        currency: Currency symbol
        categories: If True, adding indicative deal categories
    """
    amount /= 1e3  # convert to millions
    if (amount >= 0.001) and (amount <= 5):
        return f"{currency}0-5M"
    elif (amount > 5) and (amount <= 20):
        return f"{currency}5-20M"
    elif (amount > 20) and (amount <= 100):
        return f"{currency}20-100M"
    elif amount > 100:
        return f"{currency}100M+"
    else:
        return "n/a"


def add_deal_range_column(
    funding_df: pd.DataFrame,
    deal_order: List[str] = DEAL_ORDER,
) -> pd.DataFrame:
    """Add a deal_range column based on the investment amount range"""
    return (
        funding_df.assign(deal_range=lambda df: df.raised_amount_gbp.apply(convert_deal_amount_to_range))
        .astype({"deal_range": "category"})
        .assign(deal_range=lambda df: df.deal_range.cat.set_categories(deal_order))
        .drop_duplicates("funding_round_id")
    )


def get_funding_by_year_and_range(
    funding_df: pd.DataFrame,
    first_year: int,
    last_year: int,
    deal_order: List[str] = DEAL_ORDER,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a dataframe with funding round amounts and counts per year

    Grouped by deal range. Returns two dataframes with a column for
    year, each deal range and a total amount column.
    """
    funding_df_ranges = add_deal_range_column(funding_df, deal_order)

    # Get a long table with deal_range as a column
    deal_data = (
        funding_df_ranges.groupby(["year", "deal_range"], as_index=True, observed=False)
        .agg(
            counts=("funding_round_id", "count"),
            total_amount=("raised_amount_gbp", "sum"),
        )
        .reset_index()
        .query(f"year >= {first_year}")
        .query(f"year <= {last_year}")
        .assign(total_amount=lambda df: df.total_amount / 1e3)
    )
    # Convert the long tables to a wide format
    deal_data_wide_df = (
        deal_data.pivot(index="year", columns="deal_range", values="total_amount")
        .fillna(0)
        .astype(float)
        .reset_index()
        .merge(deal_data.groupby("year").total_amount.sum().reset_index(), on="year", how="left")
    )

    deal_data_wide_counts_df = (
        deal_data.pivot(index="year", columns="deal_range", values="counts")
        .fillna(0)
        .astype(int)
        .reset_index()
        .merge(deal_data.groupby("year").agg(total_counts=("counts", "sum")).reset_index(), on="year", how="left")
    )
    return deal_data_wide_df, deal_data_wide_counts_df


def _chart_investment_types(
    funding_round_types_df: pd.DataFrame,
    value_column: str = "raised_amount_gbp",
    value_label: str = "Raised amount (£ millions)",
    colour_column: str = "investment_type",
    colour_label: str = "Investment type",
    stack_order: str = None,
) -> alt.Chart:
    """Create a bar chart of investment types"""
    if stack_order is None:
        stack_order = colour_column
    fig = (
        alt.Chart(
            funding_round_types_df,
            width=500,
            height=300,
        )
        .mark_bar()
        .encode(
            x=alt.X("year:O", title=""),
            y=alt.Y(f"{value_column}:Q", title=value_label),
            color=alt.Color(
                f"{colour_column}:N",
                title=colour_label,
            ),
            order=alt.Order(stack_order),
            tooltip=["year", colour_column, value_column],
        )
    )
    return configure_plots(fig)


def _chart_deal_sizes(
    funding_round_types_df: pd.DataFrame,
    value_column: str = "raised_amount_gbp",
    value_label: str = "Raised amount (£ millions)",
    colour_column: str = "investment_type",
    colour_label: str = "Investment type",
    stack_order: str = None,
) -> alt.Chart:
    """Create a bar chart of investment types"""
    if stack_order is None:
        stack_order = colour_column
    fig = (
        alt.Chart(
            funding_round_types_df,
            width=500,
            height=300,
        )
        .mark_bar()
        .encode(
            x=alt.X("year:O", title=""),
            y=alt.Y(f"{value_column}:Q", title=value_label),
            color=alt.Color(
                f"{colour_column}:N",
                title=colour_label,
                scale=alt.Scale(
                    domain=DEAL_ORDER,
                    range=DEAL_COLOURS,
                ),
            ),
            order=alt.Order(stack_order),
            tooltip=["year", colour_column, value_column],
        )
    )
    return configure_plots(fig)


def chart_investment_types(funding_round_types_df: pd.DataFrame, title: str = "") -> alt.Chart:
    """Create a bar chart of investment amounts by type"""
    fig = _chart_investment_types(
        funding_round_types_df.assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1e3),
        value_column="raised_amount_gbp",
        value_label="Raised amount (£ millions)",
    )
    fig = configure_titles(fig, title)
    return fig


def chart_investment_types_counts(funding_round_types_df: pd.DataFrame, title: str = "") -> alt.Chart:
    """Create a bar chart of the number of funding rounds by types"""
    fig = _chart_investment_types(
        funding_round_types_df,
        value_column="counts",
        value_label="Number of funding rounds",
    )
    fig = configure_titles(fig, title)
    return fig


def chart_deal_sizes(deal_sizes_df: pd.DataFrame, title: str = "", subtitle: str = "") -> alt.Chart:
    """Create a bar chart of investment amounts by deal size"""
    deal_sizes_df_long = (
        deal_sizes_df.melt(id_vars="year")
        .query("variable != 'total_amount'")
        .query("variable != 'n/a'")
        .rename(columns={"variable": "deal_size", "value": "amount"})
        .astype({"deal_size": "category"})
        .assign(deal_size=lambda x: x.deal_size.cat.set_categories(DEAL_ORDER))
        .sort_values(["year", "deal_size"])
        .assign(
            order=lambda df: df["deal_size"].astype("str").replace({val: str(i) for i, val in enumerate(DEAL_ORDER)})
        )
        .astype({"order": "int"})
    )

    fig = _chart_deal_sizes(
        deal_sizes_df_long,
        value_column="amount",
        value_label="Raised amount (£ millions)",
        colour_column="deal_size",
        colour_label="Deal size",
        stack_order="order",
    )
    fig = configure_titles(fig, [title, subtitle])
    return fig


def chart_deal_sizes_counts(deal_sizes_counts_df: pd.DataFrame, title: str = "", subtitle: str = "") -> alt.Chart:
    """Create a bar chart of the number of funding rounds by deal size"""
    deal_sizes_counts_df_long = (
        deal_sizes_counts_df.melt(id_vars="year")
        .rename(columns={"variable": "deal_size", "value": "counts"})
        .query("deal_size != 'total_counts'")
        .astype({"deal_size": "category"})
        .assign(deal_size=lambda df: df.deal_size.cat.set_categories(DEAL_ORDER))
        .sort_values(["year", "deal_size"])
        .assign(
            order=lambda df: df["deal_size"].astype("str").replace({val: str(i) for i, val in enumerate(DEAL_ORDER)})
        )
        .astype({"order": "int"})
    )

    fig = _chart_deal_sizes(
        deal_sizes_counts_df_long,
        value_column="counts",
        value_label="Number of funding rounds",
        colour_column="deal_size",
        colour_label="Deal size",
        stack_order="order",
    )
    fig = configure_titles(fig, [title, subtitle])
    return fig
