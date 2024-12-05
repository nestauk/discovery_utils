"""
discovery_utils.utils.analysis_gtr

Analyses utils for Gateway to Research data
"""

import pandas as pd

from discovery_utils.utils.analysis import check_valid
from discovery_utils.utils.analysis import impute_empty_periods


def deduplicate_projects(
    projects_df: pd.DataFrame,
    start_date_column: str = "start",
    title_column: str = "title",
    description_column: str = "text",
    funding_amount_column: str = "amount",
) -> pd.DataFrame:
    """
    Deduplicates projects that have the same title and description.

    This can be used to report the overall funding amount of the whole project when it has
    received funding in separate installments across different years.

    Args:
        projects_df: A dataframe with columns with GtR project 'title', 'description',
            'amount' (research funding) and other data

    Returns:
        A dataframe where projects with the exact same title and description have
        been merged and their funding has been summed up
    """
    gtr_docs_summed_amounts = (
        projects_df.groupby([title_column, description_column])
        .agg(amount=(funding_amount_column, "sum"))
        .reset_index()
    )
    # Add the summed up amounts to the project and keep the earliest instance
    # of the duplicates
    return (
        projects_df.drop(funding_amount_column, axis=1)
        .merge(gtr_docs_summed_amounts, on=[title_column, description_column], how="left")
        .sort_values(start_date_column)
        .drop_duplicates([title_column, description_column], keep="first")
        .reset_index(drop=True)
        # Restore previous column order
    )[projects_df.columns]


def funding_per_period(
    projects_df: pd.DataFrame,
    period: str,
    min_year: int,
    max_year: int,
    start_date_column: str = "start",
    id_column: str = "id",
    funding_amount_column: str = "amount",
) -> pd.DataFrame:
    """
    Given a table with projects and their funding, return an aggregation by period

    Args:
        gtr_docs: A dataframe with columns for 'start', 'project_id' and 'amount'
            (research funding) among other project data
        period: Time period to group the data by, 'M', 'Q' or 'Y'
        min_year: Earliest year to impute values for
        max_year: Last year to impute values for

    Returns:
        A dataframe with the following columns:
            'period' - time period
            'no_of_projects' - number of new projects in a given period,
            'amount_total' - total amount of research funding in a given period
    """
    check_valid(period, ["year", "month", "quarter"])
    period = period[0].capitalize()
    # Convert project start dates to time period
    projects_df = (
        projects_df.copy()
        .astype({start_date_column: "datetime64[ns]"})
        .assign(time_period=lambda x: x[start_date_column].dt.strftime("%Y-%m-%d"))
        .astype({"time_period": "datetime64[ns]"})
    )
    # Group by time period
    grouped = projects_df.groupby(projects_df["time_period"].dt.to_period(period)).agg(
        # Number of new projects in a given time period
        n_projects=(id_column, "count"),
        # Total amount of research funding in a given time period
        amount=(funding_amount_column, "sum"),
        amount_median=(funding_amount_column, "median"),
    )
    grouped.index = grouped.index.astype("datetime64[ns]")
    return (
        impute_empty_periods(
            grouped.reset_index(),
            "time_period",
            period,
            min_year,
            max_year,
        )
        .assign(year=lambda df: df.time_period.dt.year)
        .astype({"year": "int", "n_projects": "int"})
    )[["time_period", "year", "n_projects", "amount", "amount_median"]]


def get_timeseries(
    projects_df: pd.DataFrame,
    period: str,
    min_year: int,
    max_year: int,
    id_column: str = "id",
    start_date_column: str = "start",
    title_column: str = "title",
    description_column: str = "text",
    funding_amount_column: str = "amount",
) -> pd.DataFrame:
    """
    Calculate all typical time series from a list of GtR projects

    Args:
        gtr_docs: A dataframe with columns for 'start', 'project_id' and 'amount'
            (research funding) among other project data
        period: Time period to group the data by, 'month', 'quarter' or 'year'

    Returns:
        Dataframe with columns for:
            - 'time_period'
            - 'year'
            - 'no_of_projects'
            - 'amount_total', in GBP
            - 'amount_median', in GBP
    """
    # Deduplicate projects. This is used to report the number of new projects
    # started each period, accounting for cases where the same project has received
    # additional funding in later periods
    projects_dedup_df = deduplicate_projects(
        projects_df, start_date_column, title_column, description_column, funding_amount_column
    )
    # Number of new projects per time period
    time_series_projects = funding_per_period(
        projects_dedup_df, period, min_year, max_year, start_date_column, id_column, funding_amount_column
    )[["time_period", "n_projects"]]
    # Amount of research funding per period (note: here we use the non-duplicated table,
    # to account for additional funding for projects that might have started in earlier periods
    time_series_funding = funding_per_period(
        projects_df, period, min_year, max_year, start_date_column, id_column, funding_amount_column
    )
    # Join up both tables
    time_series_funding["n_projects"] = time_series_projects["n_projects"]
    return time_series_funding
