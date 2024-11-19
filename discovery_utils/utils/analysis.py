"""
discovery_utils.utils.analysis

General analyses utils
"""

import datetime

from typing import List

import numpy as np
import pandas as pd


def check_valid(check_var: str, check_list: List[str]) -> None:
    """Raise ValueError is check_var not in check_list"""
    if check_var not in check_list:
        raise ValueError(f"{check_var} is not valid, it must be one of {check_list}.")


def impute_empty_periods(
    df_time_period: pd.DataFrame,
    time_period_col: str,
    period: str,
    min_year: int,
    max_year: int,
) -> pd.DataFrame:
    """
    Imputes zero values for time periods without data

    Args:
        df_time_period: A dataframe with a column containing time period data
        time_period_col: Column containing time period data
        period: Time period that the data is grouped by, 'M', 'Q' or 'Y'
        min_year: Earliest year to impute values for
        max_year: Last year to impute values for

    Returns:
        A dataframe with imputed 0s for time periods with no data
    """
    max_year_data = np.nan_to_num(df_time_period[time_period_col].max().year)
    max_year = max(max_year_data, max_year)
    full_period_range = (
        pd.period_range(
            datetime.datetime.strptime(f"01/01/{min_year}", "%d/%m/%Y"),
            datetime.datetime.strptime(f"31/12/{max_year}", "%d/%m/%Y"),
            freq=period,
        )
        .to_timestamp()
        .to_frame(index=False, name=time_period_col)
        .reset_index(drop=True)
    )
    return full_period_range.merge(df_time_period, "left").fillna(0)


def _moving_average(timeseries_df: pd.DataFrame, window: int = 3, replace_columns: bool = False) -> pd.DataFrame:
    """
    Calculate rolling mean of yearly timeseries (not centered)

    Args:
        timeseries_df: Should have a 'year' column and at least one other data column
        window: Window of the rolling mean
        rename_cols: If True, will create new set of columns for the moving average
            values with the name pattern `{column_name}_sma{window}` where sma
            stands for 'simple moving average'; otherwise this will replace the original columns

    Returns:
        Dataframe with moving average values
    """
    # Rolling mean
    df_ma = timeseries_df.rolling(window, min_periods=1).mean().drop("year", axis=1)
    # Create new renamed columns
    if not replace_columns:
        column_names = timeseries_df.drop("year", axis=1).columns
        new_column_names = ["{}_sma{}".format(s, window) for s in column_names]
        df_ma = df_ma.rename(columns=dict(zip(column_names, new_column_names)))
        return pd.concat([timeseries_df, df_ma], axis=1)
    else:
        return pd.concat([timeseries_df[["year"]], df_ma], axis=1)


def process_time_period(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Process time_period column in a time series dataframe"""
    if "time_period" in ts_df.columns:
        return ts_df.assign(year=lambda df: df.time_period.dt.year).drop("time_period", axis=1)
    else:
        return ts_df


def moving_average(ts_df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Calculate moving average for time series with a time_period column"""
    return ts_df.pipe(process_time_period).pipe(_moving_average, window=window, replace_columns=True)


def magnitude_growth(ts_df: pd.DataFrame, year_start: int, year_end: int, window: int = 3) -> pd.DataFrame:
    """
    Calculate time series magnitude, estimates growth and returns a combined dataframe

    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window
        window: Moving average windows size (in years) for the smoothed growth estimate

    Returns:
        Dataframe with magnitude and growth trend estimates; magnitude is in
        absolute units whereas growth is expresed as a percentage
    """
    return (
        magnitude(ts_df, year_start, year_end)
        .to_frame("magnitude")
        .assign(growth=smoothed_growth(ts_df, year_start, year_end, window))
    )


def magnitude(time_series: pd.DataFrame, year_start: int, year_end: int) -> pd.Series:
    """Estimate the average magnitude of a time series within a specified year range.

    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window

    Returns:
        Series with magnitude estimates for all data columns
    """
    magnitude = process_time_period(time_series).set_index("year").loc[year_start:year_end, :].mean()
    return magnitude


def percentage_change(initial_value: float, new_value: float) -> float:
    """Calculate percentage change from first_value to second_value"""
    return (new_value - initial_value) / initial_value * 100


def growth(
    time_series: pd.DataFrame,
    year_start: int,
    year_end: int,
) -> pd.Series:
    """Calculate a growth estimate comparing two years (no smoothing)

    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window
    Returns:
        Series with smoothed growth estimates for all data columns
    """
    # Smooth timeseries
    df = process_time_period(time_series).set_index("year")
    # Percentage change
    return percentage_change(initial_value=df.loc[year_start, :], new_value=df.loc[year_end, :])


def smoothed_growth(time_series: pd.DataFrame, year_start: int, year_end: int, window: int = 3) -> pd.Series:
    """Calculate a growth estimate by using smoothed (rolling mean) time series

    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window
        window: Moving average windows size (in years) for the smoothed growth estimate

    Returns:
        Series with smoothed growth estimates for all data columns
    """
    # Smooth timeseries
    ma_df = moving_average(time_series, window).set_index("year")
    # Percentage change
    return percentage_change(initial_value=ma_df.loc[year_start, :], new_value=ma_df.loc[year_end, :])
