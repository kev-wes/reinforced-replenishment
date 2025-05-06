from enum import Enum

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset


class FrequencyEnum(Enum):
    monthly = "MS"
    weekly = "W"


def aggregate_daily_forecast_benchmarks(
    df: pd.DataFrame, resolution: str, incomplete_periods: bool, per_benchmark: bool
) -> pd.DataFrame:
    """Aggregate the resolution of the daily forecast benchmark results.

    Args:
        df (pd.DataFrame): Daily forecast benchmark result df.
        resolution (str): Resolution frequency.
        incomplete_periods (bool): Drop incomplete periods if false.
        per_benchmark (bool): Aggregate forecast per benchmark.

    Raises:
        ValueError:

    Returns:
        pd.DataFrame: Aggregated forecast benchmarks
    """
    df_grouped = df.groupby(
        [
            pd.Grouper(key="date", freq=FrequencyEnum[resolution].value)
            if x == "date"
            else x
            for x in df.columns
            if x
            not in ["truth", "prediction", "horizon"]
            + (["max_date"] if not per_benchmark else [])
        ],
        as_index=False,
    ).agg(
        prediction=("prediction", "sum"),
        truth=("truth", "sum"),
        max_date=("max_date", "min"),
        ndays=("date", "nunique"),
    )

    if resolution == "weekly":
        return (
            df_grouped.pipe(
                lambda x: x.loc[x.ndays == 7] if not incomplete_periods else x
            )
            .drop(columns="ndays")
            .assign(
                max_date_rounded=lambda x: x.max_date
                + pd.to_timedelta((6 - x.max_date.dt.weekday) % 7, unit="days"),
                horizon=lambda x: x.date.dt.to_period("W").astype(int)
                - x.max_date_rounded.dt.to_period("W").astype(int)
                if per_benchmark
                else np.nan,
            )
            .drop(columns="max_date_rounded")
        )
    elif resolution == "monthly":
        return (
            df_grouped.pipe(
                lambda x: x.loc[x.ndays == x.date.dt.daysinmonth]
                if not incomplete_periods
                else x
            )
            .drop(columns="ndays")
            .assign(
                max_date_rounded=lambda x: pd.to_datetime(x.max_date.dt.strftime("%Y-%m") + "-01"),
                horizon=lambda x: x.date.dt.to_period("M").astype(int)
                - x.max_date_rounded.dt.to_period("M").astype(int)
                if per_benchmark
                else np.nan,
            )
            .drop(columns="max_date_rounded")
        )
    else:
        raise ValueError("Only weekly and monthly resolution is supported.")


def calculate_max_date_by_horizon(
    df: pd.DataFrame, resolution: str
) -> pd.Series | pd.DataFrame:
    if resolution == "daily":
        return df.date - pd.to_timedelta(df.horizon, unit="d")
    elif resolution == "weekly":
        return df.date - pd.to_timedelta(df.horizon, unit="w")
    elif resolution == "monthly":
        return df.apply(
            lambda x: x.date - DateOffset(months=x.horizon) + pd.offsets.MonthEnd(0),
            axis=1,
        )
    else:
        raise ValueError("Only daily, weekly and monthly resolution is supported.")


class ForecastPipelineError(Exception):
    """Forecast pipeline error."""

    def __init__(self, message):
        super().__init__(message)
