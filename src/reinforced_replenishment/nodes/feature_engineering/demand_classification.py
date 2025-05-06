import numpy as np
import pandas as pd


def classify_demand(
    df_input: pd.DataFrame,
    value_col: str,
    date_col: str,
    group_cols: list[str],
    freq: str,
    sparse_limit: int = 5,
):
    """Calculates average demand interval (adi) and squared coefficient of variation (cov).
    Adi and cov are used to classify the demand (smooth, intermittent, erratic, lumpy)

    Args:
        df_input (pd.dataFrame): Data
        date_col (str): column containing the time
        value_col (str): column that contains values (eg delivered pounds)
        group_cols (List(str)): grouping columns (eg Material, customer, ..)
        freq (str): {'D', 'W', 'M'}; unit of time interval

    Returns:
        pd.DataFrame: adi, cov & classification (smooth, intermittent, erratic, lumpy)
                    for each item in group_col
    """
    df = df_input.copy()
    # Change time resolution
    df[date_col] = df[date_col].dt.to_period(freq).dt.to_timestamp()
    # Remove entries with zero quantity
    df = df[(df[value_col] != 0) & (~df[value_col].isna())]
    # GroupBy time resolution & sort by group & time
    df = (
        df.groupby([*group_cols, date_col])[value_col]
        .sum()
        .reset_index()
        .sort_values(by=[*group_cols, date_col])
    )
    # Average demand interval
    assert freq in ["D", "W", "M"], 'Wrong freq, should be "D", "W" or "M"'
    df["adi"] = df.groupby(group_cols)[date_col].diff()
    df["adi"] = (
        df["adi"] / (30.5 * np.timedelta64(1, "D"))  # type: ignore
        if freq == "M"
        else df["adi"] / np.timedelta64(1, freq)  # type: ignore
    )
    df_adi = df.groupby(group_cols)["adi"].mean().to_frame()
    # Correct for zeros / nans
    df_adi["adi"] = df_adi["adi"].replace(0.0, np.nan)
    # Add number of counts
    df_adi["num_counts"] = df.groupby(group_cols)[value_col].count()
    # COV
    df_cov = df.groupby(group_cols)[value_col].agg(mean_value="mean", std_value="std")
    df_cov["cov"] = (df_cov.std_value / df_cov.mean_value) ** 2
    df_cov["cov"] = df_cov["cov"].replace([0.0, float("inf")], np.nan)
    # Join
    df_adi_cov = pd.merge(df_adi.reset_index(), df_cov, on=group_cols)
    # Classify
    ADI_LIM = 1.32
    COV_LIM = 0.49
    conditions = [
        df_adi_cov["num_counts"] < sparse_limit,
        (df_adi_cov["adi"] <= ADI_LIM) & (df_adi_cov["cov"] <= COV_LIM),
        (df_adi_cov["adi"] > ADI_LIM) & (df_adi_cov["cov"] <= COV_LIM),
        (df_adi_cov["adi"] <= ADI_LIM) & (df_adi_cov["cov"] > COV_LIM),
    ]
    classifications = [f"sparse (<{sparse_limit})", "smooth", "intermittent", "erratic"]

    df_adi_cov["classification"] = np.select(
        conditions, classifications, default="lumpy"
    )

    return df_adi_cov[[*group_cols, "adi", "cov", "classification"]]
