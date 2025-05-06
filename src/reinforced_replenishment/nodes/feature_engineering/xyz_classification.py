import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from tqdm import tqdm

from ..evaluation.error_metrics import wape


def xyz_analyse(
    df: pd.DataFrame,
    value_col: str,
    date_col: str,
    group_cols: list[str],
    verbose: bool = False,
) -> pd.DataFrame:
    """Calculate XYZ analyse with an arima model.

    Args:
        df (pd.DataFrame): Input dataframe
        value_col (str): Value column
        date_col (str): date column
        group_col (str): Group column

    Returns:
        pd.DataFrame: Input dataframe with xyz columns.
    """
    XYZ_list = []
    for group, value in tqdm(
        df.groupby(group_cols),
        total=len(df[group_cols].drop_duplicates()),
        disable=not verbose,
    ):
        grouped_df = value.loc[:, [date_col, value_col]]
        grouped_df = grouped_df.groupby(date_col).agg("sum")

        try:
            train, test = train_test_split(grouped_df, train_size=0.5)
            model = pm.auto_arima(train, seasonal=True, m=1, d=0)
            forecasts = model.predict(test.shape[0])
        except ValueError:
            XYZ_list.append([*list(group), np.inf])
        else:
            err = wape(test.values.reshape(1, -1)[0], forecasts)
            XYZ_list.append([*list(group), err])

    XYZ_group = pd.DataFrame(XYZ_list, columns=[*group_cols, "ERR"])
    XYZ_group = XYZ_group.sort_values(by=["ERR"], ascending=True)

    XYZ_group["Rank"] = XYZ_group["ERR"].rank(ascending=True)
    rank_max = XYZ_group["Rank"].max()

    XYZ_group["XYZ"] = np.where(
        XYZ_group["Rank"] <= int(0.2 * rank_max),
        "X",
        np.where(XYZ_group["Rank"] <= int(0.5 * rank_max), "Y", "Z"),
    )
    return XYZ_group
