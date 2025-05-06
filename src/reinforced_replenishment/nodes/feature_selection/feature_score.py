import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)


def __round_to_figures(
    df_input: pd.DataFrame,
    num: int = 3,
):
    df = df_input.copy()

    # Function to round to a specified number of significant figures
    def round_to_significant_figures(x, sig_figs):
        if pd.isna(x) or x == 0:
            return x
        return round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)

    # Apply rounding to each column
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].apply(lambda x: round_to_significant_figures(x, num))

    return df


def feature_score(
    df_input: pd.DataFrame,
    target: str,
    round_num_figs: int = 3,
    score_func: Union[
        f_classif, mutual_info_classif, f_regression, mutual_info_regression
    ] = f_classif,
):
    """Calculates score for numeric, object and boolean features.

    Args:
        df_input (pd.DataFrame): Table with target & Features.
        target (str): Name of target column.
        round_num_figs (int, optional): Number of figures to round numeric values to. Defaults to 3.
        score_func (f_classif | mutual_info_regression | f_regression  | mutual_info_classif, optional): Scoring function. Defaults to f_classif.

    Returns:
        pd.DataFrame: Forecast result in wide format.
    """
    df = (
        df_input.select_dtypes(include=[np.number, object, bool])
        .fillna(value={target: 0})
        .dropna()
    )
    logger.info(
        f"{(len(df_input)-len(df))} rows of {len(df_input)} dropped due to NaNs"
    )
    if round_num_figs:
        df = __round_to_figures(df, round_num_figs)

    # Remove constant columns
    df = df.loc[:, df.nunique() > 1].reset_index(drop=True)
    # Convert boolean columns to integers
    df = df.apply(lambda x: x.astype(int) if x.dtype == "bool" else x)
    # Encode categorical values
    encoder = OrdinalEncoder()
    object_columns = list(df.select_dtypes(include=object).columns)
    df[object_columns] = pd.DataFrame(
        encoder.fit_transform(df[object_columns]), columns=object_columns
    )

    X = df.drop(columns=[target])
    X.columns = [str(c) for c in X.columns]
    y = df[target].fillna(0)

    selector = SelectKBest(score_func=score_func, k="all")
    _ = selector.fit_transform(X, y)

    feature_score = (
        pd.DataFrame(
            {
                "Feature": X.columns.tolist(),
                "Score": selector.scores_,
            }
        )
    ).sort_values("Score", ascending=False)

    return feature_score
