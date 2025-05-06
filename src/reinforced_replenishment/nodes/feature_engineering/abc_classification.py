import numpy as np
import pandas as pd


def abc_analyse(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Calculate ABC groups.

    A: Products with a value share of 80 cent of the total amount.
    B: Products with a value share of 15 per cent of the total amount.
    C: Products with a value share of 5 per cent of the total amount.

    Args:
        df (pd.DataFrame): Input dataframe
        value_col (str): Value column

    Returns:
        pd.DataFrame: Input dataframe with abc group columns.
    """
    SHARE_A = 0.8
    SHARE_B = 0.95
    return df.sort_values(value_col, ascending=False).assign(
        ORDER_SHARE=lambda x: x[value_col] / x[value_col].sum(),
        ORDER_SHARE_CUM=lambda x: x.ORDER_SHARE.cumsum(),
        ABC=lambda x: np.where(
            x.ORDER_SHARE_CUM <= SHARE_A,
            "A",
            np.where(x.ORDER_SHARE_CUM <= SHARE_B, "B", "C"),
        ),
        N=lambda x: range(1, len(x) + 1),
    )
