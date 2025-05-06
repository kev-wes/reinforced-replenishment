import logging

logger = logging.getLogger(__name__)


def forecast_validator(df):
    """Check whether the forecast scope matches.

    Args:
        df (pd.DataFrame): Input dataframe with group colums as index and and different forecasts as columns.

    Returns: None
    """
    for i, j in enumerate(df.index.names):
        index_all = df.index.get_level_values(i)
        for forecast in df.columns:
            index_fc = df[forecast].dropna().index.get_level_values(i)
            if index_all.nunique() != index_fc.nunique():
                logger.error(
                    f"Only {index_fc.nunique()} unique elements in {forecast} for {j}; expected {index_all.nunique()}."
                )
            if j in ["max_date", "aggregation"]:
                if index_all.max() != index_fc.max():
                    logger.error(
                        f"{j} for {forecast} is {index_fc.max()}; expected {index_all.max()}."
                    )
