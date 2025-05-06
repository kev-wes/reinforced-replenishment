import pandas as pd


def forecast_long_to_wide(
    df: pd.DataFrame,
    forecast_prefix: str | None = None,
    forecast_translation: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Formatting forecast result from long to wide format.

    Args:
        df (pd.DataFrame): Raw forecast result
        forecast_prefix (str | None, optional): Set prefix to forecast columns. Defaults to None.
        forecast_translation (Dict[str, str] | None, optional): Rename forecast columns.
                                                                Defaults to None.

    Returns:
        pd.DataFrame: Forecast result in wide format.
    """
    index = [col for col in df.columns if col.startswith("group.")] + [
        "date",
        "max_date",
        "resolution",
    ]

    if not forecast_translation:
        if forecast_prefix:
            df["created_at"] = f"{forecast_prefix}" + df.created_at.dt.strftime(
                "%Y%m%d%H%M%S"
            )
    else:
        df.replace({"created_at": forecast_translation}, inplace=True)

    return (
        df.assign(
            max_date=lambda x: pd.to_datetime(x.max_date),
            date=lambda x: pd.to_datetime(x.date),
        )
        .pivot_table(values="prediction", columns="created_at", index=index)
        .merge(
            df.loc[lambda x: x.truth.notna()]
            .sort_values("created_at", ascending=False)
            .drop_duplicates(index)
            .set_index(index)["truth"],
            how="left",
            left_index=True,
            right_index=True,
            validate="m:1",
        )
        .assign(
            month=lambda x: x.index.get_level_values("date").strftime("%m").astype(int),
            year=lambda x: x.index.get_level_values("date").strftime("%Y").astype(int),
        )
        .set_index(["month", "year"], append=True)
    )
