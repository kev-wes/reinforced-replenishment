import json

import numpy as np
import pandas as pd


def pprint_forecast_configs(
    forecast_configs: pd.DataFrame,
    show_only_diff: bool = False,
    crop_configs: bool = False,
) -> pd.DataFrame:
    """Pretty print forecast configs

    Args:
        forecast_configs (pd.DataFrame): forecast configs
        show_only_diff (bool, optional): Show only differences. Defaults to False.
        crop_configs (bool, optional): Crop config (index) for better overview (beta version).

    Returns:
        pd.DataFrame: Formatted forecast config dataframe
    """
    df = (
        pd.json_normalize(
            forecast_configs.drop_duplicates(subset=["created_at"])
            .apply(
                lambda x: json.loads(x.forecast_config) | {"created_at": x.created_at},
                axis=1
            )
            .to_list()
        )
        .set_index("created_at")
        .T
    )
    if crop_configs:
        df = shorten_config(df)

    if show_only_diff:
        return df.loc[lambda x: ~x.eq(x.iloc[:, 0], axis=0).all(1)].dropna(how="all")

    return df


def shorten_config(df):
    df_list = []

    for col_name in df.columns:
        # Create a dataframe with the column data and split index
        col_df = (
            df[col_name]
            .to_frame()
            .assign(
                arg=lambda x: x.index.str.rsplit(".", n=1).str[-1],
                name=lambda x: x.index.str.rsplit(".", n=1).str[0],
            )
        )

        # Create a dictionary for replacements
        replace_dict = {}
        for idx in col_df.index:
            if idx.endswith(".name"):
                # Replace with parameter name
                param_name = col_df.loc[idx, col_name]
                # Items to replace
                replace_dict.update(
                    {
                        (idx[:-5] + ".kwargs"): param_name,
                        (idx[:-5] + ".kwargs.kwargs"): param_name,
                    }
                )
                # Drop indices ending with .name
                col_df.drop(idx, inplace=True)

        # Create a new index called "parameter"
        col_df["parameter"] = np.where(
            col_df["arg"] == col_df["name"],
            col_df["name"],
            col_df["name"].replace(replace_dict) + "." + col_df["arg"],
        )

        # Filter out rows containing 'error_model'
        # Set 'parameter' as the index and drop unnecessary columns and NaN-indices
        final_df = (
            col_df[~col_df.index.str.contains("error_model")]
            .set_index("parameter")
            .drop(columns=["arg", "name"])
            .dropna()[lambda x: ~(x.index.isna())]
        )

        # Catch cases with list of prediction models:
        key_index = "pipeline.kwargs.explaining_prediction_model"
        if key_index in df[col_name].dropna().index:
            df_temp = pd.json_normalize(
                df[col_name].loc["pipeline.kwargs.explaining_prediction_model"]
            ).T
            df_temp.index = (
                "pipeline.kwargs.explaining_prediction_model." + df_temp.index
            )
            df_temp = shorten_config(df_temp)
            # append all configurations
            df_list2 = []
            for i in range(len(df_temp.columns)):
                # if each parameter is only appearing once: -> no need for indexing
                if (df_temp.notna().sum(axis=1) == 1).all():
                    new_index = df_temp[[i]].index
                else:
                    new_index = f"{i}" + df_temp[[i]].index
                df_list2.append(
                    df_temp[[i]]
                    .set_index(new_index)
                    .rename(columns={i: f"{col_name}"})
                    .dropna()
                )

            index_to_remove = (
                replace_dict["pipeline.kwargs"] + ".explaining_prediction_model"
            )
            final_df = pd.concat([final_df.drop(index_to_remove), pd.concat(df_list2)])
        df_list.append(final_df)

    # Merge all dataframes in df_list
    result_df = df_list[0]
    for i in range(1, len(df_list)):
        result_df = result_df.merge(
            df_list[i], left_index=True, right_index=True, how="outer"
        )

    return result_df
