import copy
import logging
import re
from itertools import product
from pathlib import Path
from typing import Any

from azure.containerregistry import ContainerRegistryClient
from azure.identity import AzureCliCredential
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from kedro.utils import _find_kedro_project
from kedro_datasets.pandas import SQLQueryDataset
from omegaconf import DictConfig, ListConfig
from packaging.version import Version

logger = logging.getLogger(__name__)


def list_range(start: int, end: int, step: int) -> list[int]:
    """
    Creates a list of integers defined by a start, end, and step size.
    Useful for generating sequences like lag lists.

    Example:
    ```yaml
    preprocessing:
    - name: add_lag
    kwargs:
        lags: ${range:2,13,2}
    ```

    Equivalent to:
    ```yaml
    preprocessing:
    - name: add_lag
    kwargs:
        lags: [2, 4, 6, 8, 10, 12]
    ```

    Args:
        start (int): Range start.
        end (int): Range end.
        step (int): Range stepsize.

    Returns:
        list[int]: List with a sequence of integers.
    """
    return list(range(start, end, step))


def append_list(x: list, *args: Any) -> list:
    """
    Appends a value to a list. Useful for combining individual values with lists.

    Example:
    ```yaml
    preprocessing:
    - name: add_lag
    kwargs:
        lags: ${append:${range:2,13,2},52}
    ```

    Equivalent to:
    ```yaml
    preprocessing:
    - name: add_lag
    kwargs:
        lags: [2, 4, 6, 8, 10, 12, 52]
    ```

    Args:
        x (list): List.
        y (Any): Element to append to the list.

    Returns:
        list: Appended list.
    """
    for y in args:
        x = x + [y]
    return x


def create_list(*args) -> list:
    """
    Creates a list from tuple of arguments. Often combined with other resolvers.

    Example:
    ```yaml
    preprocessing:
    - name: drop_columns
    kwargs:
        columns: ${list:Feature_1,Feature_2,Feature_3}
    ```

    Equivalent to:
    ```yaml
    preprocessing:
    - name: drop_columns
    kwargs:
        columns:
        - Feature_1
        - Feature_2
        - Feature_3
    ```

    Returns:
        list: Python list.
    """
    return list(args)


def extend_list(x: list, *args: list) -> list:
    """
    Extends one list with another. Useful for merging lists into a single list.

    Example:
    ```yaml
    preprocessing:
    - name: add_lag
    kwargs:
        lags: ${extend:${list:50,51,52},${range:2,13,2}}
    ```

    Equivalent to:
    ```yaml
    preprocessing:
    - name: add_lag
    kwargs:
        lags: [50, 51, 52, 2, 4, 6, 8, 10, 12]
    ```

    Args:
        x (list): First list.
        args (list): Second list.

    Returns:
        list: Extended list.
    """
    for y in args:
        x = x + y
    return x


def flat_list(*xss: list[list]) -> list:
    """
    Flat a nested list.

    Example:
    ```yaml
    nested_list: [1, 2, 3, [4, 5]]
    flatted_list: ${flat_list:${nested_list}}
    ```

    Equivalent to:
    ```yaml
    flatted_list: [1, 2, 3, 4, 5]
    ```

    Returns:
        list: Flatted list.
    """
    result = []
    for xs in xss:
        for x in xs:
            if isinstance(x, list):
                result.extend(flat_list(x))
            else:
                result.append(x)
    return result


def update_dict(x: dict, y: dict) -> dict:
    """
    Combine to dictionaries.

    Example:
    ```yaml
    object_1:
        key_1: value_1
    object_2:
        key_2: value_2
    object: ${update_dict:object_1,object_2}
    ```

    Equivalent to:
    ```yaml
    object:
        key_1: value_1
        key_2: value_2
    ```

    Args:
        x (dict): First dictionary.
        y (dict): Second dictionary.

    Returns:
        dict: Combined dictionary
    """
    return x | y


def add_lag(feature: str, lags: list[int]) -> dict:
    """
    Creates a complete `add_lag` preprocessor object for the forecast pipeline.

    Example:
    ```yaml
    preprocessing:
    - ${add_lag:Feature_1,${range:2,13,2}}
    ```

    Equivalent to:
    ```yaml
    preprocessing:
    - name: add_lag
    kwargs:
        lags: [2, 4, 6, 8, 10, 12]
        feature: Feature_1
    ```

    Args:
        lags (list[int]): List of lag values.
        feature (str): Feature name.

    Returns:
        dict: A dict with an "add lag" preprocessor object.
    """
    return {
        "name": "add_lag",
        "kwargs": {"lags": lags, "feature": feature},
    }


def add_lags(features: list[str], lags: list[int]) -> list[dict]:
    """
    Creates a list of `add_lag` preprocessor objects for multiple features in the forecast pipeline.

    Example:
    ```yaml
    preprocessing: ${add_lags:${list:Feature_1,Feature_2},${range:2,13,2}}
    ```

    Equivalent to:
    ```yaml
    preprocessing:
    - name: add_lag
    kwargs:
        lags: [2, 4, 6, 8, 10, 12]
        feature: Feature_1
    - name: add_lag
    kwargs:
        lags: [2, 4, 6, 8, 10, 12]
        feature: Feature_2
    ```

    Args:
        lags (list[int]): List of lag values.
        features (list[str]): List of feature names.

    Returns:
        list[dict]: A list with several "add lag" preprocessor objects.
    """
    return [add_lag(x, lags) for x in features]


def assign_value_to_keys(keys: list, value: Any) -> dict:
    """
    Creates a dictionary by assigning a single value to all elements in a list. The first argument
    is the list of keys, and the second argument is the value assigned to all keys.

    Example:
    ```yaml
    aggregation_functions: ${assign_value_to_keys:${create_lagged_feature_names:${list:Feature_1},${range:2,13,2},_Lag_},sum}
    ```

    Equivalent to:
    ```yaml
    aggregation_functions:
    Feature_1_Lag_2: sum
    Feature_1_Lag_4: sum
    Feature_1_Lag_6: sum
    Feature_1_Lag_8: sum
    Feature_1_Lag_10: sum
    Feature_1_Lag_12: sum
    ```

    Args:
        keys (list): List of strings. For example, feature names.
        value (Any): Value to add.

    Returns:
        dict: Dictionary with the keys and the assigned value.
    """
    return {key: value for key in keys}


def assign_values_to_keys(keys: list, values: list) -> dict:
    """
    Creates a dictionary by pairing elements from two lists. The first list provides the keys,
    and the second list provides the corresponding values. Both lists must have the same length.

    Example:
    ```yaml
    preprocessing:
    - name: set_feature_horizon
    kwargs:
        feature_horizon: ${assign_values_to_keys:${create_lagged_feature_names:${list:Feature_1},${range:2,13,2},_Lag_},${range:2,13,2}}
    ```

    Equivalent to:
    ```yaml
    preprocessing:
    - name: set_feature_horizon
    kwargs:
        feature_horizon:
        Feature_1_Lag_2: 2
        Feature_1_Lag_4: 4
        Feature_1_Lag_6: 6
        Feature_1_Lag_8: 8
        Feature_1_Lag_10: 10
        Feature_1_Lag_12: 12
    ```

    Args:
        keys (list): List of strings. For example, feature names.
        values (list): List of values. Must have the same length as the keys.

    Raises:
        ValueError: Error if the list length of keys and values is different.

    Returns:
        dict: Dictionary with the keys and the assigned values.
    """
    if len(keys) != len(values):
        raise ValueError("The list of keys and values must have the same length.")
    return {key: value for key, value in zip(keys, values)}


def create_lagged_feature_names(
    features: list[str], lags: list[int], sep: str
) -> list[str]:
    """
    Generates lagged feature names by combining two lists. The first list contains feature names,
    and the second contains feature lags. A third parameter specifies the separator between the
    feature name and lag.

    Example:
    ```yaml
    explaining_prediction_models:
    - name: FallbackChain
    kwargs:
        depend_columns:
        - ${create_lagged_feature_names:${list:Feature_1},${range:2,13,2},_Lag_}
    ```

    Equivalent to:
    ```yaml
    explaining_prediction_models:
    - name: FallbackChain
    kwargs:
        depend_columns:
        - [Feature_1_Lag_2, Feature_1_Lag_4, Feature_1_Lag_6, Feature_1_Lag_8, Feature_1_Lag_10, Feature_1_Lag_12]
    ```

    Args:
        features (list[str]): List of feature names.
        lags (list[int]): List of lags.
        sep (str): Seperator between the feature names and the lags.

    Returns:
        list[str]: List with lagged feature names.
    """
    return [f"{feature}{sep}{lag}" for feature in features for lag in lags]


def create_lagged_feature_names_2(features: list[str], lags: list[int]) -> list[str]:
    """
    Generates lagged feature names by combining two lists. The first list contains feature names,
    and the second contains feature lags. The lag value is replaced in the position of the curly
    brackets in the feature name.

    Example:
    ```yaml
    explaining_prediction_models:
    - name: FallbackChain
    kwargs:
        depend_columns:
        - ${create_lagged_feature_names:${list:"Feature_Lag_{}"},${range:2,11,2}}
    ```

    Equivalent to:
    ```yaml
    explaining_prediction_models:
    - name: FallbackChain
    kwargs:
        depend_columns:
        - [Feature_Lag_2, Feature_Lag_4, Feature_Lag_6, Feature_Lag_8, Feature_Lag_10]
    ```

    Note: Its important to mark the feature names by quotes otherwise the curly brackets will not
          be interpreted as a string

    Args:
        features (list[str]): List of feature names.
        lags (list[int]): List of lags.

    Returns:
        list[str]: List with lagged feature names.
    """
    return [feature.format(lag) for feature in features for lag in lags]


def create_depend_columns_list(
    preprocessing: ListConfig, lag_list: list[int]
) -> list[str]:
    """
    Generates a list of lagged feature names by iterating through the `add_lag` steps in the preprocessing config.
    It includes only the lags specified in the input `lags` list that are also present in the preprocessing step.
    The naming convention follows the automatically generated lagged feature names by the forecast pipeline.
    Main use is in depend_columns of FallbackChain.

    Example:
    ```yaml
    preprocessing:
        - name: add_lag
            kwargs:
                lags: [1, 2, 3, 4, 5, 6, 7]
        - name: add_lag
            kwargs:
                lags: [1, 7, 14]
                feature: feature_1

    pipeline:
        name: FallbackChain
        kwargs:
            depend_columns:
                - ${create_depend_columns_lists:${preprocessing},${range:1,8,1}}
    ```

    Equivalent to:
    ```yaml
    preprocessing:
    - name: add_lag
        kwargs:
            lags: [1, 2, 3, 4, 5, 6, 7]
    - name: add_lag
        kwargs:
            lags: [1, 7, 14]
            feature: feature_1

    pipeline:
    name: FallbackChain
    kwargs:
        depend_columns:
        - [
            _Lag_1,
            _Lag_2,
            _Lag_3,
            _Lag_4,
            _Lag_5,
            _Lag_6,
            _Lag_7,
            _Lag_1_feature_1,
            _Lag_7_feature_1,
            ]
    ```

    Args:
        preprocessing (ListConfig): The preprocessing configuration containing `add_lag` steps.
        lag_list (list[int]): List of lag values.

    Returns:
        list[str]: A list of lagged feature names.
    """
    lags = set(lag_list)
    result = []
    for step in preprocessing:
        if step["name"] == "add_lag":
            feature_lags = step["kwargs"]["lags"]
            feature_name = step["kwargs"].get("feature", "")
            result.extend(
                [
                    f"_Lag_{feature_lag}_{feature_name}"
                    if feature_name
                    else f"_Lag_{feature_lag}"
                    for feature_lag in feature_lags
                    if feature_lag in lags
                ]
            )
    return result


def create_depend_columns_lists(
    preprocessing: ListConfig, lag_lists: list[list[int]]
) -> list[list[str]]:
    """
    Generates a list of lists with lagged feature names by iterating through the `add_lag` steps in the preprocessing config.
    It includes only the lags specified in the input `lags` list that are also present in the preprocessing step.
    The naming convention follows the automatically generated lagged feature names by the forecast pipeline.
    Main use is in depend_columns of FallbackChain.

    Example:
    ```yaml
    preprocessing:
        - name: add_lag
            kwargs:
                lags: [1, 2, 3, 4, 5, 6, 7]
        - name: add_lag
            kwargs:
                lags: [1, 7, 14]
                feature: feature_1

    fallbackchain_lags:
        - [1, 2, 3, 4, 5, 6, 7]
        - [2, 3, 4, 5, 6, 7, 14]

    pipeline:
        name: FallbackChain
        kwargs:
            depend_columns: ${create_depend_columns_lists:${preprocessing},${fallbackchain_lags}}
    ```

    Equivalent to:
    ```yaml
    preprocessing:
    - name: add_lag
        kwargs:
            lags: [1, 2, 3, 4, 5, 6, 7]
    - name: add_lag
        kwargs:
            lags: [1, 7, 14]
            feature: feature_1

    fallbackchain_lags:
    - [1, 2, 3, 4, 5, 6, 7]
    - [2, 3, 4, 5, 6, 7, 14]

    pipeline:
    name: FallbackChain
    kwargs:
        depend_columns:
        - [
            _Lag_1,
            _Lag_2,
            _Lag_3,
            _Lag_4,
            _Lag_5,
            _Lag_6,
            _Lag_7,
            _Lag_1_feature_1,
            _Lag_7_feature_1,
            ]
        - [
            _Lag_2,
            _Lag_3,
            _Lag_4,
            _Lag_5,
            _Lag_6,
            _Lag_7,
            _Lag_1_feature_1,
            _Lag_7_feature_1,
            _Lag_14_feature_1,
            ]
    ```

    Args:
        preprocessing (ListConfig): The preprocessing config containing `add_lag` steps.
        lags (list[list[int]]): List of list with lags.

    Returns:
        list[list[str]]: A list of lists with lagged feature names.
    """
    result = []
    for lag_list in lag_lists:
        result.append(create_depend_columns_list(preprocessing, lag_list))
    return result


def lagged_features(preprocessing: ListConfig) -> list[str]:
    """
    Generates a list of feature names that are listed in the preprocessing config within an `add_lag` step. To use this
    functions you need to split the preprocessing into two separate lists.

    Example:
    ```yaml
    preprocessing_add_lag:
        - name: add_lag
          kwargs:
              lags: [1, 2, 3, 4, 5, 6, 7]
        - name: add_lag
          kwargs:
              lags: [1, 7, 14]
              feature: feature_1
        - name: add_lag
          kwargs:
              lags: [7, 14, 21]
              feature: feature_2

    preprocessing_drop_columns:
        - name: drop_columns
            kwargs:
                columns: ${lagged_features:${preprocessing_add_lag}}
    ```

    Equivalent to:
    ```yaml
    preprocessing_add_lag:
        - name: add_lag
          kwargs:
              lags: [1, 2, 3, 4, 5, 6, 7]
        - name: add_lag
          kwargs:
              lags: [1, 7, 14]
              feature: feature_1
        - name: add_lag
          kwargs:
              lags: [7, 14, 21]
              feature: feature_2

    preprocessing_drop_columns:
        - name: drop_columns
            kwargs:
                columns: [feature_1, feature_2]
    ```

    Args:
        preprocessing (ListConfig): The preprocessing configuration containing `add_lag` steps.

    Returns:
        list[str]: List of feature names that are being lagged in the preprocessing.
    """
    result = [
        step["kwargs"]["feature"]
        for step in preprocessing
        if step.get("name") == "add_lag"
        and "kwargs" in step
        and "feature" in step["kwargs"]
    ]
    return result


def get_latest_table_version(prefix: str) -> str:
    """
    Get the latest table version of the forecast input table.

    Example:
    ```yaml
    data_source:
        source_type: postgres-table
        source: demo-demand-forecast
        table_name: ${latest:demo_forecast_table}
    ```

    Equivalent to:
    ```yaml
    data_source:
        source_type: postgres-table
        source: demo-demand-forecast
        table_name: demo_forecast_table12345678
    ```

    Args:
        prefix (str): Forecast input table name without the version.

    Returns:
        str: Versioned forecast input table name.
    """
    query = (
        "SELECT table_name FROM information_schema.tables WHERE table_name ~ '"
        + prefix
        + r"\d{10}$'"
    )
    conf_loader = OmegaConfigLoader(
        _find_kedro_project(Path(__file__).resolve()) / settings.CONF_SOURCE
    )
    table_names = (
        SQLQueryDataset(
            credentials=conf_loader["credentials"]["db_credentials"],
            sql=query,
        )
        .load()
        .table_name.tolist()
    )

    if table_names:
        table_names.sort(reverse=True)
        return table_names[0]
    else:
        return prefix


def find_prediction_models(
    config: DictConfig | ListConfig, model_name: str
) -> list[dict]:
    """
    Recursively searches for all occurrences of a model name within a given model configuration.

    Args:
        config (dict | list): The configuration dictionary or list containing nested pipelines.
        model_name (str): The name of the model to search for.

    Returns:
        list: A list of dictionaries where each dictionary represents a found model pipeline.
    """
    pipelines = []
    if isinstance(config, DictConfig):
        if config.get("name") == model_name:
            pipelines.append(config)
        for value in config.values():
            pipelines.extend(find_prediction_models(value, model_name))
    elif isinstance(config, ListConfig):
        for item in config:
            pipelines.extend(find_prediction_models(item, model_name))
    return pipelines


def generate_forecast_configs(
    base_config: DictConfig, model_params: ListConfig
) -> ListConfig:
    """
    Creates a list of forecast configurations. Generates all possible forecast configurations by
    applying parameter combinations to all occurrences of a model.

    Example:
    ```yaml
    demo_forecast_config:
    ...
    pipeline:
        name: DefaultGlobalPredictionModel
        kwargs:
        objective: regression
        eta_end: 0.1
        scaler:

    model_params:
    - DefaultGlobalPredictionModel:
        objective: [regression, tweedie]
        num_leaves: [10, 30]

    demo.forecast_config: ${generate_forecast_configs:${demo_forecast_config},${model_params}}
    ```

    Equivalent to:
    ```yaml
    demo.forecast_config:
    - ...
    pipeline:
        name: DefaultGlobalPredictionModel
        kwargs:
        objective: regression
        eta_end: 0.1
        num_leaves: 10
        scaler:
    - ...
    pipeline:
        name: DefaultGlobalPredictionModel
        kwargs:
        objective: regression
        eta_end: 0.1
        num_leaves: 30
        scaler:
    - ...
    pipeline:
        name: DefaultGlobalPredictionModel
        kwargs:
        objective: tweedie
        eta_end: 0.1
        num_leaves: 10
        scaler:
    - ...
    pipeline:
        name: DefaultGlobalPredictionModel
        kwargs:
        objective: tweedie
        eta_end: 0.1
        num_leaves: 30
        scaler:
    ```

    Args:
        base_config (dict): The base configuration containing pipeline definitions.
        model_params (list): A list of dictionaries specifying model names and their corresponding
                            parameter values.

    Returns:
        list: A list of configurations, each containing a unique combination of parameters applied
                to all occurrences of the specified models.
    """
    configs = ListConfig([])
    for model_param in model_params:
        model_name = next(iter(model_param.keys()), None)
        if not model_name:
            continue

        pipelines = find_prediction_models(base_config, model_name)
        if not pipelines:
            continue

        param_combinations = list(product(*model_param[model_name].values()))
        param_keys = list(model_param[model_name].keys())

        for values in param_combinations:
            new_config = copy.deepcopy(base_config)
            pipelines = find_prediction_models(new_config, model_name)
            for pipeline in pipelines:
                for key, value in zip(param_keys, values):
                    if "kwargs" not in pipeline.keys():
                        pipeline["kwargs"] = {}
                    pipeline["kwargs"][key] = value
            configs.append(new_config)

    return configs


def get_latest_image_tag(image_name: str) -> str:
    """Get latest image version of an image name in the wdlcontainers registry.

    Example:
    ```yaml
    demo.forecast_api_config:
        batch:
            company_name: demo_company
            pool_id: small
            image_tag: ${latest_image:forecastpipeline}
    ```

    Equivalent to:
    ```yaml
    demo.forecast_api_config:
        batch:
            company_name: demo_company
            pool_id: small
            image_tag: 9.0.1
    ```

    Args:
        image_name (str): Image name.

    Returns:
        str: Latest image tag
    """
    try:
        credential = AzureCliCredential()
        client = ContainerRegistryClient("https://wdlcontainers.azurecr.io", credential)

        tags = [tag.name for tag in client.list_tag_properties(image_name)]

        version_pattern = re.compile(r"^\d+\.\d+\.\d+$")
        valid_versions = [Version(tag) for tag in tags if version_pattern.match(tag)]

        return max(valid_versions).__str__()
    except Exception:
        logger.warning(
            "Latest image tag of forecastpipeline not found! Using tag 'latest'."
        )
        return "latest"
