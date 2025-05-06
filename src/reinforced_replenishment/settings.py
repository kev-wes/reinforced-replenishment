"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

# Instantiated project hooks.
# from reinforced_replenishment.hooks import MlFlowHooks  # noqa: I001

# HOOKS = (MlFlowHooks(),)

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import BaseSessionStore
# SESSION_STORE_CLASS = BaseSessionStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.
# from kedro.config import OmegaConfigLoader

# CONFIG_LOADER_CLASS = OmegaConfigLoader
from reinforced_replenishment.nodes.utils import yaml_resolver  # noqa: E402

# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # "config_patterns": {
    #     "spark" : ["spark*/"],
    #     "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
    # }
    "custom_resolvers": {
        "latest": yaml_resolver.get_latest_table_version,
        "latest_image": yaml_resolver.get_latest_image_tag,
        "range": yaml_resolver.list_range,
        "append": yaml_resolver.append_list,
        "extend": yaml_resolver.extend_list,
        "list": yaml_resolver.create_list,
        "flat_list": yaml_resolver.flat_list,
        "update_dict": yaml_resolver.update_dict,
        "assign_value_to_keys": yaml_resolver.assign_value_to_keys,
        "assign_values_to_keys": yaml_resolver.assign_values_to_keys,
        "add_lag": yaml_resolver.add_lag,
        "add_lags": yaml_resolver.add_lags,
        "lagged_features": yaml_resolver.lagged_features,
        "create_lagged_feature_names": yaml_resolver.create_lagged_feature_names,
        "create_lagged_feature_names_2": yaml_resolver.create_lagged_feature_names_2,
        "create_depend_columns_list": yaml_resolver.create_depend_columns_list,
        "create_depend_columns_lists": yaml_resolver.create_depend_columns_lists,
        "generate_forecast_configs": yaml_resolver.generate_forecast_configs,
    },
}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
