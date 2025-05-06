"""
This is a boilerplate pipeline 'modelling'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    evaluate_forecast_results,
    get_forecast_results,
    start_forecast_pipeline,
    upload_to_db,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline(
        [
            node(
                func=start_forecast_pipeline,
                inputs=[
                    "params:forecast_api_config",
                    "params:forecast_config",
                ],
                outputs=[
                    "forecast_key@json",
                    "forecast_config@json",
                    "forecast_run_config@json",
                ],
                name="start_forecast_pipeline_node",
            ),
            node(
                func=get_forecast_results,
                inputs=[
                    "forecast_key@json",
                    "params:forecast_result_config",
                    "forecast_run_config@json",
                ],
                outputs=[
                    "foresight_forecasts_",
                    "forecast_feature_importance_",
                    "forecast_configs_",
                    "forecast_run_config_",
                ],
                name="get_forecast_results_node",
            ),
            node(
                func=evaluate_forecast_results,
                inputs=["foresight_forecasts_", "params:evaluation_config"],
                outputs="forecast_metrics_",
                name="evaluate_forecast_results_node",
            ),
            node(
                func=upload_to_db,
                inputs=[
                    "foresight_forecasts_",
                    "forecast_feature_importance_",
                    "forecast_configs_",
                    "forecast_metrics_",
                    "forecast_run_config_",
                ],
                outputs=[
                    "foresight_forecasts@postgresql",
                    "forecast_feature_importance@postgresql",
                    "forecast_configs@postgresql",
                    "forecast_metrics@postgresql",
                    "forecast_run_configs@postgresql",
                ],
                name="upload_to_db_node",
            ),
        ]
    )

    forecast_pipeline = pipeline(
        pipe=pipeline_instance,
        namespace="demo",
    )

    return forecast_pipeline
