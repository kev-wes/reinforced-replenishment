import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.data.pandas_dataset
import plotly.express as px
import requests as re
from kedro.config import OmegaConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.framework.project import settings
from kedro.pipeline.node import Node
from kedro.utils import _find_kedro_project
from kedro_datasets.pandas import SQLQueryDataset
from mlflow.data.pandas_dataset import PandasDataset

from .nodes.utils.helpers import (
    delete_keys_from_nested_dict,
    get_values_by_key,
)
from .nodes.visualisation.plots import ts_plot

logger = logging.getLogger(__name__)


class MlFlowHooks:
    def __init__(
        self,
    ):
        self.experiment = None
        self.run_ids = {}

    @hook_impl
    def after_node_run(  # noqa: PLR0915
        self,
        node: Node,
        outputs: dict[str, Any],
    ):
        if node.name == f"{node.namespace}.upload_to_db_node":
            # create experiment
            if self.experiment is None:
                mlflow.set_tracking_uri(uri=os.environ["MLFLOW_TRACKING_URI"])
                self.experiment = mlflow.set_experiment(
                    experiment_name="reinforced-replenishment"
                )

            for index, series in outputs[
                f"{node.namespace}.forecast_configs@postgresql"
            ].iterrows():
                # start mlflow run
                logger.info(f"Create mlflow run: {series.forecast_key} (MLflow) ...")

                forecast_config = json.loads(series.forecast_config)

                run_name = (
                    ".".join(
                        get_values_by_key(
                            delete_keys_from_nested_dict(
                                forecast_config["pipeline"],
                                ["error_model", "recursive_pipes"],
                            ),
                            "name",
                        )
                    )
                    if forecast_config.get("pipeline")
                    else "DefaultExplainingPredictionModel"
                )

                with mlflow.start_run(
                    experiment_id=self.experiment.experiment_id,
                    run_name=run_name[: 250 - 3] + "..."
                    if len(run_name) >= 250
                    else run_name,
                ) as run:
                    try:
                        self.run_ids[series.forecast_key] = run.info.run_id

                        # set tags
                        logger.info("Saving tags (MLflow) ...")
                        mlflow.set_tag("namespace", node.namespace)
                        mlflow.set_tag("created_at", series.created_at)
                        mlflow.set_tag("forecast_id", series.forecast_key)
                        mlflow.set_tag(
                            "forecast_run_duration", series.forecast_run_duration
                        )
                        forecast_run_config = outputs[
                            f"{node.namespace}.forecast_run_configs@postgresql"
                        ].squeeze()
                        for key, value in forecast_run_config.items():
                            mlflow.set_tag(key, value)

                        # log params
                        logger.info("Saving parameters (MLflow) ...")
                        mlflow.log_params(
                            {
                                key: json.dumps(value, indent=4)
                                if isinstance(value, (dict, list))
                                else value
                                for key, value in forecast_config.items()
                            }
                        )

                        # log input data
                        logger.info("Saving input data (MLflow) ...")
                        if (
                            forecast_config.get("data_source", {}).get("source_type")
                            == "postgres-table"
                        ):
                            conf_loader = OmegaConfigLoader(
                                _find_kedro_project(Path(__file__).resolve())
                                / settings.CONF_SOURCE
                            )

                            df = SQLQueryDataset(
                                credentials=conf_loader["credentials"][
                                    "db_credentials"
                                ],
                                sql=(
                                    f"SELECT * FROM "
                                    f"{forecast_config['data_source'].get('schema', 'public')}"
                                    f'."{forecast_config["data_source"]["table_name"]}" '
                                    f'WHERE "{forecast_config["target"]}" IS NOT NULL '
                                    "LIMIT 100"
                                ),
                            ).load()
                            dataset: PandasDataset = (
                                mlflow.data.pandas_dataset.from_pandas(
                                    df,
                                    targets=forecast_config["target"],
                                    name=forecast_config["data_source"]["table_name"],
                                )
                            )
                            mlflow.log_input(dataset, context="training")

                        # log artifacts
                        logger.info("Saving artifacts (MLflow) ...")
                        if forecast_run_config.forecast_batch_config:
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                response = re.post(
                                    url=f"{os.environ['BATCH_API_URL']}/task_logs?log_type=stderr",
                                    json={
                                        key: value
                                        for key, value in forecast_run_config[
                                            "forecast_batch_config"
                                        ].items()
                                        if key in ["company_name", "pool_id"]
                                    }
                                    | {"forecast_id": series.forecast_key},
                                )
                                if response.status_code == 200:
                                    forecastpipeline_logs = response.text
                                    log_filename = os.path.join(
                                        tmp_dir, "forecastpipeline.log"
                                    )
                                    with open(log_filename, "w") as f:
                                        f.write(forecastpipeline_logs)
                                    with zipfile.ZipFile(
                                        os.path.join(
                                            tmp_dir, "forecastpipeline.log.zip"
                                        ),
                                        "w",
                                        zipfile.ZIP_DEFLATED,
                                    ) as f:
                                        f.write(
                                            log_filename, os.path.basename(log_filename)
                                        )
                                    with open(
                                        os.path.join(
                                            tmp_dir, "forecastpipeline_tail.log"
                                        ),
                                        "w",
                                    ) as f:
                                        f.write(
                                            "\n".join(
                                                forecastpipeline_logs.splitlines()[
                                                    -1000:
                                                ]
                                            )
                                        )
                                    os.remove(log_filename)
                                    mlflow.log_artifacts(tmp_dir, artifact_path="logs")

                        with tempfile.TemporaryDirectory() as tmp_dir:
                            # log forecast_config.csv
                            with open(f"{tmp_dir}/forecast_config.json", "w") as f:
                                json.dump(
                                    forecast_config,
                                    f,
                                )

                            # log deployment ready forecast_config.csv
                            with open(
                                f"{tmp_dir}/{series.forecast_key}.json", "w"
                            ) as f:
                                json.dump(
                                    delete_keys_from_nested_dict(
                                        forecast_config,
                                        [
                                            "max_date",
                                            "n_benchmarks",
                                            "shift",
                                            "prediction_explanation_method",
                                        ],
                                    ),
                                    f,
                                )

                            # log forecast_result_config.csv
                            forecast_result_config = json.loads(
                                series.forecast_result_config
                            )
                            with open(
                                f"{tmp_dir}/forecast_result_config.json", "w"
                            ) as f:
                                json.dump(
                                    forecast_result_config,
                                    f,
                                )
                            mlflow.log_artifacts(tmp_dir, artifact_path="data")

                        if series.successful:
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                outputs[
                                    f"{node.namespace}.forecast_feature_importance@postgresql"
                                ].loc[
                                    lambda x: x.forecast_key == series.forecast_key
                                ].to_csv(f"{tmp_dir}/forecast_feature_importance.csv")

                                # log forecast_metrics.csv
                                outputs[
                                    f"{node.namespace}.forecast_metrics@postgresql"
                                ].loc[
                                    lambda x: x.forecast_key == series.forecast_key
                                ].to_csv(f"{tmp_dir}/forecast_metrics.csv")

                                mlflow.log_artifacts(tmp_dir, artifact_path="data")

                            # log feature importance figure
                            fig_1 = px.bar(
                                outputs[
                                    f"{node.namespace}.forecast_feature_importance@postgresql"
                                ].loc[lambda x: x.forecast_key == series.forecast_key],
                                x="feature",
                                y="feature_importance",
                                title="Feature importance",
                            )
                            mlflow.log_figure(fig_1, "feature_importance.html")

                            # log timeseries agg figure
                            foresight_forecasts = (
                                outputs[
                                    f"{node.namespace}.foresight_forecasts@postgresql"
                                ]
                                .loc[lambda x: x.forecast_key == series.forecast_key]
                                .assign(
                                    benchmark=lambda x: "benchmark_"
                                    + x.max_date.rank(method="dense")
                                    .astype(int)
                                    .astype(str)
                                )
                            )
                            benchmarks = foresight_forecasts.benchmark.unique().tolist()
                            fig_2 = ts_plot(
                                df=foresight_forecasts.groupby(
                                    ["date", "benchmark"], as_index=False
                                )[["prediction", "truth"]]
                                .sum()
                                .pivot_table(
                                    index=["date", "truth"],
                                    columns="benchmark",
                                    values="prediction",
                                )
                                .reset_index(),
                                x="date",
                                y=["truth"] + benchmarks,
                                title="Forecast timeseries aggregated",
                            )
                            mlflow.log_figure(fig_2, "timeseries_agg.html")

                            # log timeseries figure
                            groups = [
                                x
                                for x in foresight_forecasts.columns
                                if x.startswith("group.")
                            ]
                            fig_3 = ts_plot(
                                df=foresight_forecasts.groupby(
                                    groups + ["date", "benchmark"],
                                    as_index=False,
                                )[["prediction", "truth"]]
                                .sum()
                                .pivot_table(
                                    index=["date", "truth"] + groups,
                                    columns="benchmark",
                                    values="prediction",
                                )
                                .reset_index(),
                                x="date",
                                y=["truth"] + benchmarks,
                                dropdown=groups,
                                limit_dropdown=100,
                                title="Forecast timeseries",
                            )
                            mlflow.log_figure(fig_3, "timeseries.html")

                            # log metrics
                            logger.info("Saving metrics (MLflow) ...")
                            for _, row in (
                                outputs[f"{node.namespace}.forecast_metrics@postgresql"]
                                .loc[lambda x: x.forecast_key == series.forecast_key]
                                .iterrows()
                            ):
                                mlflow.log_metric(
                                    row["name"]
                                    if row["name"] is not None
                                    else ".".join(
                                        s
                                        for s in [
                                            row["metric"],
                                            row["agg_func"],
                                            row["pre_aggregation_groups"],
                                        ]
                                        if s
                                    ),
                                    row["value"],
                                )
                            mlflow.end_run(status="FINISHED")
                        else:
                            mlflow.end_run(status="FAILED")
                    except Exception as e:
                        mlflow.end_run(status="FAILED")
                        raise ValueError(e)
