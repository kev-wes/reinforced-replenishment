"""
This is a boilerplate pipeline 'modelling'
generated using Kedro 0.19.6
"""

import json
import logging
import os
import uuid
import warnings
from datetime import datetime, timedelta
from multiprocessing.pool import ThreadPool
from time import sleep, time

import numpy as np
import pandas as pd
import redis
import requests as re
from tqdm.auto import tqdm

from ...nodes.evaluation import error_metrics
from .utils import (
    ForecastPipelineError,
    aggregate_daily_forecast_benchmarks,
    calculate_max_date_by_horizon,
)

logger = logging.getLogger(__name__)


def get_forecast_pipeline_version(url: str) -> str:
    """Get forecast pipeline version.

    Args:
        url (str): Forecast pipeline URL.

    Returns:
        str: Forecast pipeline version.
    """
    response = re.get(url=f"{url}/openapi.json")
    response.raise_for_status()

    return response.json()["info"]["version"]


def create_forecast_run_config(
    forecast_pipeline_version: str,
    forecast_batch_config: dict | None,
) -> dict:
    """Get forecast run information from forecastpipeline.

    Args:
        forecast_pipeline_version (str): Forecast pipeline version.
        forecast_batch_config (dict | None): Forecast batch config.

    Returns:
        dict: Forecast run configuration.
    """
    return {
        "forecast_run_id": str(uuid.uuid4()),
        "forecast_pipeline_version": forecast_pipeline_version,
        "forecast_batch_config": forecast_batch_config,
        "forecast_start_time": str(datetime.now()),
    }


def validate_forecast_config(url: str, forecast_config: dict):
    """Validate forecast configuration.

    Args:
        url (str): Forecast pipeline url.
        forecast_config (dict): Forecast configuration.
    """
    if "n_benchmarks" not in forecast_config.keys():
        raise ValueError("Parameter 'n_benchmarks' is missing in forecast config!")

    if "pipeline" in forecast_config.keys():
        try:
            response = re.post(
                url=f"{url}/validate-pipeline-config",
                json=forecast_config["pipeline"],
            )
            response.raise_for_status()
        except re.exceptions.HTTPError as e:
            try:
                error_message = json.dumps(response.json(), indent=4)
            except Exception:
                error_message = response.text
            raise re.exceptions.HTTPError(f"{e}\n{error_message}") from e


def validate_forecast_configs(url: str, forecast_configs: list[dict]):
    """Validate forecast configurations.

    Args:
        url (str): Forecast pipeline url.
        forecast_configs (list[dict]): Forecast configurations.
    """
    for i, forecast_config in enumerate(forecast_configs):
        try:
            validate_forecast_config(url, forecast_config)
        except Exception:
            logger.error(f"Forecast config {i + 1} raised an error!")
            raise


def start_forecast(url: str, forecast_config: dict) -> str:
    """Start forecast with forecast pipeline api.

    Args:
        url (str): Forecast pipeline URL.
        forecast_config (dict): Forecast configuration.

    Returns:
        str: Forecast key.
    """
    response = re.post(
        url=f"{url}/start-forecast",
        json=forecast_config,
    )
    response.raise_for_status()

    return response.text


def start_forecasts(
    url: str, forecast_config: list[dict], max_forecasts: int | None
) -> list[str]:
    """Start multiple forecasts with forecast pipeline api.

    Args:
        url (str): Forecast pipeline URL.
        forecast_config (dict): Forecast configurations.

    Returns:
        list[str]: Forecast keys.
    """
    response = re.post(
        url=f"{url}/start-forecasts",
        json={"configs": forecast_config, "max_forecasts": max_forecasts},
    )
    response.raise_for_status()

    return response.json()


def start_batch_forecast(url: str, forecast_config: dict, batch_config: dict) -> str:
    """Start forecast with batch api.

    Args:
        url (str): Batch api url.
        forecast_config (dict): Forecast configuration.
        batch_config (dict): Batch configuration.

    Returns:
        str: Forecast key.
    """
    batch_config["forecast_cfg"] = forecast_config
    response = re.post(
        url=f"{url}/submit_task",
        json=dict(
            batch_config,
            **{"forecast_id": None, "forecast_cfg": json.dumps(forecast_config)},
        ),
    )
    response.raise_for_status()

    return response.json()


def start_batch_forecasts(
    url: str, forecast_config: list[dict], batch_config: dict
) -> list[str]:
    """Start multiple forecasts with batch api.

    Args:
        url (str): Batch api url.
        forecast_config (dict): Forecast configuration.
        batch_config (dict): Batch configuration.

    Returns:
        list[str]: Forecast keys.
    """
    response = re.post(
        url=f"{url}/submit_tasks",
        json=dict(
            batch_config,
            **{"forecast_cfgs": [json.dumps(x) for x in forecast_config]},
        ),
    )
    response.raise_for_status()

    return response.json()


def start_forecast_pipeline(
    forecast_api_config: dict | None,
    forecast_config: dict | list[dict],
) -> tuple[dict, dict | list, dict]:
    """Execute forecast pipeline.

    Args:
        forecast_api_config (dict | None): Forecast api config.
        forecast_config (dict): Forecast pipeline config.

    Returns:
        tuple[dict, dict | list, dict]:
            Forecast key with forecast configuration,
            forecast configurations,
            forecast run config.
    """
    forecast_api_config = forecast_api_config or {}

    forecast_run_config = create_forecast_run_config(
        get_forecast_pipeline_version(os.environ["FORECAST_PIPELINE_URL"]),
        forecast_api_config.get("batch"),
    )

    if isinstance(forecast_config, dict):
        if forecast_api_config.get("validate"):
            validate_forecast_config(
                os.environ["FORECAST_PIPELINE_URL"], forecast_config
            )
        if "batch" in forecast_api_config.keys():
            logger.info("Start batch forecast.")
            forecast_key = start_batch_forecast(
                os.environ["BATCH_API_URL"],
                forecast_config,
                forecast_api_config["batch"],
            )
        else:
            logger.info("Start forecast.")
            forecast_key = start_forecast(
                os.environ["FORECAST_PIPELINE_URL"], forecast_config
            )

        return {forecast_key: forecast_config}, forecast_config, forecast_run_config
    elif isinstance(forecast_config, list):
        if forecast_api_config.get("validate"):
            validate_forecast_configs(
                os.environ["FORECAST_PIPELINE_URL"], forecast_config
            )
        if "batch" in forecast_api_config.keys():
            logger.info(f"Start {len(forecast_config)} batch forecasts.")
            forecast_keys = start_batch_forecasts(
                os.environ["BATCH_API_URL"],
                forecast_config,
                forecast_api_config["batch"],
            )
        else:
            logger.info(f"Start {len(forecast_config)} forecasts.")
            forecast_keys = start_forecasts(
                os.environ["FORECAST_PIPELINE_URL"],
                forecast_config,
                forecast_api_config.get("max_forecasts"),
            )

        return (
            {key: config for key, config in zip(forecast_keys, forecast_config)},
            forecast_config,
            forecast_run_config,
        )


def check_forecast_status(
    forecast_key: str,
    redis_host: str,
    redis_port: int,
    redis_db: int,
    redis_password: str | None,
    redis_hash: str,
    no_wait: bool = False,
) -> dict:
    """Check progress of current forecat calculation.

    Args:
        forecast_key (str): Forecast key.
        redis_host (str): Redis host.
        redis_port (int): Redis port.
        redis_db (int): Redis db.
        redis_password (str | None): Redis password.
        redis_hash (str): Redis hash.
        no_wait (bool): Skip status progress.

    Returns:
        dict: Forecast key and runtime of succesfull completed forecast pipeline
    """
    if not no_wait:
        r = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db, password=redis_password
        )
        start_time = None
        with tqdm(total=1, desc=forecast_key, leave=True) as pbar:
            while pbar.n < pbar.total:
                progress = r.hget(redis_hash, forecast_key)
                if isinstance(progress, bytes):
                    progress = json.loads(progress.decode())
                    if progress["type"] == "progress":
                        run_duration = progress["content"]["run_duration"]
                        if start_time is None:
                            start_time = time() - run_duration
                            pbar.start_t = start_time
                        run_progress = round(progress["content"]["progress"], ndigits=5)
                        pbar.n = run_progress
                        pbar.refresh()
                    elif progress["type"] == "error":
                        pbar.desc = pbar.desc + " (ERROR)"
                        pbar.refresh()
                        raise ForecastPipelineError(
                            json.dumps(progress["content"], indent=4)
                        )
                    else:
                        raise ValueError("Undefined progress type.")
                sleep(1)

            pbar.desc = pbar.desc + " (FINISHED)"
            pbar.refresh()
    else:
        run_duration = 0

    return {"forecast_key": forecast_key, "run_duration": run_duration}


def get_forecast_benchmarks(
    forecast_key: str,
    forecast_result_config: dict,
    forecast_config: dict,
    created_at: datetime,
    url: str,
) -> pd.DataFrame:
    """Get forecast result from finished forecast.

    Args:
        forecast_result_config (dict): Forecast result configuration
        forecast_config (dict): Forecast configuration.
        created_at: (datetime): Creation timestamp of the result.
        url (str): Forecast pipeline url.

    Returns:
        pd.DataFrame: Forecast benchmarks.
    """
    forecast_result_config = forecast_result_config.copy()
    benchmark_filters = forecast_result_config.pop("query", None)
    resolution = forecast_config["resolution"]
    if forecast_result_config.get("result_aggregation"):
        result_aggregation = forecast_result_config.pop("result_aggregation")
        if (forecast_config["resolution"] == "daily") | (
            result_aggregation.get("resolution") == forecast_config["resolution"]
        ):
            resolution = result_aggregation["resolution"]
        else:
            raise ValueError(
                "The result aggregation is only supported for the daily forecast."
            )

    response = re.post(
        url=f"{url}/export-benchmark-data",
        json=forecast_result_config | {"forecast_id": forecast_key},
    )
    response.raise_for_status()

    benchmark_data = (
        pd.DataFrame(response.json())
        .rename(
            columns={
                forecast_config["target"]: "truth",
                forecast_config["date_column"]: "date",
            }
            | (
                {x: f"group.{x}" for x in forecast_config["group_columns"]}
                if forecast_config.get("group_columns")
                else {}
            )
            | {"_Horizon": "horizon", "_Prediction": "prediction"}
        )
        .assign(
            date=lambda x: pd.to_datetime(x.date),
            forecast_key=forecast_key,
            created_at=created_at,
            resolution=resolution,
            max_date=lambda x: calculate_max_date_by_horizon(
                df=x[["date", "horizon"]], resolution=forecast_config["resolution"]
            ),
        )
        .pipe(lambda df: df.query(benchmark_filters) if benchmark_filters else df)
    )

    if resolution != forecast_config["resolution"]:
        return aggregate_daily_forecast_benchmarks(
            benchmark_data,
            resolution=resolution,
            incomplete_periods=result_aggregation.get("incomplete_periods", False),
            per_benchmark=result_aggregation.get("per_benchmark", True),
        )
    else:
        return benchmark_data


def get_feature_importance(
    forecast_key: str,
    created_at: datetime,
    url: str,
) -> pd.DataFrame:
    """Get feature importance from forecast api.

    Args:
        forecast_key (str): Forecast id.
        created_at (datetime): Creation timestamp.
        url (str): Forecast pipeline url.

    Returns:
        pd.DataFrame: Feature importance.
    """
    response = re.post(
        url=f"{url}/get-forecast-result",
        json={
            "forecast_id": forecast_key,
            "result_config": {
                "dates": False,
                "truth": False,
                "prediction": False,
                "s_model": False,
                "s": False,
                "df": False,
                "prediction_explanation": False,
                "feature_importance": True,
            },
        },
    )
    response.raise_for_status()
    forecast_result = response.json()
    return pd.DataFrame(
        {
            "feature": forecast_result["feature_importance"].keys(),
            "feature_importance": forecast_result["feature_importance"].values(),
            "forecast_key": forecast_key,
            "created_at": created_at,
        }
    )


def get_forecast_results(
    forecast_keys: dict,
    forecast_result_config: dict | None,
    forecast_run_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get forecast results from finished forecasts.

    Args:
        forecast_keys (dict): Forecast result keys with forecast configuration.
        forecast_result_config (dict | None): Forecast result configuration.
        forecast_run_config (dict): Forecast run configuration.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
            Foresight forecasts, feature importance,
            forecast configuration dataframes and forecast run configuration.
    """
    disable_status_validation = (
        forecast_result_config.pop("disable_status_validation", False)
        if forecast_result_config
        else False
    )
    pool = ThreadPool(len(forecast_keys))
    finished_forecast_keys = []

    for i, forecast_key in enumerate(forecast_keys.keys()):
        finished_forecast_keys.append(
            pool.apply_async(
                check_forecast_status,
                args=(
                    forecast_key,
                    os.environ["REDIS_HOST"],
                    int(os.environ["REDIS_PORT"]),
                    int(os.environ["REDIS_DB"]),
                    os.getenv("REDIS_PASSWORD"),
                    os.environ["REDIS_HASH"],
                    disable_status_validation,
                ),
            )
        )

    pool.close()
    pool.join()

    logger.info("Forecast calculation finished.")

    forecast_benchmarks_list = []
    feature_importance_list = []
    forecast_config_list = []
    forecast_result_config = forecast_result_config or {}
    for forecast_key, finished_forecast_key in zip(
        forecast_keys, finished_forecast_keys
    ):
        created_at = datetime.now()
        forecast_config = pd.DataFrame(
            {
                "forecast_key": forecast_key,
                "forecast_config": [json.dumps(forecast_keys[forecast_key])],
                "forecast_result_config": [json.dumps(forecast_result_config)],
                "created_at": created_at,
                "forecast_run_duration": None,
                "forecast_run_id": forecast_run_config["forecast_run_id"],
                "successful": True,
                "traceback": None,
            }
        )
        try:
            forecast_config["forecast_run_duration"] = str(
                timedelta(seconds=round(finished_forecast_key.get()["run_duration"]))
            )
            logger.info(f"{forecast_key}: Get forecast benchmarks ...")

            forecast_benchmarks = get_forecast_benchmarks(
                forecast_key,
                forecast_result_config,
                forecast_keys[forecast_key],
                created_at,
                os.environ["FORECAST_PIPELINE_URL"],
            )

            logger.info(f"{forecast_key}: Get feature importance ...")

            feature_importance = get_feature_importance(
                forecast_key,
                created_at,
                os.environ["FORECAST_PIPELINE_URL"],
            )

            forecast_benchmarks_list.append(forecast_benchmarks)
            feature_importance_list.append(feature_importance)
            forecast_config_list.append(
                forecast_config.assign(
                    forecast_run_duration=str(
                        timedelta(
                            seconds=round(finished_forecast_key.get()["run_duration"])
                        )
                    ),
                )
            )
        except ForecastPipelineError:
            logging.error("Forecast pipeline raised an error:", exc_info=True)
            forecast_config_list.append(
                forecast_config.assign(
                    traceback=json.loads(finished_forecast_key._value.args[0]),
                    successful=False,
                )
            )
        except Exception:
            raise

    logger.info(
        f"{len(forecast_benchmarks_list)}/{len(forecast_keys)}"
        " forecast results have been received."
    )

    return (
        pd.concat(forecast_benchmarks_list)
        if forecast_benchmarks_list
        else pd.DataFrame(),
        pd.concat(feature_importance_list)
        if feature_importance_list
        else pd.DataFrame(),
        pd.concat(forecast_config_list),
        pd.DataFrame([forecast_run_config]),
    )


def evaluate_forecast_results(
    foresight_forecasts: pd.DataFrame, metrics: list[dict]
) -> pd.DataFrame:
    """Evaluate forecast results.

    Args:
        foresight_forecasts (pd.DataFrame): Forecast forecast results.
        metrics (list[str]): List of metrics to be evaluated.

    Returns:
        pd.DataFrame: Forecast metrics.
    """
    if not foresight_forecasts.empty:
        agg_func = {
            "mean": lambda x: np.mean(x.value),
            "median": lambda x: np.median(x.value),
            "weighted_mean": lambda x: np.average(x.value, weights=x.weights),
        }

        logger.info("Calculate forecast metrics ...")

        evaluation_results = []
        for i, metric in enumerate(metrics):
            tqdm.pandas(desc=f"{i + 1}/{len(metrics)}: {metric['metric'].upper()}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if metric.get("pre_aggregation_freq") or (metric.get(
                    "pre_aggregation_groups"
                ) is not None):
                    groups = (
                        [
                            pd.Grouper(
                                key="date",
                                freq=metric.get("pre_aggregation_freq", None),
                            )
                        ]
                        + [
                            "group." + x
                            for x in metric.get("pre_aggregation_groups", [])
                        ]
                        + ["max_date", "created_at", "forecast_key"]
                    )
                else:
                    groups = []
                evaluation_result = (
                    foresight_forecasts.pipe(
                        lambda df: df.query(metric["query"])
                        if metric.get("query")
                        else df
                    )
                    .pipe(
                        lambda df: df.groupby(
                            groups,
                            as_index=False,
                        )[["truth", "prediction"]].sum()
                        if groups
                        else df
                    )
                    .pipe(
                        lambda df: df.groupby(
                            [
                                x
                                for x in df.columns
                                if x.startswith("group.")
                                or x in ["created_at", "forecast_key"]
                            ]
                        )
                    )
                    .progress_apply(
                        lambda x: pd.Series(
                            {
                                "value": getattr(error_metrics, metric["metric"])(
                                    x.truth.to_numpy(), x.prediction.to_numpy()
                                ),
                                "weights": x.truth.sum(),
                            }
                        )
                    )  # type: ignore
                    .replace([np.inf, -np.inf], np.nan)
                )

                if evaluation_result.value.notna().sum() == 0:
                    logger.warning("No metric could be calculated.")
                    continue
                elif (
                    evaluation_result.shape[0] - evaluation_result.value.notna().sum()
                    > 0
                ):
                    logger.warning(
                        f"{evaluation_result.shape[0] - evaluation_result.value.notna().sum()} "
                        f"of {evaluation_result.shape[0]} "
                        "groups are dropped because no metric could be calculated!"
                    )

                evaluation_results.append(
                    evaluation_result.dropna()
                    .groupby(["created_at", "forecast_key"])
                    .apply(agg_func[metric.get("agg_func", "mean")])
                    .to_frame(name=metric["metric"])
                    .melt(ignore_index=False, var_name="metric")
                    .assign(
                        metric=metric["metric"],
                        agg_func=metric.get("agg_func", "mean"),
                        pre_aggregation_groups=(
                            (
                                "/".join(metric["pre_aggregation_groups"])
                                if len(metric["pre_aggregation_groups"]) > 0
                                else "_TOTAL_"
                            )
                            if metric.get("pre_aggregation_groups") is not None
                            else None
                        ),
                        pre_aggregation_freq=metric.get("pre_aggregation_freq"),
                        n_groups=evaluation_result.value.notna().sum(),
                        query=metric.get("query"),
                        name=metric.get("name"),
                    ).pipe(
                        lambda df: df.round(metric["metric_decimals"])
                        if metric.get("metric_decimals")
                        else df
                    )                )

        return pd.concat(evaluation_results).reset_index()
    return pd.DataFrame()


def upload_to_db(*args: pd.DataFrame) -> tuple:
    """Upload forecast tables to postgres db.

    Returns:
        tuple: Tuples of pandas dataframes.
    """
    return args
