import logging
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from pandas._typing import MergeHow
from tqdm.auto import tqdm

from ..evaluation import error_metrics

logger = logging.getLogger(__name__)


class ForecastResultAggregator:
    """A class for aggregating forecast results and calculating error metrics.

    This class allows the collection of results from different forecasting models
    and provides methods for aggregation and calculation of error metrics such as MAE
    (Mean Absolute Error), RMSE (Root Mean Squared Error), etc.
    """

    def __init__(self, df: pd.DataFrame, verbose: bool = True):
        """Initializes the ForecastResultAggregator class.

        Args:
            df (pd.DataFrame): A pandas dataframe with only forecast results and truth values as
            columns. Grouping columns must be passed as an index.
            The results should be in wide format, typically obtained from the
            `forecast_long_to_wide` function.
        """
        self.data_object = df.copy()
        self.verbose = verbose
        self.aggregation_keys: list = []

    def sum(self, row_count: bool = False, **kwargs):
        """Aggregate data and calculate sum."""
        self.data_object = self.__aggregate_data(lambda x: x.sum(**kwargs), row_count)

        return self

    def median(self, row_count: bool = False):
        """Aggregate data and calculate median."""
        self.data_object = self.__aggregate_data(lambda x: x.median(), row_count)

        return self

    def mean(self, row_count: bool = False):
        """Aggregate data and calculate mean."""
        self.data_object = self.__aggregate_data(lambda x: x.mean(), row_count)

        return self

    def weighted_mean(self, weight_col: str, row_count: bool = False):
        """Aggregate data and calculate weighted mean.

        Args:
            weight_col (str): Name of the column with the weights
            row_count (bool): If true, rows are counted when aggregating data.
        """
        self.data_object = self.__aggregate_data(
            lambda x: pd.Series(
                {
                    col: np.average(x[col], weights=x[weight_col])
                    for col in x.columns
                    if col != weight_col
                },
            ),
            row_count,
        )

        return self

    def rmse(self, dropna: bool = True, row_count: bool = False):
        """Aggregate data and calculate rmse."""
        self.data_object = self.__aggregate_error_metric(
            error_metrics.rmse, dropna, row_count
        )

        return self

    def rmsse(self, dropna: bool = True, row_count: bool = False):
        """Aggregate data and calculate rmsse."""
        self.data_object = self.__aggregate_error_metric(
            error_metrics.rmsse, dropna, row_count
        )

        return self

    def wape(self, dropna: bool = True, row_count: bool = False):
        """Aggregate data and calculate wape."""
        self.data_object = self.__aggregate_error_metric(
            error_metrics.wape, dropna, row_count
        )

        return self

    def smape(self, dropna: bool = True, row_count: bool = False):
        """Aggregate data and calculate smape."""
        self.data_object = self.__aggregate_error_metric(
            error_metrics.smape, dropna, row_count
        )

        return self

    def mase(self, dropna: bool = True, row_count: bool = False):
        """Aggregate data and calculate mase."""
        self.data_object = self.__aggregate_error_metric(
            error_metrics.mase, dropna, row_count
        )

        return self

    def mae(self, dropna: bool = True, row_count: bool = False):
        """Aggregate data and calculate mae."""
        self.data_object = self.__aggregate_error_metric(
            error_metrics.mae, dropna, row_count
        )

        return self

    def spec(self, dropna: bool = True, row_count: bool = False):
        """Aggregate data and calculate spec."""
        self.data_object = self.__aggregate_error_metric(
            error_metrics.spec, dropna, row_count
        )

        return self

    def sc_spec(self, dropna: bool = True, row_count: bool = False):
        """Aggregate data and calculate spec."""
        self.data_object = self.__aggregate_error_metric(
            error_metrics.sc_spec, dropna, row_count
        )

        return self

    def groupby(self, keys: list[str | pd.Grouper]):
        """Set grouping keys for the aggregation. The keys must be present in the index.

        Args:
            keys (List[Union[str, pd.Grouper]]): Names of the index used for grouping.
            count_groups (bool):
                Add a column called "COUNT" to count the number of rows for each group.
        """
        self.aggregation_keys = keys.copy()

        return self

    def merge(self, df: pd.DataFrame, how: MergeHow):
        """Merge additional features on index of the forecast result.

        Args:
            df (pd.DataFrame): DataFrame with additional features.
                               Merge keys must be included in the index.
            how (MergeHow): Merge strategy
        """
        self.data_object = self.data_object.merge(
            df, how=how, left_index=True, right_index=True
        )

        return self

    def get_table(self) -> pd.Series | pd.DataFrame:
        """Get aggregated forecast result.

        Returns:
            Union[pd.Series, pd.DataFrame]: Aggregated forecast result.
        """
        return self.data_object

    def __calculate_error_metric(
        self, x: pd.DataFrame, error_metric: Callable
    ) -> pd.Series:
        """Calculate error metric and return result as pandas series.

        Args:
            x (pd.DataFrame): Unique timeseries with one or more forecast results.
            error_metric (Callable): Error metric.

        Returns:
            pd.Series: Series with Error metrics.
        """
        return pd.Series(
            {
                col: error_metric(x.truth.to_numpy(), x[col].to_numpy())
                for col in x.columns
                if col != "truth"
            },
        )

    def __aggregate_error_metric(
        self, error_metric: Callable, dropna: bool, row_count: bool
    ) -> pd.Series | pd.DataFrame:
        agg_data = self.__aggregate_data(
            lambda x: self.__calculate_error_metric(x, error_metric), row_count
        ).replace([np.inf, -np.inf], np.nan)

        if dropna:
            if self.verbose:
                if (len(agg_data) - len(agg_data.dropna())) > 0:
                    logger.warning(
                        f"{len(agg_data) - len(agg_data.dropna())} of {len(agg_data)} "
                        "groups are dropped because no metric could be calculated!"
                    )
            agg_data.dropna(inplace=True)

        return agg_data

    def __add_row_count(
        self, x: pd.DataFrame | pd.Series, agg_func: Callable
    ) -> pd.Series:
        """Add row count to aggregation func.

        Args:
            x (pd.DataFrame | pd.Series): Data to aggregate
            agg_func (Callable): Aggregation function

        Returns:
            pd.Series: Aggregated data
        """
        aggregated_series = agg_func(x)
        count_series = pd.Series({"COUNT": int(round(len(x)))})
        return pd.concat([count_series, aggregated_series])

    def __apply_agg_func(
        self, x: pd.DataFrame | pd.Series, agg_func: Callable, row_count: bool
    ) -> pd.Series:
        """Applying the aggregation function with optional row counting.

        Args:
            x (pd.DataFrame | pd.Series): Data to aggregate
            agg_func (Callable): Aggregation function
            row_count (bool): If true, rows are counted when aggregating data.

        Returns:
            pd.Series: Aggregated data
        """
        if row_count:
            return self.__add_row_count(x, agg_func)
        else:
            return agg_func(x)

    def __aggregate_data(
        self, agg_func: Callable, row_count: bool
    ) -> pd.Series | pd.DataFrame:
        """Aggregate forecast result.

        If group keys are available, group data and use the aggregation function,
        otherwise only use the aggregation function.

        Args:
            agg_func (Callable): _description_

        Returns:
            Union[pd.Series, pd.DataFrame]: _description_
        """
        tqdm.pandas(disable=not self.verbose)
        if self.aggregation_keys:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                agg_data = (
                    self.data_object.groupby(self.aggregation_keys)
                    .progress_apply(
                        lambda x: self.__apply_agg_func(x, agg_func, row_count)
                    )  # type: ignore
                    .convert_dtypes()
                    .pipe(
                        lambda x: x.set_index("COUNT", append=True) if row_count else x
                    )
                )
            self.aggregation_keys.clear()
        else:
            agg_data = (
                self.__apply_agg_func(self.data_object, agg_func, row_count)
                .to_frame()
                .T.convert_dtypes()
                .pipe(lambda x: x.set_index("COUNT") if row_count else x)
            )

        return agg_data
