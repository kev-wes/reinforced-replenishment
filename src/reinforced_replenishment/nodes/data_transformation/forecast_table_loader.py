import os
from datetime import datetime

import pandas as pd
from kedro_datasets.pandas import SQLQueryDataset


class ForecastTableLoader:
    """Load a forecast table for a specific namespace."""

    def __init__(
        self,
        namespace: str,
        forecast_key: list[str] | None = None,
        created_at: list[str] | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        last_n: int | None = None,
    ) -> None:
        """Initialise ForecastTableLoader.

        Args:
            namespace (str): Forecast namespace.
            forecast_key (list[str] | None, optional): Forecast key. Defaults to None.
            created_at (list[str] | None, optional): Forecast creation timestamp. Defaults to None.
            start: (str | datetime | None, optional): Filter created_at greater or equal.
            end: (str | datetime | None, optional): Filter created_at less equal.
            last_n: (int | None, optional): Filter last n created_at's.
        """
        self.namespace = namespace
        self.forecast_key = forecast_key
        self.created_at = created_at
        self.start = start
        self.end = end
        self.last_n = last_n
        self.credentials = dict(con=os.environ["POSTGRES_CONNECTION_STRING"])

    def foresight_forecast(self) -> pd.DataFrame:
        """Load foresight forecasts.

        Returns:
            pd.DataFrame: foresight forecasts
        """
        return self.__load_data(
            self.__build_sql_query(f"{self.namespace}_foresight_forecasts")
        )

    def hindsight_forecast(self) -> pd.DataFrame:
        """Load hindsight forecasts.

        Returns:
            pd.DataFrame: hindsight forecasts
        """
        return self.__load_data(
            self.__build_sql_query(f"{self.namespace}_hindsight_forecasts")
        )

    def forecast_configs(self) -> pd.DataFrame:
        """Load forecast configs.

        Returns:
            pd.DataFrame: forecast configs
        """
        return self.__load_data(
            self.__build_sql_query(f"{self.namespace}_forecast_configs")
        )

    def forecast_feature_importance(self) -> pd.DataFrame:
        """Load forecast feature importance.

        Returns:
            pd.DataFrame: forecast feature importance
        """
        return self.__load_data(
            self.__build_sql_query(f"{self.namespace}_forecast_feature_importance")
        )

    def __build_sql_query(self, table: str) -> str:
        """Build SQL query for SQLQeryDataset.

        Args:
            table (str): Table name

        Returns:
            str: SQL Query
        """
        query = f'SELECT * FROM "{table}"'

        filter_list = []
        if self.forecast_key:
            forecast_key_string = ", ".join([f"'{x}'" for x in self.forecast_key])
            filter_list.append(f" forecast_key in ({forecast_key_string})")
        if self.created_at:
            created_at_string = ", ".join([f"'{x}'" for x in self.created_at])
            filter_list.append(f" created_at in ({created_at_string})")
        if self.start:
            filter_list.append(f" created_at >= '{self.start}'")
        if self.end:
            filter_list.append(f" created_at <= '{self.end}'")
        if self.last_n:
            filter_list.append(
                f" created_at in (SELECT DISTINCT created_at "
                f"FROM {self.namespace}_foresight_forecasts "
                f"ORDER BY created_at DESC LIMIT {self.last_n})"
            )
        if filter_list:
            return query + " WHERE " + " AND".join(filter_list)

        return query

    def __load_data(self, query: str) -> pd.DataFrame:
        """Load data with kedro SQLQueryDataset

        Args:
            query (str): SQL Query

        Returns:
            pd.DataFrame: pandas dataframe
        """
        return SQLQueryDataset(
            sql=query,
            credentials=self.credentials,
        ).load()
