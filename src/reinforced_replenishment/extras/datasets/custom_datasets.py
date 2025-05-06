import csv
import io
import logging
from datetime import datetime

import pandas as pd
from kedro_datasets.pandas import SQLTableDataset

logger = logging.getLogger(__name__)


class CustomSQLTableDataset(SQLTableDataset):
    def __init__(self, **kwargs):
        if kwargs.get("save_args", {}).get("versioned") is not None:
            if kwargs["save_args"]["versioned"]:
                kwargs["table_name"] = kwargs["table_name"] + str(
                    int(round(datetime.now().timestamp()))
                )
            del kwargs["save_args"]["versioned"]
        super().__init__(**kwargs)

    def _psql_insert_copy(self, table, conn, keys, data_iter):
        df = pd.DataFrame(data_iter)
        buf = io.StringIO()
        df.to_csv(
            buf,
            sep="\t",
            index=False,
            header=False,
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
        buf.getvalue()
        buf.seek(0)

        table_name = (
            f'"{table.schema}"."{table.name}"' if table.schema else f'"{table.name}"'
        )
        columns = ", ".join([f'"{key}"' for key in keys])

        sql = f"""
            COPY {table_name} ({columns}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\t', NULL '')
        """

        conn.connection.cursor().copy_expert(sql, buf)

    def _save(self, data: pd.DataFrame) -> None:
        if not data.empty:
            if self._save_args.get("method") == "bulk":
                self._save_args.pop("method")
                with self.engine.begin() as con:
                    if not self._exists() or (
                        self._save_args.get("if_exists") == "replace"
                    ):
                        data.head(0).to_sql(con=con, method="multi", **self._save_args)
                    elif self._exists() and (
                        (self._save_args.get("if_exists") == "fail")
                        or (self._save_args.get("if_exists") is None)
                    ):
                        raise ValueError(
                            f"Table {self._save_args['name']} already exists."
                        )
                    data.to_sql(
                        con=con,
                        method=self._psql_insert_copy,
                        **self._save_args,
                    )
            else:
                data.to_sql(con=self.engine, **self._save_args)
