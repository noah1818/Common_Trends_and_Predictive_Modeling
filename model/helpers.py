import os
from typing import Tuple

import pandas as pd
import numpy as np

class Helpers:
    def __init__(self):
        pass

    def _convert_timestamp_to_datetime(self, df_column: pd.Series, t_unit: str = 'ms') -> pd.Series:
        """
        Convert an epoch-based timestamp Series to datetime.

        Parameters
        ----------
        df_column : pandas.Series
            A Series containing integer or float epoch timestamps.
        t_unit : str, default "ms"
            The unit of the epoch timestamps. Common values are:
            - "s"  : seconds
            - "ms" : milliseconds
            - "us" : microseconds
            - "ns" : nanoseconds

        Returns
        -------
        pandas.Series
            A Series with the same index as 'df_column', but converted
            to datetime64[ns] dtype.

        Raises
        ------
        ValueError
            If the provided 't_unit' is not recognized by pandas.
        """
        # Sanity check
        allowed_units = {"s", "ms", "us", "ns"}
        if t_unit not in allowed_units:
            ValueError(f"Invalid time unit {t_unit}. Must be one of {sorted(allowed_units)}.")
        return pd.to_datetime(df_column, unit=t_unit)

    def read_csv_to_dataframe(self, path: str, skip_rows: int, n_rows: int, data_frame_names: list[str]) -> Tuple[pd.DataFrame, ...]:
        """
        Read a delimited text/CSV file, split its columns into 7-column blocks, 
        and return one cleaned DataFrame per requested name.

        Each produced DataFrame has the fixed schema:
        ['machine_time', 'server_time', 'bids', 'asks', 'bid_size', 'ask_size', 'latency'].
        
        Cleaning steps:
        - Convert 'machine_time' and 'server_time' from epoch milliseconds to pandas datetimes.
        - Drop all rows where either 'bids' <= 0 or 'asks' <= 0.

        Parameters
        ----------
        path : str
            Path to the CSV/text file.
        skip_rows : int
            Number of initial rows to skip (passed to pandas 'skiprows').
        n_rows : int
            Number of rows to read (passed to pandas 'nrows').
        data_frame_names : list[str]
            Logical names for each 7-column block in the file. 
            The i-th name consumes columns [i*7 : i*7+7].

        Returns
        -------
        Tuple[pandas.DataFrame, ...]
            A tuple of DataFrames, in the same order as 'data_frame_names'.

        Raises
        ------
        FileNotFoundError
            If 'path' does not exist.
        ValueError
            If arguments are invalid or if the file has insufficient columns.
        pandas.errors.ParserError
            If pandas fails to parse the file.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        if not isinstance(skip_rows, int) or skip_rows < 0:
            raise ValueError(f"'skip_rows' must be an integer >= 0; got {skip_rows}.")
        if not isinstance(n_rows, int) or n_rows < 1:
            raise ValueError(f"'n_rows' must be an integer >= 1; got {n_rows}.")
        if not isinstance(data_frame_names, list) or len(data_frame_names) == 0:
            raise ValueError("'data_frame_names' must be a non-empty list of strings.")
        
        data_frame: pd.DataFrame = pd.read_csv(
            path, skiprows=skip_rows, nrows=n_rows, header = None
        )
        result: list[pd.DataFrame] = []
        total_cols = len(data_frame.columns)

        for i in range(len(data_frame_names)):
            start_col = i * 7
            end_col = start_col + 7
            if end_col > total_cols:
                raise ValueError(
                    f"Not enough columns for block {i} ({start_col}:{end_col}). "
                    f"File has {total_cols} columns."
                )

            cols = data_frame.columns[start_col:end_col]
            df = pd.DataFrame(
                {
                    "machine_time": data_frame[cols[0]],
                    "server_time": data_frame[cols[1]],
                    "bids": data_frame[cols[2]],
                    "asks": data_frame[cols[3]],
                    "bid_size": data_frame[cols[4]],
                    "ask_size": data_frame[cols[5]],
                    "latency": data_frame[cols[6]],
                }
            )

            # Convert timestamps
            df["machine_time"] = self._convert_timestamp_to_datetime(df["machine_time"], t_unit="ms")
            df["server_time"] = self._convert_timestamp_to_datetime(df["server_time"], t_unit="ms")

            result.append(df)

        # ---- Synchronised cleaning across all DataFrames ----
        # build mask from all dfs: drop rows where *any* df has invalid bids/asks
        combined_mask = np.ones(len(result[0]), dtype=bool)
        for df in result:
            combined_mask &= (df["bids"] > 0) & (df["asks"] > 0)

        # apply the same mask to all dfs
        cleaned = [df.loc[combined_mask].reset_index(drop=True) for df in result]

        return tuple(cleaned)
    
    @staticmethod
    def argmin_dict_value(data: dict[int, float]) -> int:
        """
        Find the dictionary key corresponding to the minimum value.

        Parameters
        ----------
        data : dict[int, float]
            A dictionary mapping integer keys to float values.

        Returns
        -------
        int
            The key whose value is the smallest among all entries in 'data'.

        Raises
        ------
        ValueError
            If the dictionary 'data' is empty.
        """
        # Sanity check
        if not data:
            raise ValueError("Input dictionary is empty.")
        
        return min(data, key=data.get)
