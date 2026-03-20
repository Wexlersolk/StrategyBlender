import numpy as np
import pandas as pd
from data.features import compute_features, get_feature_columns
import config.settings as settings


class TaskGenerator:
    """Generates meta-learning tasks from historical data."""

    def __init__(self, storage, symbols):
        self.storage = storage
        self.symbols = symbols
        self.feature_cols = get_feature_columns()

    def create_tasks(self, start_date, end_date,
                     task_length=settings.TASK_LENGTH,
                     support_size=settings.SUPPORT_SIZE,
                     query_size=settings.QUERY_SIZE):
        """
        Create tasks by sliding window over time.
        Skips any symbols that have no data in the database.
        """
        # Load all data — only keep symbols that actually returned data
        all_data_raw = self.storage.get_all_symbols_data(self.symbols, start_date, end_date)

        if not all_data_raw:
            raise ValueError("No data loaded for any symbol. Run update_data.py first.")

        # Compute features, skip symbols where feature computation fails
        all_data = {}
        for sym, df in all_data_raw.items():
            try:
                featured = compute_features(df)
                if not featured.empty:
                    all_data[sym] = featured
                else:
                    print(f"Warning: empty feature DataFrame for {sym}, skipping")
            except Exception as e:
                print(f"Warning: could not compute features for {sym}: {e}")

        if not all_data:
            raise ValueError("No symbols have usable feature data.")

        available_symbols = list(all_data.keys())
        print(f"Training on {len(available_symbols)} symbols: {available_symbols}")

        # Align dates across available symbols (intersection)
        dates = None
        for sym, df in all_data.items():
            sym_dates = set(df.index)
            dates = sym_dates if dates is None else dates.intersection(sym_dates)
        dates = sorted(dates)

        if len(dates) < task_length:
            raise ValueError(
                f"Only {len(dates)} aligned dates available but task_length={task_length}. "
                f"Reduce TASK_LENGTH or fetch more data."
            )

        # Update feature_cols based on actual data
        self.feature_cols = get_feature_columns()

        # Create sliding window tasks
        tasks = []
        for i in range(len(dates) - task_length + 1):
            task_dates = dates[i:i + task_length]
            support_dates = task_dates[:support_size]
            query_dates = task_dates[support_size:support_size + query_size]

            task = {
                'dates': task_dates,
                'support': {sym: all_data[sym].loc[support_dates] for sym in available_symbols},
                'query':   {sym: all_data[sym].loc[query_dates]   for sym in available_symbols},
            }
            tasks.append(task)

        return tasks
