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
    
    def create_tasks(self, start_date, end_date, task_length=settings.TASK_LENGTH,
                     support_size=settings.SUPPORT_SIZE, query_size=settings.QUERY_SIZE):
        """
        Create tasks by sliding window over time.
        Each task is a dict with 'support_x', 'support_y' (dummy), 'query_x', 'query_y'
        (y will be used to compute loss via simulation).
        For simplicity, we store the actual price data and features; the meta-model will
        generate parameters and a simulator will compute returns.
        """
        # Load all data for all symbols
        all_data = self.storage.get_all_symbols_data(self.symbols, start_date, end_date)
        
        # Compute features for each symbol
        for sym in all_data:
            all_data[sym] = compute_features(all_data[sym])
        
        # Align dates across symbols (take intersection of available dates)
        dates = None
        for sym, df in all_data.items():
            if dates is None:
                dates = set(df.index)
            else:
                dates = dates.intersection(df.index)
        dates = sorted(dates)
        if len(dates) < task_length:
            raise ValueError("Not enough aligned data to create tasks")
        
        # Create tasks: each task is a contiguous block of 'task_length' days
        tasks = []
        for i in range(len(dates) - task_length + 1):
            task_dates = dates[i:i+task_length]
            support_dates = task_dates[:support_size]
            query_dates = task_dates[support_size:support_size+query_size]
            
            # Build feature tensors for each symbol (concatenated)
            # For simplicity, we'll just store the whole DataFrame slice; the meta-model will
            # later extract features as needed.
            task = {
                'dates': task_dates,
                'support': {sym: all_data[sym].loc[support_dates] for sym in self.symbols},
                'query': {sym: all_data[sym].loc[query_dates] for sym in self.symbols}
            }
            tasks.append(task)
        
        return tasks
