"""
generate_schedule.py

Runs the trained MAML model over each calendar month in the historical
database and writes a CSV like:

    date,mmLots,StopLossCoef1,ProfitTargetCoef1,...
    2020-01-01,32.1,2.3,4.1,...
    2020-02-01,28.5,1.9,3.8,...

The CSV is then placed in MT5's Files directory so the modified EA can
read it during a backtest.
"""

import sys
sys.path.append('.')

import os
import csv
import datetime
import torch
import yaml

from data.storage import DataStorage
from data.features import compute_features, get_feature_columns
from meta_learning.maml import create_maml_model
from meta_learning.adapt import extract_features_from_task
from execution.parameter_validator import validate_parameters
import config.settings as settings


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Where to write the CSV.  Copy this path into the EA (see MQL5 code).
OUTPUT_CSV = os.path.expanduser(
    "~/.wine/drive_c/users/wexlersolk/AppData/Roaming/MetaQuotes/"
    "Terminal/Common/Files/ml_params_schedule.csv"
)

# How many days of history to feed the model each month (support window)
SUPPORT_DAYS = settings.SUPPORT_SIZE


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def month_range(start: datetime.date, end: datetime.date):
    """Yield the first day of every month between start and end."""
    current = start.replace(day=1)
    while current <= end:
        yield current
        # advance one month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def load_model(input_dim, output_dim):
    maml = create_maml_model(input_dim, output_dim,
                             inner_lr=settings.INNER_LEARNING_RATE)
    maml.load_state_dict(
        torch.load(settings.MODEL_SAVE_PATH, map_location='cpu')
    )
    maml.eval()
    return maml


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    # Load symbols
    with open(settings.SYMBOLS_CONFIG, 'r') as f:
        cfg = yaml.safe_load(f)
    symbols = [s['name'] for s in cfg['symbols'] if s.get('tradable', True)]

    storage = DataStorage()
    feature_cols = get_feature_columns()
    param_names  = list(settings.PARAMETER_BOUNDS.keys())

    input_dim  = len(feature_cols) * len(symbols)
    output_dim = len(param_names)

    model = load_model(input_dim, output_dim)

    # Determine date range from DB
    # Load all data once and find common date range
    print("Loading historical data...")
    all_data = {}
    for sym in symbols:
        df = storage.load_bars(sym)
        if not df.empty:
            df = compute_features(df)
            if not df.empty:
                all_data[sym] = df

    if not all_data:
        print("No data found. Run scripts/update_data.py first.")
        return

    available_symbols = list(all_data.keys())
    print(f"Using symbols: {available_symbols}")

    # Find intersection of dates
    dates = None
    for sym, df in all_data.items():
        d = set(df.index)
        dates = d if dates is None else dates.intersection(d)
    dates = sorted(dates)

    if len(dates) < SUPPORT_DAYS + 1:
        print(f"Not enough data. Need at least {SUPPORT_DAYS + 1} aligned days.")
        return

    start_date = dates[SUPPORT_DAYS].date()   # first month we can compute
    end_date   = dates[-1].date()

    print(f"Generating schedule from {start_date} to {end_date} ...")

    rows = []

    for month_start in month_range(start_date, end_date):
        # Find the last SUPPORT_DAYS trading days BEFORE this month
        month_dt = datetime.datetime.combine(month_start, datetime.time.min)
        prior_dates = [d for d in dates if d < month_dt]

        if len(prior_dates) < SUPPORT_DAYS:
            print(f"  {month_start}: not enough prior data, skipping")
            continue

        support_dates = prior_dates[-SUPPORT_DAYS:]

        # Build task support dict
        support = {}
        for sym in available_symbols:
            df_sym = all_data[sym]
            slice_df = df_sym[df_sym.index.isin(support_dates)]
            if not slice_df.empty:
                support[sym] = slice_df

        if not support:
            continue

        task = {'support': support, 'query': {}}

        # Forward pass
        x = extract_features_from_task(task, mode='support')
        with torch.no_grad():
            raw_params = model(x).numpy()

        params_dict = {name: raw_params[i] for i, name in enumerate(param_names)}
        validated   = validate_parameters(params_dict)

        row = {'date': month_start.strftime('%Y.%m.%d')}
        row.update({k: round(v, 4) for k, v in validated.items()})
        rows.append(row)

        print(f"  {month_start}: {validated}")

    if not rows:
        print("No rows generated.")
        return

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = ['date'] + param_names

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSchedule written to: {OUTPUT_CSV}")
    print(f"Total months: {len(rows)}")
    print("\nFirst few rows:")
    for r in rows[:3]:
        print(" ", r)


if __name__ == '__main__':
    main()
