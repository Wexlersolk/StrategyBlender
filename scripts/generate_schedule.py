"""
generate_schedule.py — compatible with new RegimeModel checkpoint format
"""
import sys
sys.path.append('.')

import os, csv, yaml, datetime
import numpy as np
import torch
import torch.nn as nn

from data.storage import DataStorage
from data.features import compute_features
import config.settings as settings


class RegimeModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[32, 16]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


OUTPUT_CSV = os.path.expanduser(
    "~/.wine/drive_c/users/wexlersolk/AppData/Roaming/MetaQuotes/"
    "Terminal/Common/Files/ml_params_schedule.csv"
)


def month_range(start, end):
    current = start.replace(day=1)
    while current <= end:
        yield current
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def extract_features(df, feature_cols):
    if df.empty:
        return np.zeros(len(feature_cols))
    feats = df[feature_cols].mean().values.astype(np.float32)
    std = feats.std()
    if std > 1e-8:
        feats = (feats - feats.mean()) / std
    return feats


def main():
    ckpt = torch.load(settings.MODEL_SAVE_PATH, map_location='cpu')

    if 'model_state' not in ckpt:
        print("ERROR: old model format. Run: python scripts/train_meta.py")
        return

    model = RegimeModel(ckpt['input_dim'], ckpt['output_dim'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    base_params   = ckpt['base_params']
    scale_bounds  = ckpt['scale_bounds']
    param_names   = ckpt['param_names']
    symbols       = ckpt['symbols']
    feature_cols  = ckpt['feature_cols']

    print(f"Model loaded | symbols: {symbols} | params: {param_names}")

    storage = DataStorage()
    all_data = {}
    for sym in symbols:
        df = storage.load_bars(sym)
        if not df.empty:
            df = compute_features(df)
            all_data[sym] = df

    if not all_data:
        print("No market data. Run scripts/update_data.py first.")
        return

    all_dates = sorted({d.date() for df in all_data.values() for d in df.index})
    start_date, end_date = all_dates[0], all_dates[-1]
    print(f"Generating schedule: {start_date} → {end_date}")

    rows = []
    for month_start in month_range(start_date, end_date):
        month_dt    = datetime.datetime.combine(month_start, datetime.time.min)
        prior_start = month_dt - datetime.timedelta(days=60)
        prior_end   = month_dt

        feats_list = []
        for sym, df in all_data.items():
            mask = (df.index >= prior_start) & (df.index < prior_end)
            feats_list.append(extract_features(df[mask], feature_cols))

        if not feats_list:
            continue

        features = np.concatenate(feats_list)
        if np.any(~np.isfinite(features)):
            continue

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            scales = model(x).squeeze(0).numpy()

        final_params = {}
        for j, name in enumerate(param_names):
            lo, hi = scale_bounds[name]
            scale = lo + scales[j] * (hi - lo)
            final_params[name] = round(base_params[name] * scale, 4)

        row = {'date': month_start.strftime('%Y.%m.%d')}
        row.update(final_params)
        rows.append(row)
        print(f"  {month_start}: lots={final_params['mmLots']:.1f}  "
              f"SL1={final_params['StopLossCoef1']:.2f}  "
              f"PT1={final_params['ProfitTargetCoef1']:.2f}")

    if not rows:
        print("No rows generated.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['date'] + param_names)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWritten: {OUTPUT_CSV}  ({len(rows)} months)")


if __name__ == '__main__':
    main()
