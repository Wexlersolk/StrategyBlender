
import sys
sys.path.append('.')

import os
import yaml
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from data.storage import DataStorage
from data.features import compute_features, get_feature_columns
import config.settings as settings


BASE_PARAMS = {
    'mmLots':            70.0,
    'StopLossCoef1':      2.0,
    'ProfitTargetCoef1':  1.5,
    'StopLossCoef2':      2.4,
    'ProfitTargetCoef2':  1.5,
    'TrailingActCef1':    1.4,
}

SCALE_BOUNDS = {
    'mmLots':            (0.2, 1.0),  # only this moves
    'StopLossCoef1':     (1.0, 1.0),  # locked at base
    'ProfitTargetCoef1': (1.0, 1.0),  # locked at base
    'StopLossCoef2':     (1.0, 1.0),  # locked at base
    'ProfitTargetCoef2': (1.0, 1.0),  # locked at base
    'TrailingActCef1':   (1.0, 1.0),  # locked at base
}

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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def extract_monthly_features(df, feature_cols):
    if df.empty:
        return np.zeros(len(feature_cols))
    feats = df[feature_cols].mean().values.astype(np.float32)
    std = feats.std()
    if std > 1e-8:
        feats = (feats - feats.mean()) / std
    return feats


def sharpe_to_target(sharpe):
    clipped = np.clip(sharpe, -10.0, 10.0)
    return float((clipped + 10.0) / 20.0)


def main():
    available_path = 'config/available_symbols.yaml'
    if os.path.exists(available_path):
        with open(available_path) as f:
            symbols = yaml.safe_load(f)['symbols']
    else:
        with open(settings.SYMBOLS_CONFIG) as f:
            cfg = yaml.safe_load(f)
        symbols = [s['name'] for s in cfg['symbols'] if s.get('tradable', True)]

    print(f"Symbols: {symbols}")

    monthly_path = 'data/backtest_monthly.csv'
    if not os.path.exists(monthly_path):
        print(f"ERROR: {monthly_path} not found. Run: python scripts/parse_backtest.py first.")
        return

    monthly_df = pd.read_csv(monthly_path, index_col='year_month')
    print(f"Loaded {len(monthly_df)} months | Sharpe range: {monthly_df['sharpe'].min():.2f} to {monthly_df['sharpe'].max():.2f}")

    storage = DataStorage()
    feature_cols = get_feature_columns()

    all_data = {}
    for sym in symbols:
        df = storage.load_bars(sym)
        if not df.empty:
            df = compute_features(df)
            all_data[sym] = df

    if not all_data:
        print("No market data. Run scripts/update_data.py first.")
        return

    X_list, y_list, month_labels = [], [], []

    for year_month, row in monthly_df.iterrows():
        sharpe = row['sharpe']
        target = sharpe_to_target(sharpe)

        year, month = int(year_month[:4]), int(year_month[5:7])
        month_start = datetime.datetime(year, month, 1)
        prior_end   = month_start
        prior_start = month_start - datetime.timedelta(days=60)

        feats_list = []
        for sym, df in all_data.items():
            mask = (df.index >= prior_start) & (df.index < prior_end)
            feats_list.append(extract_monthly_features(df[mask], feature_cols))

        if not feats_list:
            continue

        features = np.concatenate(feats_list)
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            continue

        X_list.append(features)
        y_list.append(target)
        month_labels.append(year_month)

    if len(X_list) < 5:
        print(f"Only {len(X_list)} training samples. Need more data.")
        return

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_scalar = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
    y = y_scalar.expand(-1, len(BASE_PARAMS))

    print(f"\nTraining samples: {len(X_list)}, input_dim: {X.shape[1]}, output_dim: {y.shape[1]}")

    model = RegimeModel(X.shape[1], y.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()

    for epoch in range(3000):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            with torch.no_grad():
                corr = float(np.corrcoef(pred[:,0].numpy(), y[:,0].numpy())[0,1])
            print(f"Epoch {epoch:4d} | loss: {loss.item():.6f} | corr: {corr:.3f}")

    torch.save({
        'model_state':  model.state_dict(),
        'input_dim':    X.shape[1],
        'output_dim':   y.shape[1],
        'base_params':  BASE_PARAMS,
        'scale_bounds': SCALE_BOUNDS,
        'param_names':  list(BASE_PARAMS.keys()),
        'symbols':      symbols,
        'feature_cols': feature_cols,
    }, settings.MODEL_SAVE_PATH)

    print(f"\nModel saved to {settings.MODEL_SAVE_PATH}")

    # Show what the model predicts for each month
    model.eval()
    with torch.no_grad():
        all_scales = model(X).numpy()

    print("\nPredicted parameters per month:")
    print(f"{'Month':<12} {'Sharpe':>8} {'mmLots':>8} {'SL1':>6} {'PT1':>6} {'SL2':>6} {'PT2':>6} {'TSA':>6}")
    print("-" * 64)
    param_names = list(BASE_PARAMS.keys())
    for i, (month, sharpe_row) in enumerate(zip(month_labels, monthly_df['sharpe'])):
        scales = all_scales[i]
        vals = []
        for j, name in enumerate(param_names):
            lo, hi = SCALE_BOUNDS[name]
            scale = lo + scales[j] * (hi - lo)
            vals.append(BASE_PARAMS[name] * scale)
        print(f"{month:<12} {sharpe_row:>8.1f} {vals[0]:>8.1f} {vals[1]:>6.2f} {vals[2]:>6.2f} {vals[3]:>6.2f} {vals[4]:>6.2f} {vals[5]:>6.2f}")

    # Save available symbols
    os.makedirs('config', exist_ok=True)
    with open('config/available_symbols.yaml', 'w') as f:
        yaml.dump({'symbols': symbols}, f)


if __name__ == '__main__':
    main()
