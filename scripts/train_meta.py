"""
scripts/train_meta.py

Trains the RegimeModel using ALL available data sources:

  1. Backtest HTML reports   (monthly Sharpe per parameter set)  ← original
  2. Optimizer XML results   (overall Sharpe per parameter set)  ← NEW

With optimizer data the training set grows from ~30 to ~1000+ samples:
    optimizer_rows × months_in_market_data = training samples
    18 rows × 70 months = 1260 samples

The model learns:
    market_features (prior 60 days) → lot_size_scale (0.0 to 1.0)
"""

import sys
import os
import datetime
import json
sys.path.append('.')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import config.settings as settings
from data.storage import DataStorage
from data.features import compute_features, get_feature_columns


# ── Model ─────────────────────────────────────────────────────────────────────

class RegimeModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.Tanh(),
            nn.Linear(32, 16),        nn.Tanh(),
            nn.Linear(16, output_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


# ── Config ────────────────────────────────────────────────────────────────────

BASE_PARAMS = {
    'mmLots':            70.0,
    'StopLossCoef1':      2.0,
    'ProfitTargetCoef1':  1.5,
    'StopLossCoef2':      2.4,
    'ProfitTargetCoef2':  1.5,
    'TrailingActCef1':    1.4,
}

SCALE_BOUNDS = {
    'mmLots':            (0.2, 1.0),
    'StopLossCoef1':     (1.0, 1.0),
    'ProfitTargetCoef1': (1.0, 1.0),
    'StopLossCoef2':     (1.0, 1.0),
    'ProfitTargetCoef2': (1.0, 1.0),
    'TrailingActCef1':   (1.0, 1.0),
}

EPOCHS   = 3000
LR       = 0.005
LOOKBACK = 60   # days of market data before each month


# ── Sharpe → scale ────────────────────────────────────────────────────────────

def sharpe_to_scale(sharpe: float, sharpe_min: float, sharpe_max: float) -> float:
    """
    Map a Sharpe value to a 0-1 scale relative to the observed range.
    Uses min-max normalisation so the model sees the full 0-1 range.
    """
    if sharpe_max == sharpe_min:
        return 0.5
    return float(np.clip((sharpe - sharpe_min) / (sharpe_max - sharpe_min), 0.0, 1.0))


# ── Data source 1: HTML backtest reports ──────────────────────────────────────

def load_monthly_from_reports(reports_path: str = 'data/exports/backtest_monthly.csv') -> pd.DataFrame:
    if not os.path.exists(reports_path):
        return pd.DataFrame()
    df = pd.read_csv(reports_path)
    print(f"  HTML reports: {len(df)} monthly rows from {reports_path}")
    return df


# ── Data source 2: Optimizer XML results ─────────────────────────────────────

def load_optimizer_results(opt_path: str = 'data/exports/optimizer_results.csv') -> pd.DataFrame:
    if not os.path.exists(opt_path):
        # Also try root level XML files
        xmls = list(__import__('pathlib').Path('.').glob('*.xml'))
        if xmls:
            print(f"  Found XML: {xmls[0]} — run scripts/parse_optimizer.py first")
        return pd.DataFrame()
    df = pd.read_csv(opt_path)
    print(f"  Optimizer results: {len(df)} parameter sets from {opt_path}")
    return df


# ── Market features ───────────────────────────────────────────────────────────

def load_market_data(symbols: list) -> dict:
    storage      = DataStorage()
    feature_cols = get_feature_columns()
    market_data  = {}

    for sym in symbols:
        df = storage.load_bars(sym)
        if not df.empty:
            df = compute_features(df)
            market_data[sym] = df
            print(f"  Market data: {len(df)} bars for {sym}")
        else:
            print(f"  WARNING: No data for {sym}")

    return market_data


def get_monthly_features(market_data: dict, year: int, month: int,
                          feature_cols: list, lookback_days: int = 60) -> np.ndarray | None:
    """
    Extract normalised feature vector for the 60 days before month start.
    Returns None if insufficient data.
    """
    month_dt    = datetime.datetime(year, month, 1)
    prior_start = month_dt - datetime.timedelta(days=lookback_days)

    feats_list = []
    for sym, df in market_data.items():
        mask = (df.index >= prior_start) & (df.index < month_dt)
        sl   = df[mask]
        if sl.empty:
            feats_list.append(np.zeros(len(feature_cols)))
        else:
            f   = sl[feature_cols].mean().values.astype(np.float32)
            std = f.std()
            f   = (f - f.mean()) / std if std > 1e-8 else f
            feats_list.append(f)

    if not feats_list:
        return None

    combined = np.concatenate(feats_list)
    if not np.all(np.isfinite(combined)):
        combined = np.nan_to_num(combined, nan=0.0)

    return combined


# ── Build training set ────────────────────────────────────────────────────────

def build_training_data(market_data: dict, feature_cols: list) -> tuple:
    """
    Build X (features) and y (targets) from all available data sources.

    Source 1 — HTML reports (monthly Sharpe per row):
        Each row is one month. Direct mapping: features → sharpe_scale

    Source 2 — Optimizer results (overall Sharpe per parameter set):
        For each parameter set, generate one training sample per available
        month in the market data. All months get the same target (the
        parameter set's overall Sharpe normalised).
        This dramatically increases training set size.
    """
    X_list, y_list, sources = [], [], []

    # ── Source 1: HTML reports ────────────────────────────────────────────────
    monthly_df = load_monthly_from_reports()
    if not monthly_df.empty and 'sharpe' in monthly_df.columns:
        sh_min = monthly_df['sharpe'].min()
        sh_max = monthly_df['sharpe'].max()

        for _, row in monthly_df.iterrows():
            ym = str(row.get('year_month', ''))
            if len(ym) < 7:
                continue
            try:
                year, month = int(ym[:4]), int(ym[5:7])
            except ValueError:
                continue

            feats = get_monthly_features(market_data, year, month, feature_cols)
            if feats is None:
                continue

            target = sharpe_to_scale(row['sharpe'], sh_min, sh_max)
            X_list.append(feats)
            y_list.append(target)
            sources.append('html')

        print(f"  Training samples from HTML reports: {sum(1 for s in sources if s == 'html')}")

    # ── Source 2: Optimizer results ───────────────────────────────────────────
    opt_df = load_optimizer_results()
    if not opt_df.empty and 'Sharpe_Ratio' in opt_df.columns:
        sh_min_opt = opt_df['Sharpe_Ratio'].min()
        sh_max_opt = opt_df['Sharpe_Ratio'].max()

        # Get all months available in market data
        all_months = set()
        for df in market_data.values():
            for dt in df.index:
                all_months.add((dt.year, dt.month))
        all_months = sorted(all_months)

        opt_samples = 0
        for _, opt_row in opt_df.iterrows():
            sharpe = opt_row['Sharpe_Ratio']
            target = sharpe_to_scale(sharpe, sh_min_opt, sh_max_opt)

            for year, month in all_months:
                feats = get_monthly_features(market_data, year, month, feature_cols)
                if feats is None:
                    continue
                X_list.append(feats)
                y_list.append(target)
                sources.append('optimizer')
                opt_samples += 1

        print(f"  Training samples from optimizer: {opt_samples}")

    total = len(X_list)
    print(f"  Total training samples: {total}")

    if total < 5:
        raise ValueError(
            f"Only {total} training samples — need at least 5.\n"
            "Run update_data.py to fetch market data, then try again."
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, sources


# ── Training ──────────────────────────────────────────────────────────────────

def train(epochs: int = EPOCHS, lr: float = LR):
    print("\n=== StrategyBlender — Training RegimeModel ===\n")

    # Load available symbols from config
    try:
        import yaml
        with open(settings.SYMBOLS_CONFIG) as f:
            cfg = yaml.safe_load(f)
        symbols = [s['name'] for s in cfg.get('symbols', []) if s.get('tradable')]
        if not symbols:
            symbols = [s['name'] for s in cfg.get('symbols', [])]
    except Exception:
        symbols = ['HK50.cash']
    print(f"Symbols: {symbols}")

    feature_cols = get_feature_columns()
    print(f"Features: {len(feature_cols)} indicators")

    print("\nLoading data...")
    market_data = load_market_data(symbols)

    if not market_data:
        print("ERROR: No market data. Run: python scripts/update_data.py")
        return

    print("\nBuilding training set...")
    X, y, sources = build_training_data(market_data, feature_cols)

    # Expand y to match output_dim (one output per parameter)
    output_dim = len(SCALE_BOUNDS)
    Y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).expand(-1, output_dim)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    print(f"\nModel: {X.shape[1]} inputs → [32, 16] → {output_dim} outputs")
    print(f"Training: {epochs} epochs, lr={lr}\n")

    model     = RegimeModel(X.shape[1], output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    best_loss  = float('inf')
    best_state = None

    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            with torch.no_grad():
                corr = float(np.corrcoef(
                    pred.numpy()[:, 0], Y[:, 0].numpy()
                )[0, 1])
            print(f"Epoch {epoch:5d} | loss: {loss.item():.6f} | corr: {corr:.3f} "
                  f"| samples: {len(X)}")
            if loss.item() < best_loss:
                best_loss  = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # Save checkpoint
    checkpoint = {
        'model_state':  model.state_dict(),
        'input_dim':    X.shape[1],
        'output_dim':   output_dim,
        'base_params':  BASE_PARAMS,
        'scale_bounds': SCALE_BOUNDS,
        'param_names':  list(SCALE_BOUNDS.keys()),
        'symbols':      symbols,
        'feature_cols': feature_cols,
        'n_samples':    len(X),
        'sources':      {s: sources.count(s) for s in set(sources)},
    }
    torch.save(checkpoint, settings.MODEL_SAVE_PATH)

    print(f"\n✓ Model saved: {settings.MODEL_SAVE_PATH}")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Samples used: html={sources.count('html')}, "
          f"optimizer={sources.count('optimizer')}")

    # Write available symbols for generate_schedule.py
    avail_path = 'config/available_symbols.yaml'
    os.makedirs('config', exist_ok=True)
    with open(avail_path, 'w') as f:
        import yaml
        yaml.dump(symbols, f)
    print(f"  Symbols written: {avail_path}")

    return model


if __name__ == '__main__':
    train()
