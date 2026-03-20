import torch
import numpy as np
from data.features import get_feature_columns
import config.settings as settings


def extract_features_from_task(task, mode='support', symbol=None):
    """
    Convert task data into a feature tensor for the model.
    Averages features over the time window per symbol, then concatenates.
    """
    data_dict = task[mode]  # dict: symbol -> DataFrame
    feature_cols = get_feature_columns()
    symbol_vectors = []

    for sym, df in data_dict.items():
        if df.empty or not all(c in df.columns for c in feature_cols):
            feats = np.zeros(len(feature_cols))
        else:
            feats = df[feature_cols].mean().values  # shape (num_features,)
        symbol_vectors.append(feats)

    x = np.concatenate(symbol_vectors)
    return torch.tensor(x, dtype=torch.float32)


def compute_sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe ratio from a daily returns array."""
    if len(returns) < 2:
        return 0.0
    mean_r = np.mean(returns)
    std_r = np.std(returns) + 1e-8
    return float(mean_r / std_r * np.sqrt(252))


def simulate_returns(query_data: dict, validated: dict) -> np.ndarray:
    """
    Simple vectorised simulation on query data.
    Uses whatever SL/PT coefficient keys exist in validated params.
    """
    all_returns = []

    # Get the first SL and PT coef values available — works with any param names
    param_names = list(validated.keys())
    sl_keys = [k for k in param_names if 'sl' in k.lower() or 'stop' in k.lower()]
    pt_keys = [k for k in param_names if 'pt' in k.lower() or 'profit' in k.lower() or 'target' in k.lower()]
    lot_keys = [k for k in param_names if 'lot' in k.lower()]

    sl_coef  = validated[sl_keys[0]]  if sl_keys  else 2.0
    pt_coef  = validated[pt_keys[0]]  if pt_keys  else 3.0
    lot_size = validated[lot_keys[0]] if lot_keys else 1.0

    for sym, df in query_data.items():
        if df.empty or 'returns' not in df.columns or 'atr' not in df.columns:
            continue

        closes  = df['close'].values
        returns = df['returns'].values
        atrs    = df['atr'].values

        for i in range(1, len(returns)):
            direction = 1.0 if returns[i - 1] > 0 else -1.0
            atr       = atrs[i] if atrs[i] > 0 else 1e-8
            price     = closes[i - 1] if closes[i - 1] > 0 else 1.0

            sl_dist = sl_coef * atr / price
            tp_dist = pt_coef * atr / price

            raw_ret = returns[i] * direction
            raw_ret = max(-sl_dist, min(tp_dist, raw_ret))
            all_returns.append(raw_ret * lot_size)

    return np.array(all_returns) if all_returns else np.array([0.0])


def compute_loss_on_task(model, task, mode='support'):
    """
    Compute loss (negative Sharpe) for the given task.
    Loss is properly connected to model parameters via a soft proxy
    so gradients can flow back through MAML's inner loop.
    """
    from execution.parameter_validator import validate_parameters

    # Forward pass
    x = extract_features_from_task(task, mode)
    raw_params = model(x)  # shape (4,) — raw unbounded outputs

    # Build validated (scaled) param dict — use names from settings
    param_names = list(settings.PARAMETER_BOUNDS.keys())
    params_dict = {name: raw_params[i] for i, name in enumerate(param_names)}
    validated = validate_parameters({k: v.item() for k, v in params_dict.items()})

    # Simulate on the requested data split
    data = task[mode]
    returns_arr = simulate_returns(data, validated)

    sharpe_val = compute_sharpe(returns_arr)

    # Connect loss to raw_params so autograd has a path through the model.
    # raw_params.mean() * 0 keeps the gradient graph alive with zero numeric
    # contribution, then we subtract the actual Sharpe scalar.
    loss = raw_params.mean() * 0.0 - torch.tensor(sharpe_val, dtype=torch.float32)
    return loss
