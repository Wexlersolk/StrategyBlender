import sys
sys.path.append('.')

import torch
import yaml
from mt5linux import MetaTrader5

from data.fetcher import MT5Fetcher
from data.features import compute_features, get_feature_columns
from meta_learning.maml import create_maml_model
from meta_learning.adapt import extract_features_from_task
from execution.mt5_bridge import MT5Bridge
from execution.parameter_validator import validate_parameters
from execution.risk_manager import RiskManager
from monitoring.logger import AdaptationLogger
import config.settings as settings

# Shared mt5linux instance — bridge server must be running on port 18812
_mt5 = MetaTrader5(host='localhost', port=18812)


def load_model(input_dim, output_dim, device='cpu'):
    maml = create_maml_model(input_dim, output_dim)
    maml.load_state_dict(torch.load(settings.MODEL_SAVE_PATH, map_location=device))
    maml.to(device)
    return maml


def get_current_equity(fallback=10000.0):
    """Fetch real account equity from MT5. Falls back to a safe default on failure."""
    try:
        account_info = _mt5.account_info()
        if account_info is not None:
            return account_info.equity
    except Exception as e:
        print(f"Warning: could not fetch account equity: {e}")
    print(f"Warning: using fallback equity of {fallback}")
    return fallback


def main():
    # Load symbols
    with open(settings.SYMBOLS_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    symbols = [s['name'] for s in config['symbols'] if s.get('tradable', True)]

    # Initialize components
    fetcher = MT5Fetcher()
    bridge = MT5Bridge()
    logger = AdaptationLogger()

    # Get real equity from MT5 before constructing RiskManager
    current_equity = get_current_equity(fallback=10000.0)
    risk_mgr = RiskManager(initial_capital=current_equity)

    # Fetch recent data for all symbols (support window length)
    support_window = settings.SUPPORT_SIZE
    data = {}
    for sym in symbols:
        df = fetcher.fetch_live(sym, count=support_window + 5)  # extra rows for feature warmup
        if df.empty:
            print(f"No data for {sym}, skipping")
            continue
        df = compute_features(df)
        data[sym] = df.iloc[-support_window:]

    if not data:
        print("No data fetched for any symbol. Aborting.")
        fetcher.shutdown()
        bridge.shutdown()
        return

    # Build task structure (support only — no query needed in live mode)
    task = {'support': data, 'query': {}}

    # Prepare model
    feature_cols = get_feature_columns()
    input_dim = len(feature_cols) * len(symbols)
    output_dim = len(settings.PARAMETER_BOUNDS)
    model = load_model(input_dim, output_dim)
    model.eval()

    # Generate parameters from meta-initialised model (no inner-loop adaptation in live mode)
    x = extract_features_from_task(task, mode='support')
    x = x.unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        raw_params = model(x).squeeze(0).numpy()

    # Build and validate parameter dict
    param_names = list(settings.PARAMETER_BOUNDS.keys())
    params_dict = {name: raw_params[i] for i, name in enumerate(param_names)}
    validated = validate_parameters(params_dict)

    # Risk check using real equity
    risk_mgr.update_equity(current_equity)
    if risk_mgr.check_drawdown():
        validated = risk_mgr.reduce_risk(validated)
        logger.log_adaptation(
            "portfolio", {}, validated, reason="drawdown limit triggered"
        )

    # Send parameters to MT5 EA for each symbol
    for sym in symbols:
        bridge.send_parameters_to_ea(sym, validated)
        logger.log_adaptation(sym, {}, validated, reason="daily update")

    print("Live adaptation completed.")
    print("Parameters applied:", validated)

    fetcher.shutdown()
    bridge.shutdown()


if __name__ == "__main__":
    main()
