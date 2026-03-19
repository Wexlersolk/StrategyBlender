# This script will be called periodically (e.g., every day) by a scheduler.
# It fetches latest data, adapts the model, and updates MT5.
import sys
sys.path.append('.')

import torch
import yaml
from data.fetcher import MT5Fetcher
from data.features import compute_features, get_feature_columns
from meta_learning.maml import create_maml_model
from meta_learning.adapt import extract_features_from_task
from execution.mt5_bridge import MT5Bridge
from execution.parameter_validator import validate_parameters
from execution.risk_manager import RiskManager
from monitoring.logger import AdaptationLogger
import config.settings as settings

def load_model(input_dim, output_dim, device='cpu'):
    maml = create_maml_model(input_dim, output_dim)
    maml.load_state_dict(torch.load(settings.MODEL_SAVE_PATH, map_location=device))
    maml.to(device)
    return maml

def main():
    # Load symbols
    with open(settings.SYMBOLS_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    symbols = [s['name'] for s in config['symbols'] if s.get('tradable', True)]
    
    # Initialize components
    fetcher = MT5Fetcher()
    bridge = MT5Bridge()
    logger = AdaptationLogger()
    risk_mgr = RiskManager(initial_capital=10000)  # need actual equity
    
    # Fetch recent data for all symbols (support window length)
    support_window = settings.SUPPORT_SIZE
    data = {}
    for sym in symbols:
        df = fetcher.fetch_live(sym, count=support_window + 5)  # extra for features
        if df.empty:
            print(f"No data for {sym}, skipping")
            continue
        df = compute_features(df)
        # Take last support_window days
        data[sym] = df.iloc[-support_window:]
    
    # Build task structure (only support part)
    task = {'support': data, 'query': {}}  # query empty because we're not computing loss
    
    # Prepare input
    feature_cols = get_feature_columns()
    input_dim = len(feature_cols) * len(symbols)
    output_dim = len(settings.PARAMETER_BOUNDS)
    model = load_model(input_dim, output_dim)
    
    # Adapt (inner loop)
    learner = model.clone()
    # We need a loss to adapt; in live mode we could either:
    # a) Use a surrogate loss (e.g., maximize recent performance) OR
    # b) Skip adaptation and just use meta-initialization (no inner loop).
    # Here we'll skip adaptation for simplicity and just use the meta-initialized output.
    # To adapt, you'd need a loss function based on recent data (but we have no query period).
    # You could simulate on the support set itself (though that may overfit).
    
    # Generate parameters
    x = extract_features_from_task(task, mode='support')  # average features across symbols
    x = x.unsqueeze(0)  # add batch dimension
    raw_params = model(x).squeeze(0).detach().numpy()
    
    # Build parameter dict for each symbol (here we use same params for all symbols)
    # In reality you might want symbol-specific outputs; modify model output accordingly.
    param_names = list(settings.PARAMETER_BOUNDS.keys())
    params_dict = {name: raw_params[i] for i, name in enumerate(param_names)}
    validated = validate_parameters(params_dict)
    
    # Risk check
    # In a real system you'd get current equity from MT5
    current_equity = 15000  # dummy
    risk_mgr.update_equity(current_equity)
    if risk_mgr.check_drawdown():
        validated = risk_mgr.reduce_risk(validated)
        logger.log_adaptation("portfolio", {}, validated, reason="drawdown limit")
    
    # Send to MT5 (for each symbol)
    for sym in symbols:
        bridge.send_parameters_to_ea(sym, validated)
        logger.log_adaptation(sym, {}, validated, reason="daily update")
    
    print("Live adaptation completed.")
    fetcher.shutdown()
    bridge.shutdown()

if __name__ == "__main__":
    main()
