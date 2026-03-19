import torch
import numpy as np
from data.features import get_feature_columns

def extract_features_from_task(task, mode='support', symbol=None):
    """
    Convert task data into feature tensor for the model.
    For simplicity, we average features over the support window per symbol,
    then concatenate across symbols.
    This is a placeholder; you may design more sophisticated input representations.
    """
    data_dict = task[mode]  # dict symbol -> DataFrame
    feature_cols = get_feature_columns()
    symbol_vectors = []
    for sym, df in data_dict.items():
        # Average features over time window
        feats = df[feature_cols].mean().values  # shape (num_features,)
        symbol_vectors.append(feats)
    # Concatenate all symbols
    x = np.concatenate(symbol_vectors)
    return torch.tensor(x, dtype=torch.float32)

def compute_loss_on_task(model, task, mode='support'):
    """
    Compute loss (negative Sharpe) for the given task after obtaining parameters from model.
    This requires a simulator that, given parameters, returns a return series for the query period.
    Here we stub it out.
    """
    # Get input features for this task
    x = extract_features_from_task(task, mode)
    params = model(x)  # raw network output
    
    # Clip / scale parameters to valid ranges (using a validator)
    from execution.parameter_validator import validate_parameters
    params_dict = {
        'lot_size': params[0].item(),
        'sl_atr': params[1].item(),
        'tp_atr': params[2].item(),
        'rsi_period': params[3].item()
    }
    validated = validate_parameters(params_dict)
    
    # Simulate strategy on query data and compute negative Sharpe
    # This is a placeholder: you need to implement a backtester for your specific strategy.
    # For now, return a dummy loss.
    query_data = task['query']  # dict symbol->DataFrame
    # ... run simulation using validated parameters ...
    sharpe = np.random.randn()  # replace with actual Sharpe
    loss = -torch.tensor(sharpe, requires_grad=True)
    return loss
