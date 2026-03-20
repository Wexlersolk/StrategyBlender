import torch
import torch.nn as nn

class ParameterGenerator(nn.Module):
    """
    Neural network that takes market features (over support window) and outputs
    trading parameters for a single symbol.
    Input: flattened feature matrix of shape (support_size * num_features)
    Output: vector of parameter values (e.g., lot_size, sl_atr, tp_atr, rsi_period)
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        # Optional: apply sigmoid to bound outputs (will be scaled later)
    
    def forward(self, x):
        return self.net(x)
