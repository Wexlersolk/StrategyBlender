import learn2learn as l2l
import torch
import torch.nn as nn
import torch.optim as optim
from meta_learning.model import ParameterGenerator

def create_maml_model(input_dim, output_dim, inner_lr=0.01):
    """
    Create a MAML-wrapped model.
    """
    model = ParameterGenerator(input_dim, output_dim)
    maml = l2l.algorithms.MAML(model, lr=inner_lr, first_order=False)
    return maml
