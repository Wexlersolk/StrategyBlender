import torch
import torch.nn as nn
import copy
from meta_learning.model import ParameterGenerator


class MAML:
    """
    Pure Python MAML implementation — no learn2learn dependency.
    Wraps a ParameterGenerator and provides clone() and adapt() methods
    that mirror the learn2learn API so the rest of the codebase is unchanged.
    """

    def __init__(self, model, lr=0.01, first_order=False):
        self.model = model
        self.lr = lr
        self.first_order = first_order

    def __call__(self, x):
        return self.model(x)

    def parameters(self):
        return self.model.parameters()

    def to(self, device):
        self.model.to(device)
        return self

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def state_dict(self):
        return self.model.state_dict()

    def eval(self):
        self.model.eval()
        return self

    def train(self):
        self.model.train()
        return self

    def clone(self):
        """Return a deep copy of this MAML wrapper with cloned model weights."""
        new_model = copy.deepcopy(self.model)
        new_maml = MAML(new_model, lr=self.lr, first_order=self.first_order)
        return new_maml

    def adapt(self, loss):
        """
        Perform one step of inner-loop gradient update on the cloned model.
        Modifies this instance's model parameters in-place.
        """
        grads = torch.autograd.grad(
            loss,
            self.model.parameters(),
            create_graph=not self.first_order,
            allow_unused=True
        )
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), grads):
                if grad is not None:
                    param -= self.lr * grad


def create_maml_model(input_dim, output_dim, inner_lr=0.01):
    """
    Create a MAML-wrapped ParameterGenerator.
    Drop-in replacement for the learn2learn version.
    """
    model = ParameterGenerator(input_dim, output_dim)
    maml = MAML(model, lr=inner_lr, first_order=False)
    return maml
