import math
import config.settings as settings


def _stable_sigmoid(x):
    """
    Numerically stable sigmoid — avoids math overflow for large |x|.
    For x >= 0: 1 / (1 + exp(-x))
    For x <  0: exp(x) / (1 + exp(x))
    Either way the result is clamped safely to (0, 1).
    """
    # Hard clamp to prevent exp overflow before we even try
    x = max(-500.0, min(500.0, x))
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def validate_parameters(params):
    """
    Clip parameters to allowed ranges and ensure they are within bounds.
    Applies sigmoid to squash raw network output (unbounded reals) into (0, 1),
    then scales to [low, high] for each parameter.

    params: dict of {param_name: raw_float_value}
    returns: dict of {param_name: scaled_float_value}
    """
    bounds = settings.PARAMETER_BOUNDS
    validated = {}

    for key, (low, high) in bounds.items():
        val = params.get(key, 0.0)

        # Handle both plain floats and torch tensors
        if hasattr(val, 'item'):
            val = val.item()

        # Numerically stable sigmoid squashes to (0, 1)
        val_01 = _stable_sigmoid(val)

        # Scale to [low, high]
        scaled = low + val_01 * (high - low)
        validated[key] = scaled

    return validated
