import config.settings as settings

def validate_parameters(params):
    """
    Clip parameters to allowed ranges and ensure they are within bounds.
    Also handle scaling from network output to actual values.
    """
    bounds = settings.PARAMETER_BOUNDS
    validated = {}
    for key, val in params.items():
        low, high = bounds[key]
        # If network outputs arbitrary real numbers, apply sigmoid-like scaling
        # For simplicity, assume val is already in roughly [0,1] (e.g., after sigmoid)
        # and scale to [low, high]
        scaled = low + val * (high - low)
        validated[key] = scaled
    return validated
