import math

import numpy as np


def make_json_serializable(data):
    """Recursively convert numpy/NaN values for JSON output."""
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [make_json_serializable(v) for v in data]
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None
    if isinstance(data, np.generic):
        return data.item()
    return data
