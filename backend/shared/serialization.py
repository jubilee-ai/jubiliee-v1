from agents.training.core.state import TrainingAgentState


def serialize_state(state: TrainingAgentState | dict[str, object]) -> dict:
    """Convert training state to a JSON-serializable dict."""
    import math
    import numpy as np

    def make_serializable(obj: object):
        if obj is None:
            return None
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return [make_serializable(x) for x in obj.tolist()]
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, (str, int, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        return str(obj)

    if isinstance(state, dict):
        return make_serializable(state)
    return make_serializable(dict(state))
