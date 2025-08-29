def scale_for_nees(nees_avg: float, target: float = 2.0, gain: float = 0.25) -> float:
    """
    Return multiplicative scale for measurement std to move NEES toward 'target'.
    Example: new_std = old_std * scale_for_nees(last_nees)
    """
    nees_avg = max(1e-6, float(nees_avg))
    err = (nees_avg / target) - 1.0
    scale = 1.0 + gain * err
    return float(max(0.6, min(1.6, scale)))

