#swarm_ml/safety.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional
import numpy as np

@dataclass
class SafetyLimits:
    v_max: float = 2.0           # m/s
    a_max: float = 1.0           # m/s^2 (simple first-order cap)
    z_min: float = 0.5           # m
    z_max: float = 10.0          # m
    geofence_xy: Optional[Tuple[float,float,float,float]] = None  # (xmin,xmax,ymin,ymax)
    keepouts: Tuple[Tuple[np.ndarray, float], ...] = tuple()       # spheres: (center, radius)

def _cap_velocity(vec: np.ndarray, v_max: float) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec if n <= v_max or n == 0.0 else vec * (v_max / n)

def _push_from_keepouts(p_next: np.ndarray, mv: np.ndarray, keepouts) -> np.ndarray:
    if not keepouts: return mv
    grad = np.zeros(3, float)
    for c, r in keepouts:
        rel = (p_next) - c.reshape(3)
        d = float(np.linalg.norm(rel) + 1e-9)
        if d < r:
            grad += (rel / d) * (r - d)
    if np.linalg.norm(grad) > 0:
        mv = mv + 0.5 * grad
    return mv

def _clamp_altitude(p: np.ndarray, mv: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    z_next = p[2] + mv[2]
    if z_next < z_min: mv[2] = z_min - p[2]
    if z_next > z_max: mv[2] = z_max - p[2]
    return mv

def _clamp_geofence(p: np.ndarray, mv: np.ndarray, fence) -> np.ndarray:
    if fence is None: return mv
    xmin, xmax, ymin, ymax = fence
    x_next = p[0] + mv[0]; y_next = p[1] + mv[1]
    if x_next < xmin: mv[0] = xmin - p[0]
    if x_next > xmax: mv[0] = xmax - p[0]
    if y_next < ymin: mv[1] = ymin - p[1]
    if y_next > ymax: mv[1] = ymax - p[1]
    return mv

def project_to_safe(p_self: np.ndarray, move: np.ndarray,
                    neighbors: Dict[str, np.ndarray], d_min: float = 1.5,
                    limits: SafetyLimits = SafetyLimits()) -> np.ndarray:
    """Project a proposed displacement to respect min inter‑robot spacing and basic limits."""
    p_self = np.asarray(p_self, float).reshape(3)
    mv = np.asarray(move,   float).reshape(3)

    # Inter‑robot spacing barrier (repulsive correction)
    grad = np.zeros(3, float)
    for _, pn in neighbors.items():
        pn = np.asarray(pn, float).reshape(3)
        rel = (p_self + mv) - pn
        d = float(np.linalg.norm(rel) + 1e-9)
        if d < d_min:
            grad += (rel / d) * (d_min - d)
    if np.linalg.norm(grad) > 0:
        mv = mv + 0.5 * grad

    # Keep‑outs, geofence, altitude band
    mv = _push_from_keepouts(p_self + mv, mv, limits.keepouts)
    mv = _clamp_geofence(p_self, mv, limits.geofence_xy)
    mv = _clamp_altitude(p_self, mv, limits.z_min, limits.z_max)

    # Velocity cap (treat mv as Δx per planning tick; caller maps to velocity)
    mv = _cap_velocity(mv, limits.v_max / 5.0)  # conservative per‑tick displacement
    return mv

