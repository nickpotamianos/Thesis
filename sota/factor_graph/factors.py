"""
Light‑weight single‑range factors for the stock PyPI **gtsam** wheel.

Two factor types are offered (all 1‑D residuals):

  • Pose ↔ Anchor  – between a `Pose3` variable and a fixed `Point3`
  • Pose ↔ Pose    – inter‑robot tag‑to‑tag range

The implementation is pure‑Python and relies on *Jacobian callbacks* so it
works even with the minimalist wheels that ship on PyPI (no custom C++ build
needed).  Those wheels expose only the 3‑argument `CustomFactor` constructor
and hand each callback an *empty* NumPy view if they expect the user to fill
in an analytic Jacobian.  The helper below now grows that zero‑length view to
a 1×6 row **in‑place** – without touching the underlying C++ memory – so the
result is accepted by GTSAM’s internal `JacobianFactor` builder.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import gtsam
from gtsam import noiseModel
from sota.uwb.infer_uwb import UwbML

# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def _to_xyz(obj) -> np.ndarray:  # noqa: D401
    """Return an `(x, y, z)` array from either Point3 or array‑like."""
    return (
        np.array([obj.x(), obj.y(), obj.z()]) if hasattr(obj, "x") else np.asarray(obj, float).reshape(3)
    )


def _fill_jac(jac, unit_vec: np.ndarray, sign: float) -> None:  # noqa: D401
    """Write a *single* Jacobian row `[0,0,0, ±ux, ±uy, ±uz]`.

    We *must* write something with exactly **one row** (residual dimension) so
    the subsequent `JacobianFactor` construction succeeds.  Wheels older than
    Sept‑2023 pass a **zero‑length** 1‑D array – we grow it *in‑place* to a 6‑D
    row vector which keeps the Python wrapper in sync with the underlying C++
    buffer.
    """
    if jac is None:
        return

    # Older PyPI wheels hand out an empty 1‑D view → grow to length 6.
    if jac.size == 0:
        # `refcheck=False` is safe here – GTSAM owns the memory uniquely.
        jac.resize(6, refcheck=False)  # shape becomes (6,) → 1 row, 6 cols

    # Zero‑initialise then fill the translation part (columns 3‑5).
    jac[:] = 0.0
    if jac.ndim == 1:  # (6,)
        jac[3:6] = sign * unit_vec
    elif jac.ndim == 2:  # (1,6) or anything wider
        jac[0, 3:6] = sign * unit_vec


# ---------------------------------------------------------------------------
# custom factors
# ---------------------------------------------------------------------------

class _PosePose(gtsam.CustomFactor):  # type: ignore[misc]
    """Range factor (1‑D residual) between two `Pose3` variables."""

    def __init__(self, k_i: int, k_j: int, measured_range: float, model):
        self.k_i, self.k_j, self.rng = int(k_i), int(k_j), float(measured_range)

        def err(_self, vals: gtsam.Values, jacobians=None):  # noqa: ANN001
            pi = _to_xyz(vals.atPose3(self.k_i).translation())
            pj = _to_xyz(vals.atPose3(self.k_j).translation())
            diff = pj - pi
            dist = np.linalg.norm(diff)
            res = np.array([dist - self.rng])
            if jacobians is not None:
                unit = diff / dist if dist > 1e-9 else np.zeros(3)
                _fill_jac(jacobians[0], unit, -1.0)
                _fill_jac(jacobians[1], unit, +1.0)
            return res

        # 3‑argument constructor keeps us compatible with the stock wheel.
        super().__init__(model, [self.k_i, self.k_j], err)


class _PoseAnchor(gtsam.CustomFactor):  # type: ignore[misc]
    """Range factor (1‑D residual) between a `Pose3` and a fixed anchor."""

    def __init__(self, k_pose: int, anchor_xyz: np.ndarray, measured_range: float, model):
        self.k_pose = int(k_pose)
        self.anchor = np.asarray(anchor_xyz, float).reshape(3)
        self.rng = float(measured_range)

        def err(_self, vals: gtsam.Values, jacobians=None):  # noqa: ANN001
            p = _to_xyz(vals.atPose3(self.k_pose).translation())
            diff = p - self.anchor
            dist = np.linalg.norm(diff)
            res = np.array([dist - self.rng])
            if jacobians is not None:
                unit = diff / dist if dist > 1e-9 else np.zeros(3)
                _fill_jac(jacobians[0], unit, +1.0)
            return res

        super().__init__(model, [self.k_pose], err)


# ---------------------------------------------------------------------------
# factory helper
# ---------------------------------------------------------------------------

class RangeFactory:  # noqa: D101
    """Applies learned NLoS correction then spawns the right factor type."""

    def __init__(self, uwb_ml: UwbML):
        self.ml = uwb_ml

    # ------------------------------------------------------------------
    def make(
        self,
        key_i: int,
        key_j_or_point: Union[int, Sequence[float], np.ndarray],
        cir_blob,
        raw_range: float,
    ):
        """Return a ready‑to‑use GTSAM factor.

        Parameters
        ----------
        key_i
            Key of the *transmitting* robot pose.
        key_j_or_point
            Either a second `Pose3` key (robot‑to‑robot) or a 3‑vector anchor
            position.
        cir_blob
            Raw CIR blob that the CNN frontend turns into a path‑bias estimate.
        raw_range
            Direct UWB range measurement in **metres**.
        """
        rng_corr, sigma = self.ml.correct(cir_blob, raw_range)
        model = noiseModel.Isotropic.Sigma(1, sigma)

        # Pose ↔ Anchor
        if isinstance(key_j_or_point, (np.ndarray, list, tuple)):
            anchor = np.asarray(key_j_or_point, float).reshape(3)
            return _PoseAnchor(key_i, anchor, rng_corr, model)

        # Pose ↔ Pose
        return _PosePose(key_i, int(key_j_or_point), rng_corr, model)


# ---------------------------------------------------------------------------
# sanity check (run `python factors.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Basic regression test – should run without raising.
    model = noiseModel.Isotropic.Sigma(1, 0.1)
    pose_key_a = 1
    pose_key_b = 2
    anchor = np.array([1.0, 2.0, 3.0])

    # Dummy UwbML that returns raw inputs unchanged.
    class _NoML:  # noqa: D401
        def correct(self, _cir, r):  # noqa: D401
            return r, 0.1

    print("Running self‑test …", end=" ")
    fac1 = _PosePose(pose_key_a, pose_key_b, 5.0, model)
    fac2 = _PoseAnchor(pose_key_a, anchor, 2.5, model)
    RangeFactory(_NoML()).make(pose_key_a, anchor, None, 2.5)
    print("✓ OK")
