"""
Light‑weight single‑range factors for the PyPI GTSAM wheel
with robust handling of the Jacobian callback.

  • Pose ↔ Anchor  →  1‑D factor
  • Pose ↔ Pose    →  1‑D factor
"""

from typing import Sequence, Union

import gtsam
import numpy as np
from gtsam import noiseModel
from sota.uwb.infer_uwb import UwbML


# --------------------------------------------------------------------------- #
#                               utilities                                     #
# --------------------------------------------------------------------------- #
def _to_xyz(obj) -> np.ndarray:
    """Convert either a gtsam.Point3 or an array‑like to shape (3,)."""
    return (
        np.array([obj.x(), obj.y(), obj.z()])
        if hasattr(obj, "x")
        else np.asarray(obj, dtype=float).reshape(3)
    )


def _fill_jac(jac, unit_vec, sign: float):
    """
    Populate a 1×6 Jacobian block [ 0 0 0 | ±ux ±uy ±uz ]   (rot part = 0).

    If GTSAM passes a zero‑sized placeholder we leave it untouched so that
    it will use numerical differentiation instead.
    """
    if jac is None or jac.size == 0:          # placeholder → let GTSAM handle
        return

    if jac.ndim == 1 and jac.size >= 6:       # shape (6,)
        jac[:] = 0.0
        jac[3:6] = sign * unit_vec
    elif jac.shape[0] == 1 and jac.shape[1] >= 6:   # shape (1,6)
        jac[:] = 0.0
        jac[0, 3:6] = sign * unit_vec
    # else: unexpected layout – silently ignore


# --------------------------------------------------------------------------- #
#                              custom factors                                 #
# --------------------------------------------------------------------------- #
class _PosePose(gtsam.CustomFactor):
    """Range factor between two Pose3 variables."""

    def __init__(self, k_i: int, k_j: int,
                 measured_range: float,
                 model: gtsam.noiseModel.Base):

        self.k_i, self.k_j, self.rng = int(k_i), int(k_j), float(measured_range)

        def err(_self, vals: gtsam.Values, jacobians=None):
            pi = _to_xyz(vals.atPose3(self.k_i).translation())
            pj = _to_xyz(vals.atPose3(self.k_j).translation())
            diff = pj - pi
            dist = np.linalg.norm(diff)
            res  = np.array([dist - self.rng])

            if jacobians is not None:
                unit = diff / dist if dist > 1e-9 else np.zeros(3)
                _fill_jac(jacobians[0], unit, -1.0)   # wrt pose_i
                _fill_jac(jacobians[1], unit,  1.0)   # wrt pose_j
            return res

        super().__init__(model, [self.k_i, self.k_j], err)


class _PoseAnchor(gtsam.CustomFactor):
    """Range factor between a Pose3 variable and a fixed Point3 anchor."""

    def __init__(self, k_pose: int, anchor_xyz: np.ndarray,
                 measured_range: float,
                 model: gtsam.noiseModel.Base):

        self.k_pose = int(k_pose)
        self.anchor = np.asarray(anchor_xyz, dtype=float).reshape(3)
        self.rng    = float(measured_range)

        def err(_self, vals: gtsam.Values, jacobians=None):
            p    = _to_xyz(vals.atPose3(self.k_pose).translation())
            diff = p - self.anchor
            dist = np.linalg.norm(diff)
            res  = np.array([dist - self.rng])

            if jacobians is not None:
                unit = diff / dist if dist > 1e-9 else np.zeros(3)
                _fill_jac(jacobians[0], unit, 1.0)    # wrt pose
            return res

        super().__init__(model, [self.k_pose], err)


# --------------------------------------------------------------------------- #
#                           factory convenience                               #
# --------------------------------------------------------------------------- #
class RangeFactory:
    """
    Creates the appropriate factor (anchor or tag‑to‑tag) after applying the
    learned NLoS correction from sota.uwb.infer_uwb.UwbML.
    """

    def __init__(self, uwb_ml: UwbML):
        self.ml = uwb_ml

    # --------------------------------------------------------------------- #
    def make(self,
             key_i: int,
             key_j_or_point: Union[int, Sequence[float], np.ndarray],
             cir_blob,
             raw_range: float) -> gtsam.NoiseModelFactor:
        """
        Parameters
        ----------
        key_i : int
            Pose key of the transmitting robot.
        key_j_or_point : int | array‑like(3,)
            Pose key of the receiving robot *or* fixed anchor XYZ.
        cir_blob : any
            Raw CIR bytes/string (fed to the ML corrector).
        raw_range : float
            Uncorrected UWB range in metres.
        """
        rng_corr, sigma = self.ml.correct(cir_blob, raw_range)
        model = noiseModel.Isotropic.Sigma(1, sigma)

        # Pose ↔ Anchor
        if isinstance(key_j_or_point, (np.ndarray, list, tuple)):
            anchor = np.asarray(key_j_or_point, dtype=float).reshape(3)
            return _PoseAnchor(key_i, anchor, rng_corr, model)

        # Pose ↔ Pose
        return _PosePose(key_i, int(key_j_or_point), rng_corr, model)