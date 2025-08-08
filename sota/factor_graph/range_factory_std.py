# -*- coding: utf-8 -*-
"""
RangeFactoryStd (version-agnostic)
----------------------------------
Uses native GTSAM range factors only.

• If RangeFactorPose3Point3 is available:
    anchors as Point3 with PriorFactorPoint3
    Pose<->Point3 : RangeFactorPose3Point3
    Pose<->Pose   : RangeFactorPose3

• Else (older wrappers):
    anchors as Pose3 (R=I, t=xyz) with PriorFactorPose3
    Pose<->Pose   : RangeFactorPose3   (for both robot and anchor)
"""

from __future__ import annotations
import numpy as np
import gtsam
from gtsam import noiseModel
from gtsam.symbol_shorthand import A  # we will use A(aid) for anchor keys
from sota.uwb.infer_uwb import UwbML

# Capability probe
_HAS_POSE_POINT = hasattr(gtsam, "RangeFactorPose3Point3")

def _robust_sigma(sigma):
    """Create a robust noise model with Huber kernel for outlier tolerance."""
    base = gtsam.noiseModel.Isotropic.Sigma(1, sigma)
    huber = gtsam.noiseModel.mEstimator.Huber(1.345)  # classic choice
    return gtsam.noiseModel.Robust.Create(huber, base)


class RangeFactoryStd:
    """Handles ML correction + factor creation with a build-compatible anchor backend."""

    def __init__(self, uwb_ml: UwbML):
        self.ml              = uwb_ml
        self._anchors_done   = False
        self.anchor_keys     = {}    # {anchor_id: anchor_key}
        self._anchor_as_pose = not _HAS_POSE_POINT  # fallback if Pose3<->Point3 factor missing

    def load_anchors(
        self,
        graph: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
        anchors_xyz: dict[int, np.ndarray],
        sigma_fix_xyz: float = 1e-6,     # very tight position prior (meters)
        sigma_fix_rot: float = 1e-6      # very tight rotation prior (radians) for Pose3 fallback
    ) -> None:
        """
        Register anchors either as Point3 (preferred) or Pose3 (fallback),
        each with a very tight prior so they behave as fixed landmarks.
        """
        if self._anchor_as_pose:
            # Pose3 anchors + Pose3 prior
            sigmas = np.array([sigma_fix_rot, sigma_fix_rot, sigma_fix_rot,
                               sigma_fix_xyz, sigma_fix_xyz, sigma_fix_xyz], dtype=float)
            model_fix = noiseModel.Diagonal.Sigmas(sigmas)

            for aid, xyz in anchors_xyz.items():
                key = A(aid)  # this key will hold a Pose3 now
                self.anchor_keys[aid] = key
                pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(*np.asarray(xyz, float).reshape(3)))
                values.insert(key, pose)
                graph.add(gtsam.PriorFactorPose3(key, pose, model_fix))
        else:
            # Point3 anchors + Point3 prior
            model_fix = noiseModel.Isotropic.Sigma(3, float(sigma_fix_xyz))
            for aid, xyz in anchors_xyz.items():
                key = A(aid)  # this key will hold a Point3
                self.anchor_keys[aid] = key
                pt = gtsam.Point3(*np.asarray(xyz, float).reshape(3))
                values.insert(key, pt)
                graph.add(gtsam.PriorFactorPoint3(key, pt, model_fix))

        self._anchors_done = True

    def _anchor_factor_pose_point(
        self, key_pose: int, key_point: int, measured: float, model
    ):
        """
        Build Pose3<->Point3 range factor if supported; otherwise raise.
        """
        if not _HAS_POSE_POINT:
            raise AttributeError("Pose3<->Point3 range factor not available in this GTSAM build")
        return gtsam.RangeFactorPose3Point3(key_pose, key_point, float(measured), model)

    def add_factor(
        self,
        graph: gtsam.NonlinearFactorGraph,
        key_i: int,           # Pose3 key of transmitting robot
        to_id: int,           # anchor id OR Pose3 key
        cir_blob,
        raw_range: float,
        noise_floor: float = 0.20
    ) -> None:
        """
        ML-corrected range. Appends the factor to graph.
        """
        assert self._anchors_done, "call load_anchors() first"

        rng_corr, sigma_ml = self.ml.correct(cir_blob, float(raw_range))
        sigma = float(max(float(sigma_ml), noise_floor))
        model = noiseModel.Isotropic.Sigma(1, sigma)

        # --- Pose <-> Anchor -----------------------------------------
        if isinstance(to_id, (int, np.integer)) and int(to_id) in self.anchor_keys:
            key_anchor = self.anchor_keys[int(to_id)]
            if self._anchor_as_pose:
                # Fallback path: anchor is Pose3
                f = gtsam.RangeFactorPose3(int(key_i), int(key_anchor), rng_corr, model)
            else:
                # Preferred path: anchor is Point3
                f = self._anchor_factor_pose_point(int(key_i), int(key_anchor), rng_corr, model)
            graph.add(f)
            return

        # --- Pose <-> Pose (robot-robot) -----------------------------
        key_j = int(to_id)
        f = gtsam.RangeFactorPose3(int(key_i), key_j, rng_corr, model)
        graph.add(f)

    def add_range_only_factor(
        self,
        graph: gtsam.NonlinearFactorGraph,
        key_i: int,
        to_id: int,
        raw_range: float,
        sigma_m: float = 0.25
    ) -> None:
        """
        Plain (non-ML) range factor. Uses same anchor backend selection.
        Now with robust loss for outlier tolerance.
        """
        model = _robust_sigma(float(sigma_m))  # Use robust noise model
        r = float(raw_range)

        # Anchor target?
        if isinstance(to_id, (int, np.integer)) and int(to_id) in self.anchor_keys:
            key_anchor = self.anchor_keys[int(to_id)]
            if self._anchor_as_pose:
                f = gtsam.RangeFactorPose3(int(key_i), int(key_anchor), r, model)
            else:
                f = self._anchor_factor_pose_point(int(key_i), int(key_anchor), r, model)
            graph.add(f)
            return

        # Robot target
        key_j = int(to_id)
        f = gtsam.RangeFactorPose3(int(key_i), key_j, r, model)
        graph.add(f)
