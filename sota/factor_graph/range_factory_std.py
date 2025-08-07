# -*- coding: utf-8 -*-
"""
RangeFactoryStd
===============

A drop‑in replacement for the previous Python‑level factors.
Uses only **native C++ GTSAM range factors** – no custom Jacobians.

• Pose ↔ Point3 (anchor) : gtsam.RangeFactorPose3Point3
• Pose ↔ Pose            : gtsam.RangeFactorPose3
"""

from __future__ import annotations
import numpy as np
import gtsam
from gtsam import noiseModel
from gtsam.symbol_shorthand import P, A         # convenience symbols
from sota.uwb.infer_uwb import UwbML


class RangeFactoryStd:
    """Handles ML correction + factor creation."""

    # ------------------------------------------------------------------ #
    def __init__(self, uwb_ml: UwbML):
        self.ml            = uwb_ml
        self._anchors_done = False        # safety flag
        self._akey         = {}           # anchor_id  ->  Point3 key

    # ------------------------------------------------------------------ #
    # Call ONCE, immediately after you know all anchor coordinates.
    def load_anchors(self,
                     graph: gtsam.NonlinearFactorGraph,
                     values: gtsam.Values,
                     anchors_xyz: dict[int, np.ndarray],
                     sigma_fix: float = 1e-6) -> None:
        """
        Adds every anchor as a Point3 *variable* plus a very tight prior
        so it behaves as a fixed landmark.

        Parameters
        ----------
        graph / values : the global containers you already created.
        anchors_xyz    : {anchor_id: np.ndarray([x, y, z])}
        sigma_fix      : prior std‑dev in metres (default 10⁻⁶ m).
        """
        model_fix = noiseModel.Isotropic.Sigma(3, sigma_fix)

        for aid, xyz in anchors_xyz.items():
            key = A(aid)                     # 64‑bit symbol → int
            self._akey[aid] = key

            pt = gtsam.Point3(*xyz)
            values.insert(key, pt)
            graph.add(gtsam.PriorFactorPoint3(key, pt, model_fix))

        self._anchors_done = True

    # ------------------------------------------------------------------ #
    # Call EVERY TIME you want to fuse one UWB measurement.
    def add_factor(self,
                   graph: gtsam.NonlinearFactorGraph,
                   key_i: int,          # Pose3 key of transmitting robot
                   to_id: int,          # anchor id OR Pose3 key
                   cir_blob,
                   raw_range: float,
                   noise_floor: float = 0.20) -> None:
        """
        Performs ML correction and *directly appends* the factor to graph.
        """
        assert self._anchors_done, "call load_anchors() first"

        rng_corr, sigma_ml = self.ml.correct(cir_blob, raw_range)
        sigma = float(max(sigma_ml, noise_floor))    # never < noise_floor
        model = noiseModel.Isotropic.Sigma(1, sigma)

        # Pose ↔ **Anchor**
        if to_id in self._akey:
            factor = gtsam.RangeFactor3D(
                key_i,                   # Pose3 key
                self._akey[to_id],       # Point3 key
                rng_corr, model)
        # Pose ↔ **Pose**
        else:
            factor = gtsam.RangeFactorPose3(
                key_i,
                int(to_id),
                rng_corr, model)

        graph.add(factor)
