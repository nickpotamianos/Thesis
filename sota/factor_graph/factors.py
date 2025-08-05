import gtsam, torch, numpy as np
from gtsam import (BetweenFactorPose3, noiseModel)
from sota.uwb.infer_uwb import UwbML
from scipy.spatial.transform import Rotation as R

class DeepImuFactor(gtsam.CustomFactor):
    # pose_i, vel_i, bias_i  → pose_j, vel_j, bias_j
    pass  # wraps RepILN outputs + covariance

class RangeFactorWithBias(gtsam.CustomFactor):
    def __init__(self, key_from, key_to, cir, raw_range, uwb_ml:UwbML):
        self.key_from, self.key_to = key_from, key_to
        self.cir = cir; self.raw = raw_range; self.uwb_ml = uwb_ml
        super().__init__(self.get_noise(), [key_from, key_to])
    def get_noise(self):
        r_cor, sigma = self.uwb_ml.correct(self.cir, self.raw)
        return noiseModel.Isotropic.Sigma(1, sigma), r_cor
    def error(self, values):
        p_i = values.atPose3(self.key_from).translation()
        p_j = values.atPose3(self.key_to).translation()
        return np.array([np.linalg.norm(p_i-p_j) - self.r_cor])
