"""
Given a pd.Series row from uwb_cir.csv, returns:
    range_corrected [m], sigma [m]
"""
import torch, numpy as np
from sota.uwb.models.cnn_nlos import Conv1dTiny
from sota.uwb.models.cnn_bias_regressor import Conv1dTiny as BiasNet
from pathlib import Path, PurePath
_ckpt_dir = Path(__file__).parent/'models'

class UwbML:
    def __init__(self, exp_name):
        self.cls = Conv1dTiny(2);   self.cls.load_state_dict(torch.load(_ckpt_dir/f"nlos_{exp_name}.pth")); self.cls.eval()
        self.reg = BiasNet(1);      self.reg.load_state_dict(torch.load(_ckpt_dir/f"bias_{exp_name}.pth")); self.reg.eval()
    @torch.no_grad()
    def correct(self, cir, raw_range):
        # -------------------------------------------------- #
        # 1) no CIR?  →  fall back to raw reading
        # -------------------------------------------------- #
        if cir is None or (isinstance(cir, float) and np.isnan(cir)):
            return raw_range, 0.10

        # -------------------------------------------------- #
        # 2) convert to np array (handles str, list, ndarray)
        # -------------------------------------------------- #
        if isinstance(cir, str):
            cir = np.asarray(eval(cir), dtype=np.float32)
        else:
            cir = np.asarray(cir, dtype=np.float32)

        x = torch.from_numpy(cir[:128]).float().unsqueeze(0)               # (1,128)

        los_prob = torch.softmax(self.cls(x), -1)[0, 0].item()   # P(LOS)
        bias     = self.reg(x).item()

        sigma = 0.05 if los_prob >= 0.5 else 0.20
        return raw_range - bias, sigma
