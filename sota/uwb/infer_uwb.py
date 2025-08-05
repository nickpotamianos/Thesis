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
    def correct(self, cir: np.ndarray, raw_range: float):
        x = torch.from_numpy(cir[:128]).float().unsqueeze(0)   # (1, 128)
        los_prob = torch.softmax(self.cls(x),-1)[0,0].item()               # P(LOS)
        bias     = self.reg(x).item()
        if los_prob < 0.5:   # treat NLOS – inflate noise
            sigma = 0.20
        else:
            sigma = 0.05
        return raw_range - bias, sigma
