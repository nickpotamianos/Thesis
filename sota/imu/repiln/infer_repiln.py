import torch
from sota.imu.repiln.model import RepILN

class DeepImuFrontend:
    def __init__(self, ckpt='repiln_miluv.pth'):
        self.net = RepILN(); self.net.load_state_dict(torch.load(ckpt)); self.net.eval()
        self.window = []
    def feed_sample(self, gyrox, gyroy, gyroz, accx, accy, accz, dt):
        self.window.append([gyrox,gyroy,gyroz,accx,accy,accz])
        if len(self.window)==200:
            import torch
            imu_seq = torch.tensor(self.window).float().unsqueeze(0)  # 1×200×6
            ΔR,Δv,Δp,Σ = self.net(imu_seq)
            self.window.clear()
            return ΔR.numpy()[0], Δv.numpy()[0], Δp.numpy()[0], Σ.numpy()[0]
        return None
