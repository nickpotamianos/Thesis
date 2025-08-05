import torch, torch.nn as nn

class Conv1dTiny(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(1,16,5,2), nn.ReLU(),
            nn.Conv1d(16,32,5,2), nn.ReLU(),
            nn.Conv1d(32,64,5,2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(64, out_dim)
    def forward(self,x):
        x = self.fe(x.unsqueeze(1))
        return self.head(x.squeeze(-1))
