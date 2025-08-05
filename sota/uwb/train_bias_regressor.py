import torch, torch.nn as nn, torch.optim as optim, numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from torch.utils.data import TensorDataset, DataLoader
from sota.uwb.models.cnn_bias_regressor import Conv1dTiny
import argparse, pathlib

def train(exp_name):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    d = np.load(f"sota/uwb/datasets/{exp_name}.npz")
    ds_train = TensorDataset(torch.from_numpy(d["X_train"]).float(),
                             torch.from_numpy(d["y_reg_train"]).float())
    ds_val   = TensorDataset(torch.from_numpy(d["X_val"]).float(),
                             torch.from_numpy(d["y_reg_val"]).float())
    net = Conv1dTiny(1).to(device)
    opt = optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    huber = nn.HuberLoss()
    for epoch in range(30):
        net.train()
        for X,y in DataLoader(ds_train, batch_size=256, shuffle=True):
            X,y = X.to(device), y.to(device)
            loss = huber(net(X).squeeze(-1), y)
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        net.eval(); val_loss = 0
        with torch.no_grad():
            for X,y in DataLoader(ds_val, 512):
                pred = net(X.to(device)).squeeze(-1)
                val_loss += huber(pred, y.to(device)).item()
        print(f"Epoch {epoch:02d}  val‑loss = {val_loss:.3f}")
    torch.save(net.state_dict(), f"sota/uwb/models/bias_{exp_name}.pth")

if __name__ == "__main__":
    import sys; train(sys.argv[1])
