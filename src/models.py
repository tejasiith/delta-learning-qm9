import numpy as np
import torch
from torch import nn


class ResidualMLP:
    # fits whatever the linear model leaves over

    def __init__(self, hidden=256, lr=1e-3, wd=1e-4, dropout=0.0, epochs=300,
                 patience=30, batch=128, seed=0, device=None):
        self.hidden, self.lr, self.wd, self.dropout = hidden, lr, wd, dropout
        self.epochs, self.patience, self.batch, self.seed = epochs, patience, batch, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _x(self, X):
        X = (np.asarray(X, dtype=np.float32) - self.mu) / self.sd
        return torch.tensor(X, dtype=torch.float32, device=self.device)

    def fit(self, X_tr, r_tr, X_va, r_va):
        torch.manual_seed(self.seed)
        X_tr = np.asarray(X_tr, dtype=np.float32)
        self.mu = X_tr.mean(0)
        self.sd = X_tr.std(0)
        self.sd[self.sd < 1e-6] = 1.0
        self.ry_mu, self.ry_sd = float(np.mean(r_tr)), float(np.std(r_tr)) + 1e-8

        Xt, Xv = self._x(X_tr), self._x(X_va)
        yt = torch.tensor((np.asarray(r_tr) - self.ry_mu) / self.ry_sd,
                          dtype=torch.float32, device=self.device)
        r_va = np.asarray(r_va)

        d = Xt.shape[1]
        self.net = nn.Sequential(
            nn.Linear(d, self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        ).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.MSELoss()

        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), device=self.device)
            for s in range(0, len(Xt), self.batch):
                b = perm[s:s + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[b]).squeeze(1), yt[b])
                loss.backward()
                opt.step()
            val_mae = float(np.mean(np.abs(self._predict_t(Xv) - r_va)))
            if val_mae < best - 1e-6:
                best, bad = val_mae, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_val_mae = best
        return self

    def _predict_t(self, Xt):
        self.net.eval()
        with torch.no_grad():
            out = self.net(Xt).squeeze(1).cpu().numpy()
        return out * self.ry_sd + self.ry_mu

    def predict(self, X):
        return self._predict_t(self._x(X))