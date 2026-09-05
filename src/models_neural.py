"""Neural network regressors: a feed-forward MLP on tabular weather
features, and a 1D-CNN over short lag-windows of weather readings (captures
thermal lag / ramp effects that a pointwise model cannot).

Both are wrapped in a small sklearn-style fit/predict class so they slot
into the same training harness as the classical models.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TorchRegressorBase:
    def __init__(self, epochs: int = 40, batch_size: int = 256, lr: float = 1e-3, random_state: int = 42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.x_mean_ = None
        self.x_std_ = None
        self.y_mean_ = None
        self.y_std_ = None
        self.model_ = None

    def _build_model(self, n_features: int) -> nn.Module:
        raise NotImplementedError

    def _prepare_x(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        raw_X = np.asarray(X, dtype=np.float32)
        n_features = raw_X.shape[-1]
        X = self._prepare_x(raw_X)
        y = np.asarray(y, dtype=np.float32)

        flat = X.reshape(len(X), -1)
        self.x_mean_ = flat.mean(axis=0)
        self.x_std_ = flat.std(axis=0) + 1e-6
        self.y_mean_ = y.mean()
        self.y_std_ = y.std() + 1e-6

        X_norm = (flat - self.x_mean_) / self.x_std_
        X_norm = X_norm.reshape(X.shape)
        y_norm = (y - self.y_mean_) / self.y_std_

        device = _device()
        self.model_ = self._build_model(n_features).to(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        dataset = TensorDataset(torch.from_numpy(X_norm), torch.from_numpy(y_norm))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self.model_(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        X = self._prepare_x(np.asarray(X, dtype=np.float32))
        flat = X.reshape(len(X), -1)
        X_norm = ((flat - self.x_mean_) / self.x_std_).reshape(X.shape)

        device = _device()
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(torch.from_numpy(X_norm).to(device)).squeeze(-1).cpu().numpy()
        return pred * self.y_std_ + self.y_mean_


class MLPRegressorTorch(_TorchRegressorBase):
    """Two hidden layers (64/32 units) on flat tabular weather features."""

    def _build_model(self, n_features: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )


class CNN1DRegressor(_TorchRegressorBase):
    """1D convolution over a (window, n_features) lag sequence per sample,
    i.e. input shape (batch, window, n_features)."""

    def _prepare_x(self, X: np.ndarray) -> np.ndarray:
        # torch conv1d expects (batch, channels, length); we treat each
        # weather feature as a channel and the lag window as the sequence.
        return np.transpose(X, (0, 2, 1))

    def _build_model(self, n_features: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )


def get_neural_models(random_state: int = 42) -> dict:
    return {
        "mlp": MLPRegressorTorch(epochs=40, random_state=random_state),
        "cnn_1d": CNN1DRegressor(epochs=40, random_state=random_state),
    }
