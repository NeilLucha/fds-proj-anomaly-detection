"""Small PyTorch LSTM next-step predictor for comparison benchmarks.

Training consumes only the supplied training prefix. At scoring time, each
prediction uses the preceding fixed-length context and never sees the target
observation itself.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm


@dataclass
class LSTMResult:
    scores: np.ndarray
    alerts: np.ndarray
    training_time_s: float
    inference_time_s: float
    parameter_count: int
    lookback: int
    hidden_size: int
    epochs: int


class LSTMAnomalyDetector:
    """Single-layer LSTM forecaster with robustly calibrated residual scores."""

    def __init__(self, lookback: int = 24, hidden_size: int = 16, epochs: int = 5,
                 learning_rate: float = 1e-2, seed: int = 42) -> None:
        self.lookback = int(lookback)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.model = None
        self.center = 0.0
        self.scale = 1.0

    def fit(self, values) -> float:
        import torch
        from torch import nn

        raw = np.asarray(values, dtype=np.float32)
        self.center = float(np.median(raw)) if len(raw) else 0.0
        mad = float(np.median(np.abs(raw - self.center))) if len(raw) else 1.0
        self.scale = max(1.4826 * mad, float(np.std(raw)) * 0.1 if len(raw) else 0.0, 1e-6)
        normalized = np.clip((raw - self.center) / self.scale, -20.0, 20.0)
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)

        class Forecaster(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
                self.linear = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.linear(out[:, -1, :]).squeeze(-1)

        start = time.perf_counter()
        self.model = Forecaster(self.hidden_size)
        if len(normalized) > self.lookback:
            x = np.stack([normalized[i-self.lookback:i] for i in range(self.lookback, len(normalized))])
            y = normalized[self.lookback:]
            tx = torch.from_numpy(x[:, :, None])
            ty = torch.from_numpy(y)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            loss_fn = nn.SmoothL1Loss()
            self.model.train()
            batch_size = min(128, len(tx))
            for _ in tqdm(range(self.epochs), desc="LSTM training", unit="epoch", leave=False):
                order = torch.randperm(len(tx))
                for ids in order.split(batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(self.model(tx[ids]), ty[ids])
                    loss.backward()
                    optimizer.step()
        self.model.eval()
        return time.perf_counter() - start

    def score(self, values, tau: float = 3.0, context=None) -> LSTMResult:
        import torch

        raw = np.asarray(values, dtype=np.float32)
        prior = np.asarray(context if context is not None else [], dtype=np.float32)
        normalized = np.clip((raw - self.center) / self.scale, -20.0, 20.0)
        prior_normalized = np.clip((prior - self.center) / self.scale, -20.0, 20.0)
        scores = np.zeros(len(raw), dtype=float)
        start = time.perf_counter()
        combined = np.concatenate([prior_normalized, normalized])
        if self.model is not None and len(raw) > 0 and len(combined) > self.lookback:
            self.model.eval()
            with torch.no_grad():
                windows = np.stack([
                    combined[j-self.lookback:j]
                    for j in range(max(self.lookback, len(prior_normalized)), len(combined))
                ])
                pred = self.model(torch.from_numpy(windows[:, :, None])).numpy()
            start_idx = max(self.lookback, len(prior_normalized)) - len(prior_normalized)
            scores[start_idx:] = np.abs(normalized[start_idx:] - pred)
        inference = time.perf_counter() - start
        return LSTMResult(
            scores=scores,
            alerts=scores > tau,
            training_time_s=0.0,
            inference_time_s=inference,
            parameter_count=sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0,
            lookback=self.lookback,
            hidden_size=self.hidden_size,
            epochs=self.epochs,
        )

    def threshold_from(self, calibration_scores, target_fpr: float = 0.01) -> float:
        scores = np.asarray(calibration_scores, dtype=float)
        return float(np.quantile(scores, 1.0 - target_fpr)) if len(scores) else 3.0

    def fit_score(self, train_values, values, tau: float = 3.0):
        training_time = self.fit(train_values)
        result = self.score(values, tau=tau)
        result.training_time_s = training_time
        return result
