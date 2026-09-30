"""Selbstlernender Signal-Filter (logistische Regression, ohne Zusatzpakete).

Lernt aus ABGESCHLOSSENEN Trades, bei welchen Merkmalen ein Signal eher gewinnt,
und verwirft Signale mit geringer Gewinnwahrscheinlichkeit. Im Backtest wird nur
mit Trades trainiert, die VOR dem aktuellen Signal abgeschlossen waren (kein Blick
in die Zukunft).
"""
import json
import math

import numpy as np

REGIMES = ["trend_up", "trend_down", "range", "squeeze", "unclear"]
STRATEGIES = ["trend", "range", "breakout"]
NUMERIC = ["rsi", "adx", "bb_width", "atr_pct", "vol_ratio", "macd_n", "trend_align"]
FEATURE_NAMES = NUMERIC + [f"regime_{r}" for r in REGIMES] + [f"strat_{s}" for s in STRATEGIES] + ["side"]


def _num(v, default=0.0) -> float:
    try:
        f = float(v)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def features(row: dict, side: str) -> list[float]:
    """Merkmale eines Signals. `row` enthaelt die Indikatorwerte des Signal-Bars."""
    sign = 1 if side == "long" else -1
    x = [
        _num(row.get("rsi"), 50) / 100,
        _num(row.get("adx"), 20) / 100,
        _num(row.get("bb_width")) * 10,
        _num(row.get("atr_pct")),
        min(_num(row.get("vol_ratio"), 1), 5),
        max(-3, min(3, _num(row.get("macd_n")))) * sign,
        _num(row.get("trend")) * sign,
    ]
    x += [1.0 if row.get("regime") == r else 0.0 for r in REGIMES]
    x += [1.0 if row.get("strategy") == s else 0.0 for s in STRATEGIES]
    x.append(float(sign))
    return x


class SignalModel:
    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        self.w: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.n = 0

    @property
    def ready(self) -> bool:
        return self.w is not None

    def fit(self, X: list[list[float]], y: list[int], iters: int = 300, lr: float = 0.3) -> None:
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        if len(X) < 10 or y.min() == y.max():
            return
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-9] = 1.0  # konstante Merkmale (auch mit Rundungsrauschen) nicht aufblasen
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Z @ w))
            grad = Z.T @ (p - y) / len(y)
            grad[1:] += self.l2 * w[1:] / len(y)
            w -= lr * grad
        self.w, self.n = w, len(X)

    def proba(self, x: list[float]) -> float | None:
        if not self.ready:
            return None
        z = np.r_[1.0, (np.asarray(x, float) - self.mu) / self.sd]
        return float(1 / (1 + np.exp(-z @ self.w)))

    def to_json(self) -> str:
        return json.dumps({"w": self.w.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                           "n": self.n, "features": FEATURE_NAMES}) if self.ready else "{}"

    @classmethod
    def from_json(cls, text: str) -> "SignalModel":
        m = cls()
        d = json.loads(text or "{}")
        if d.get("features") == FEATURE_NAMES and d.get("w"):
            m.w, m.mu, m.sd, m.n = np.array(d["w"]), np.array(d["mu"]), np.array(d["sd"]), d["n"]
        return m


def train_from_examples(examples: list[dict]) -> SignalModel:
    """examples: [{'x': [...], 'win': 0/1}, ...]"""
    m = SignalModel()
    m.fit([e["x"] for e in examples], [int(e["win"]) for e in examples])
    return m
