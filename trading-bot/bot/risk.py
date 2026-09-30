"""Risiko-Management: Positionsgroesse und Schutzregeln."""
import math
from datetime import datetime, timedelta, timezone


def position_size(equity: float, entry: float, sl: float, leverage: int,
                  risk_pct: float, max_margin_pct: float, fee: float) -> float:
    """Menge so, dass ein Stop-Loss inkl. Gebuehren max. risk_pct vom Konto kostet."""
    dist = abs(entry - sl)
    if equity <= 0 or dist <= 0:
        return 0.0
    risk_amount = equity * risk_pct / 100
    amount = risk_amount / (dist + 2 * fee * entry)
    max_notional = equity * max_margin_pct / 100 * leverage
    return min(amount, max_notional / entry)


def round_amount(amount: float, step: float, min_amount: float) -> float:
    if step and step > 0:
        amount = math.floor(amount / step) * step
    return amount if amount >= (min_amount or 0) else 0.0


def stop_is_safe(entry: float, sl: float, leverage: int) -> bool:
    """Stop muss deutlich vor der Liquidation liegen (max. halber Abstand)."""
    return abs(entry - sl) / entry < 0.5 / leverage


class RiskGuard:
    """Tageslimit, Trade-Limit und Pause nach Verlustserie.

    `state` ist ein dict, damit es als JSON gespeichert werden kann.
    """

    def __init__(self, cfg: dict, state: dict):
        self.cfg = cfg
        self.s = state
        self.s.setdefault("day", "")
        self.s.setdefault("day_start_equity", 0.0)
        self.s.setdefault("trades_today", 0)
        self.s.setdefault("consec_losses", 0)
        self.s.setdefault("pause_until", 0.0)

    def update_day(self, equity: float, now: datetime) -> None:
        day = now.astimezone(timezone.utc).strftime("%Y-%m-%d")
        if self.s["day"] != day:
            self.s["day"] = day
            self.s["day_start_equity"] = equity
            self.s["trades_today"] = 0

    def can_open(self, equity: float, open_positions: int, now: datetime) -> tuple[bool, str]:
        c = self.cfg
        if now.timestamp() < self.s["pause_until"]:
            return False, "Pause nach Verlustserie"
        start = self.s["day_start_equity"]
        if start > 0 and equity <= start * (1 - c["daily_loss_limit_pct"] / 100):
            return False, "Tagesverlust-Limit erreicht"
        if self.s["trades_today"] >= c["max_trades_per_day"]:
            return False, "Max. Trades pro Tag erreicht"
        if open_positions >= c["max_open_positions"]:
            return False, "Max. offene Positionen"
        return True, ""

    def on_open(self) -> None:
        self.s["trades_today"] += 1

    def on_close(self, pnl: float, now: datetime) -> None:
        if pnl < 0:
            self.s["consec_losses"] += 1
            if self.s["consec_losses"] >= self.cfg["max_consecutive_losses"]:
                pause = timedelta(minutes=self.cfg["pause_after_losses_minutes"])
                self.s["pause_until"] = (now + pause).timestamp()
                self.s["consec_losses"] = 0
        else:
            self.s["consec_losses"] = 0
