"""Zusatz-Filter, die im Backtest UND live gleich arbeiten."""

LEADER = "BTC"


def is_leader(symbol: str) -> bool:
    return symbol.split("/")[0] == LEADER


def leader_blocks(side: str, leader_regime: str | None) -> bool:
    """BTC faellt im Trend -> keine Altcoin-Longs; BTC steigt im Trend -> keine Altcoin-Shorts."""
    if leader_regime == "trend_down" and side == "long":
        return True
    if leader_regime == "trend_up" and side == "short":
        return True
    return False


def strategy_healthy(pnls: list[float], s: dict) -> bool:
    """Letzte `health_window` Trades dieser Strategie: Profit-Faktor unter Grenze -> Pause."""
    n = s.get("health_window", 0)
    if not n or len(pnls) < n:
        return True
    last = pnls[-n:]
    loss = -sum(p for p in last if p <= 0)
    win = sum(p for p in last if p > 0)
    if loss == 0:
        return True
    return win / loss >= s.get("health_min_pf", 0.6)


def health_gate(strategy: str, pnls: list[float], s: dict, skips: dict) -> bool:
    """True = Signal handeln. Kranke Strategie pausiert, aber nach `health_pause_signals`
    uebersprungenen Signalen gibt es einen Probe-Trade - sonst bliebe sie fuer immer aus."""
    if strategy_healthy(pnls, s):
        skips[strategy] = 0
        return True
    skips[strategy] = skips.get(strategy, 0) + 1
    if skips[strategy] > s.get("health_pause_signals", 10):
        skips[strategy] = 0
        return True
    return False


def time_stop_due(bars_held: int, progress_r: float, partial_done: bool, s: dict) -> bool:
    """Trade kommt nach `max_hold_bars` nicht vom Fleck (< +0,5R, kein Teilverkauf) -> schliessen."""
    max_bars = s.get("max_hold_bars", 0)
    return bool(max_bars) and bars_held >= max_bars and not partial_done and progress_r < 0.5
