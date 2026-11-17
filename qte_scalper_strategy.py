"""
QTE Micro-Scalping Strategy & Session-Window Engine

Vectorizes EMA(10) price-action momentum triggers and multi-session time-window filters
(including overnight wraparound sessions) in Polars, and simulates asymmetric tick-level
bracket execution (tight stop-loss vs. fixed profit-target OCO orders).
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
import numpy as np
import polars as pl


@dataclass
class SessionWindow:
    start: time
    end: time
    enabled: bool = True

    def contains(self, ts: time) -> bool:
        if not self.enabled:
            return False
        if self.start <= self.end:
            return self.start <= ts <= self.end
        # Handles overnight sessions that wrap past midnight (e.g., 18:00 -> 03:00)
        return ts >= self.start or ts <= self.end


@dataclass
class ScalperConfig:
    ema_period: int = 10
    tick_size: float = 0.25
    point_value: float = 20.0
    contracts: int = 1
    stop_loss_ticks: int = 1
    profit_target_ticks: int = 8
    windows: list[SessionWindow] = field(
        default_factory=lambda: [
            SessionWindow(time(6, 30), time(13, 0), enabled=True),
            SessionWindow(time(18, 0), time(21, 0), enabled=False),
            SessionWindow(time(3, 0), time(5, 0), enabled=False),
        ]
    )


def compute_scalper_signals(df: pl.DataFrame, cfg: ScalperConfig) -> pl.DataFrame:
    """Computes EMA(10) and vectorized long/short micro-scalp entry signals."""
    df = df.sort("datetime").with_columns([
        pl.col("close").ewm_mean(span=cfg.ema_period, adjust=False).alias("ema"),
        pl.col("high").shift(1).alias("prev_high"),
        pl.col("low").shift(1).alias("prev_low"),
    ])

    df = df.with_columns([
        (
            (pl.col("close") > pl.col("open"))
            & (pl.col("high") >= pl.col("ema"))
            & (pl.col("high") < pl.col("prev_high"))
        ).alias("long_signal"),
        (
            (pl.col("close") < pl.col("open"))
            & (pl.col("low") <= pl.col("ema"))
            & (pl.col("low") > pl.col("prev_low"))
        ).alias("short_signal"),
    ])

    return df


def run_qte_scalper_backtest(df: pl.DataFrame, cfg: ScalperConfig = ScalperConfig()) -> pl.DataFrame:
    """Simulates session-filtered entries and tight asymmetric OCO bracket exits."""
    data = compute_scalper_signals(df, cfg)
    rows = data.to_dicts()
    trades = []
    position = None

    stop_dist = cfg.stop_loss_ticks * cfg.tick_size
    target_dist = cfg.profit_target_ticks * cfg.tick_size

    for i in range(cfg.ema_period + 2, len(rows)):
        bar = rows[i]

        # Manage open bracket order (conservative: check stop before target on same bar)
        if position is not None:
            is_long = position["dir"] == 1
            stop_hit = (bar["low"] <= position["stop"]) if is_long else (bar["high"] >= position["stop"])
            target_hit = (bar["high"] >= position["target"]) if is_long else (bar["low"] <= position["target"])

            if stop_hit or target_hit:
                exit_px = position["stop"] if stop_hit else position["target"]
                exit_reason = "Stop" if stop_hit else "Target"
                pnl = (exit_px - position["entry"]) * position["dir"] * cfg.point_value * cfg.contracts

                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": bar["datetime"],
                    "direction": "Long" if is_long else "Short",
                    "entry_px": round(position["entry"], 2),
                    "exit_px": round(exit_px, 2),
                    "exit_reason": exit_reason,
                    "pnl_usd": round(pnl, 2),
                })
                position = None
            continue

        # Check session time windows for new entries
        bar_time = bar["datetime"].time()
        if not any(w.contains(bar_time) for w in cfg.windows):
            continue

        if bar["long_signal"]:
            entry_px = bar["close"]
            position = {
                "dir": 1,
                "entry": entry_px,
                "stop": entry_px - stop_dist,
                "target": entry_px + target_dist,
                "entry_time": bar["datetime"],
            }
        elif bar["short_signal"]:
            entry_px = bar["close"]
            position = {
                "dir": -1,
                "entry": entry_px,
                "stop": entry_px + stop_dist,
                "target": entry_px - target_dist,
                "entry_time": bar["datetime"],
            }

    return pl.DataFrame(trades)


def generate_sample_ohlcv(n_bars: int = 5_000, seed: int = 42) -> pl.DataFrame:
    """Generates synthetic 1-minute OHLCV bars across intraday sessions."""
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 2, 6, 30)
    datetimes = [start + timedelta(minutes=i) for i in range(n_bars)]

    returns = rng.normal(loc=0.0, scale=3.5, size=n_bars)
    close = 21000.0 + np.cumsum(returns)
    open_ = np.roll(close, 1)
    open_[0] = 21000.0
    high = np.maximum(open_, close) + rng.uniform(0.25, 4.0, size=n_bars)
    low = np.minimum(open_, close) - rng.uniform(0.25, 4.0, size=n_bars)
    volume = rng.integers(50, 2000, size=n_bars)

    return pl.DataFrame({
        "datetime": datetimes,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


if __name__ == "__main__":
    sample_df = generate_sample_ohlcv()
    trade_log = run_qte_scalper_backtest(sample_df)
    print(f"Simulated {len(trade_log)} QTE Scalper trades:")
    print(trade_log.tail(10))
