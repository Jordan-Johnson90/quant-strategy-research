"""
Holy Grail Trend-Pullback Strategy & Risk Engine (Raschke / Connors, "Street Smarts", Ch. 10)

Computes vectorized Wilder's ADX(14) and EMA(20) indicators via Polars/NumPy, then runs an
event-driven bar-close simulation featuring rolling stop-order entries, swing-point targets
(Point A / Point B), single re-entry logic, and real-time daily loss/drawdown lockouts.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import polars as pl


@dataclass
class StrategyConfig:
    adx_period: int = 14
    adx_threshold: float = 30.0
    ema_period: int = 20
    tick_size: float = 0.25
    point_value: float = 20.0
    entry_offset_ticks: int = 1
    stop_offset_ticks: int = 1
    entry_window_bars: int = 5
    roll_entry: bool = True
    max_swing_lookback: int = 100
    reenter_after_stop: bool = True
    max_bars_in_trade: int = 50
    contracts: int = 1
    max_daily_loss: float = 500.0
    max_daily_drawdown: float = 500.0


def compute_indicators(df: pl.DataFrame, cfg: StrategyConfig) -> pl.DataFrame:
    """Computes EMA(20) and Wilder's ADX(14) without lookahead bias."""
    df = df.sort("datetime").with_columns([
        pl.col("close").ewm_mean(span=cfg.ema_period, adjust=False).alias("ema"),
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ).alias("tr"),
        (pl.col("high") - pl.col("high").shift(1)).alias("up_move"),
        (pl.col("low").shift(1) - pl.col("low")).alias("down_move"),
    ])

    df = df.with_columns([
        pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0))
        .then(pl.col("up_move"))
        .otherwise(0.0)
        .alias("plus_dm"),
        pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0))
        .then(pl.col("down_move"))
        .otherwise(0.0)
        .alias("minus_dm"),
    ])

    alpha = 1.0 / cfg.adx_period
    df = df.with_columns([
        pl.col("tr").ewm_mean(alpha=alpha, adjust=False).alias("atr"),
        pl.col("plus_dm").ewm_mean(alpha=alpha, adjust=False).alias("plus_di_raw"),
        pl.col("minus_dm").ewm_mean(alpha=alpha, adjust=False).alias("minus_di_raw"),
    ]).with_columns([
        (100.0 * pl.col("plus_di_raw") / pl.col("atr")).alias("plus_di"),
        (100.0 * pl.col("minus_di_raw") / pl.col("atr")).alias("minus_di"),
    ]).with_columns([
        (
            100.0
            * (pl.col("plus_di") - pl.col("minus_di")).abs()
            / (pl.col("plus_di") + pl.col("minus_di")).fill_nan(1.0)
        )
        .ewm_mean(alpha=alpha, adjust=False)
        .alias("adx")
    ])

    return df.select(["datetime", "open", "high", "low", "close", "volume", "ema", "adx"])


def run_holy_grail_backtest(df: pl.DataFrame, cfg: StrategyConfig = StrategyConfig()) -> pl.DataFrame:
    """Simulates bar-by-bar execution, rolling stop entries, re-entries, and daily risk limits."""
    data = compute_indicators(df, cfg)
    rows = data.to_dicts()
    trades = []

    armed = False
    wait_turn = False
    pending = None
    position = None
    last_trade_info = None

    current_day = None
    day_realized = 0.0
    day_peak = 0.0
    limit_hit = False

    for i in range(cfg.adx_period * 3, len(rows)):
        bar = rows[i]
        prev = rows[i - 1]
        bar_date = bar["datetime"].date()

        # Daily reset & risk-budget check
        if bar_date != current_day:
            current_day = bar_date
            day_realized = 0.0
            day_peak = 0.0
            limit_hit = False

        if limit_hit:
            continue

        # Update ADX arming state
        if bar["adx"] <= cfg.adx_threshold:
            armed = False
            wait_turn = False
        elif wait_turn:
            if bar["adx"] < prev["adx"]:
                wait_turn = False
        elif bar["adx"] > prev["adx"]:
            armed = True

        # Manage open position
        if position is not None:
            position["bars_held"] += 1
            is_long = position["dir"] == 1
            stop_hit = (bar["low"] <= position["stop"]) if is_long else (bar["high"] >= position["stop"])
            target_hit = (bar["high"] >= position["target"]) if is_long else (bar["low"] <= position["target"])
            time_hit = cfg.max_bars_in_trade > 0 and position["bars_held"] >= cfg.max_bars_in_trade

            if stop_hit or target_hit or time_hit:
                exit_px = position["stop"] if stop_hit else (position["target"] if target_hit else bar["close"])
                exit_reason = "Stop" if stop_hit else ("Target" if target_hit else "Time")
                pnl = (exit_px - position["entry"]) * position["dir"] * cfg.point_value * cfg.contracts

                day_realized += pnl
                day_peak = max(day_peak, day_realized)
                if (cfg.max_daily_loss > 0 and day_realized <= -cfg.max_daily_loss) or (
                    cfg.max_daily_drawdown > 0 and day_peak - day_realized >= cfg.max_daily_drawdown
                ):
                    limit_hit = True

                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": bar["datetime"],
                    "direction": "Long" if is_long else "Short",
                    "entry_px": round(position["entry"], 2),
                    "exit_px": round(exit_px, 2),
                    "exit_reason": exit_reason,
                    "is_reentry": position["is_reentry"],
                    "pnl_usd": round(pnl, 2),
                })

                if pnl > 0:
                    armed = False
                    wait_turn = True
                    last_trade_info = None
                elif cfg.reenter_after_stop and stop_hit and not position["is_reentry"]:
                    last_trade_info = position.copy()
                else:
                    last_trade_info = None

                position = None
            continue

        # Check pending stop order fill or roll
        if pending is not None:
            is_long = pending["dir"] == 1
            filled = (bar["high"] >= pending["entry"]) if is_long else (bar["low"] <= pending["entry"])

            if filled:
                position = {
                    "dir": pending["dir"],
                    "entry": pending["entry"],
                    "stop": pending["stop"],
                    "target": pending["target"],
                    "entry_time": bar["datetime"],
                    "bars_held": 0,
                    "is_reentry": pending["is_reentry"],
                }
                pending = None
                continue

            # Roll entry if still touching EMA, or expire
            touch = (bar["low"] <= bar["ema"]) if is_long else (bar["high"] >= bar["ema"])
            if cfg.roll_entry and touch and not pending["is_reentry"]:
                pending["extreme"] = min(pending["extreme"], bar["low"]) if is_long else max(pending["extreme"], bar["high"])
                pending["stop"] = pending["extreme"] - cfg.stop_offset_ticks * cfg.tick_size if is_long else pending["extreme"] + cfg.stop_offset_ticks * cfg.tick_size
                pending["entry"] = bar["high"] + cfg.entry_offset_ticks * cfg.tick_size if is_long else bar["low"] - cfg.entry_offset_ticks * cfg.tick_size

            if i >= pending["expire_bar"] or bar["adx"] <= cfg.adx_threshold:
                pending = None
            continue

        # Single re-entry attempt after initial stop-out
        if last_trade_info is not None and bar["adx"] > cfg.adx_threshold:
            is_long = last_trade_info["dir"] == 1
            valid_side = (bar["close"] < last_trade_info["entry"]) if is_long else (bar["close"] > last_trade_info["entry"])
            if valid_side:
                new_extreme = min(last_trade_info["stop"], bar["low"]) if is_long else max(last_trade_info["stop"], bar["high"])
                pending = {
                    "dir": last_trade_info["dir"],
                    "entry": last_trade_info["entry"],
                    "stop": new_extreme - cfg.stop_offset_ticks * cfg.tick_size if is_long else new_extreme + cfg.stop_offset_ticks * cfg.tick_size,
                    "target": last_trade_info["target"],
                    "expire_bar": i + cfg.entry_window_bars,
                    "extreme": new_extreme,
                    "is_reentry": True,
                }
            last_trade_info = None
            continue

        # Detect new 20-EMA pullback touch while ADX > 30 and rising
        if armed and bar["adx"] > cfg.adx_threshold:
            long_touch = bar["low"] <= bar["ema"] and prev["low"] > prev["ema"] and bar["ema"] >= prev["ema"]
            short_touch = bar["high"] >= bar["ema"] and prev["high"] < prev["ema"] and bar["ema"] <= prev["ema"]

            if long_touch ^ short_touch:
                is_long = long_touch
                lookback_start = max(0, i - cfg.max_swing_lookback)
                swing_slice = rows[lookback_start:i]
                point_a = max(r["high"] for r in swing_slice) if is_long else min(r["low"] for r in swing_slice)
                extreme = bar["low"] if is_long else bar["high"]
                entry_px = bar["high"] + cfg.entry_offset_ticks * cfg.tick_size if is_long else bar["low"] - cfg.entry_offset_ticks * cfg.tick_size
                stop_px = extreme - cfg.stop_offset_ticks * cfg.tick_size if is_long else extreme + cfg.stop_offset_ticks * cfg.tick_size

                if (is_long and point_a > entry_px) or (not is_long and point_a < entry_px):
                    pending = {
                        "dir": 1 if is_long else -1,
                        "entry": entry_px,
                        "stop": stop_px,
                        "target": point_a,
                        "expire_bar": i + cfg.entry_window_bars,
                        "extreme": extreme,
                        "is_reentry": False,
                    }

    return pl.DataFrame(trades)


def generate_sample_ohlcv(n_bars: int = 5_000, seed: int = 42) -> pl.DataFrame:
    """Generates synthetic trending OHLCV data for standalone testing."""
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 2, 9, 30)
    datetimes = [start + timedelta(minutes=15 * i) for i in range(n_bars)]

    # Inject mild directional regimes so ADX exceeds 30 during simulated trends
    drift = np.sin(np.linspace(0, 12 * np.pi, n_bars)) * 2.5
    returns = rng.normal(loc=drift, scale=8.0, size=n_bars)
    close = 21000.0 + np.cumsum(returns)
    open_ = np.roll(close, 1)
    open_[0] = 21000.0
    high = np.maximum(open_, close) + rng.uniform(1.0, 10.0, size=n_bars)
    low = np.minimum(open_, close) - rng.uniform(1.0, 10.0, size=n_bars)
    volume = rng.integers(100, 3000, size=n_bars)

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
    trade_log = run_holy_grail_backtest(sample_df)
    print(f"Simulated {len(trade_log)} Holy Grail trades:")
    print(trade_log.tail(10))
