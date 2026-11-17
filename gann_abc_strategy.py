"""
Gann ABC Retracement & Daily High Breakout Strategy

Detects prior-day high breaches via Polars daily resampling, dynamically selects the
intraday evaluation timeframe based on proximity to market open, reconstructs Gann
swing legs (absorbing inside-bar "Hoagie" consolidations), validates ABC retracement
geometry, and simulates trailing buy-stop execution with R-multiple targets.
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
import numpy as np
import polars as pl


@dataclass
class GannABCConfig:
    tick_size: float = 0.25
    point_value: float = 20.0
    contracts: int = 1
    profit_multiplier: float = 4.0
    leg_c_max_ratio: float = 3.0
    enable_time_filter: bool = True
    trade_start: time = time(6, 30)
    trade_end: time = time(13, 0)
    continue_after_loss: bool = True
    use_fixed_timeframe: bool = False
    fixed_timeframe_minutes: int = 5
    tf_5min_threshold_hours: float = 2.0
    tf_15min_threshold_hours: float = 4.0


@dataclass
class GannLeg:
    start_idx: int
    end_idx: int
    direction: int  # 1 = UP, -1 = DOWN
    start_price: float
    end_price: float
    high: float
    low: float

    @property
    def size(self) -> float:
        return abs(self.end_price - self.start_price)


def attach_prior_day_high(df: pl.DataFrame) -> pl.DataFrame:
    """Computes prior-day high via Polars dynamic daily grouping without lookahead bias."""
    df = df.sort("datetime").with_columns(pl.col("datetime").dt.date().alias("session_date"))

    daily_highs = (
        df.group_by("session_date", maintain_order=True)
        .agg(pl.col("high").max().alias("day_high"))
        .with_columns(pl.col("day_high").shift(1).alias("prev_day_high"))
        .select(["session_date", "prev_day_high"])
    )

    return df.join(daily_highs, on="session_date", how="left").drop("session_date")


def determine_timeframe(breach_time: datetime, cfg: GannABCConfig) -> int:
    """Selects 5m, 15m, or 30m evaluation timeframe based on hours before market open."""
    if cfg.use_fixed_timeframe:
        return cfg.fixed_timeframe_minutes

    market_open = datetime.combine(breach_time.date(), cfg.trade_start)
    hours_until_open = max(0.0, (market_open - breach_time).total_seconds() / 3600.0)

    if hours_until_open <= cfg.tf_5min_threshold_hours:
        return 5
    if hours_until_open <= cfg.tf_15min_threshold_hours:
        return 15
    return 30


def rebuild_gann_legs(bars: list[dict]) -> list[GannLeg]:
    """Reconstructs directional Gann swing legs while absorbing inside-bar consolidations."""
    if not bars:
        return []

    directions: list[int] = [1 if bars[0]["close"] > bars[0]["open"] else -1]
    hoagie_active = False
    hoagie_high = 0.0
    hoagie_low = 0.0

    for i in range(1, len(bars)):
        curr = bars[i]
        prev = bars[i - 1]

        if hoagie_active:
            if curr["high"] > hoagie_high and curr["low"] > hoagie_low:
                directions.append(1)
                hoagie_active = False
            elif curr["low"] < hoagie_low and curr["high"] < hoagie_high:
                directions.append(-1)
                hoagie_active = False
            elif curr["high"] <= hoagie_high and curr["low"] >= hoagie_low:
                directions.append(directions[-1])
            else:
                directions.append(directions[-1])
                hoagie_active = False
        else:
            if curr["high"] > prev["high"] and curr["low"] > prev["low"]:
                directions.append(1)
            elif curr["low"] < prev["low"] and curr["high"] < prev["high"]:
                directions.append(-1)
            elif curr["high"] < prev["high"] and curr["low"] > prev["low"]:
                directions.append(directions[-1])
                hoagie_active = True
                hoagie_high = prev["high"]
                hoagie_low = prev["low"]
            else:
                directions.append(directions[-1])

    # Group contiguous direction runs into overlapping GannLeg structures
    legs: list[GannLeg] = []
    leg_dir = directions[0]
    start_idx = 0

    for i in range(1, len(directions)):
        if directions[i] != leg_dir:
            legs.append(_build_leg(bars, start_idx, i - 1, leg_dir))
            leg_dir = directions[i]
            start_idx = i - 1

    legs.append(_build_leg(bars, start_idx, len(directions) - 1, leg_dir))
    return legs


def _build_leg(bars: list[dict], start_idx: int, end_idx: int, direction: int) -> GannLeg:
    leg_slice = bars[start_idx : end_idx + 1]
    leg_high = max(b["high"] for b in leg_slice)
    leg_low = min(b["low"] for b in leg_slice)
    return GannLeg(
        start_idx=start_idx,
        end_idx=end_idx,
        direction=direction,
        start_price=leg_low if direction == 1 else leg_high,
        end_price=leg_high if direction == 1 else leg_low,
        high=leg_high,
        low=leg_low,
    )


def find_valid_abc_pattern(legs: list[GannLeg], max_ratio: float) -> tuple[GannLeg, GannLeg, GannLeg] | None:
    """Scans recent Gann legs backwards for a valid Down(A) - Up(B) - Down(C) retracement."""
    if len(legs) < 3:
        return None

    for i in range(len(legs) - 1, 1, -1):
        leg_c, leg_b, leg_a = legs[i], legs[i - 1], legs[i - 2]
        if leg_a.direction == -1 and leg_b.direction == 1 and leg_c.direction == -1:
            if (
                leg_b.high < leg_a.start_price
                and leg_c.low < leg_a.low
                and leg_a.size > 0
                and leg_c.size <= leg_a.size * max_ratio
            ):
                return leg_a, leg_b, leg_c
    return None


def run_gann_abc_backtest(df: pl.DataFrame, cfg: GannABCConfig = GannABCConfig()) -> pl.DataFrame:
    """Simulates prior-day high breaches, Gann ABC detection, and trailing buy-stop execution."""
    data = attach_prior_day_high(df)
    rows = data.to_dicts()
    trades = []

    current_day = None
    daily_high_breached = False
    setup_complete = False
    post_breach_bars: list[dict] = []
    buy_stop_px: float | None = None
    position: dict | None = None

    for bar in rows:
        bar_date = bar["datetime"].date()
        bar_time = bar["datetime"].time()

        # Reset daily state at session boundary
        if bar_date != current_day:
            current_day = bar_date
            daily_high_breached = False
            setup_complete = False
            post_breach_bars = []
            buy_stop_px = None

        if bar["prev_day_high"] is None:
            continue

        # Manage open position
        if position is not None:
            stop_hit = bar["low"] <= position["stop"]
            target_hit = bar["high"] >= position["target"]
            eod_hit = cfg.enable_time_filter and bar_time >= cfg.trade_end

            if stop_hit or target_hit or eod_hit:
                exit_px = position["stop"] if stop_hit else (position["target"] if target_hit else bar["close"])
                exit_reason = "Stop" if stop_hit else ("Target" if target_hit else "SessionClose")
                pnl = (exit_px - position["entry"]) * cfg.point_value * cfg.contracts

                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": bar["datetime"],
                    "timeframe_min": position["tf_min"],
                    "entry_px": round(position["entry"], 2),
                    "exit_px": round(exit_px, 2),
                    "exit_reason": exit_reason,
                    "pnl_usd": round(pnl, 2),
                })
                position = None
                if pnl < 0 and cfg.continue_after_loss:
                    setup_complete = False
            continue

        if cfg.enable_time_filter and not (cfg.trade_start <= bar_time <= cfg.trade_end):
            buy_stop_px = None
            continue

        # Step 1: Detect prior-day high breach
        if not daily_high_breached and bar["high"] > bar["prev_day_high"]:
            daily_high_breached = True
            selected_tf = determine_timeframe(bar["datetime"], cfg)
            post_breach_bars = [bar]
            continue

        # Step 2: Reconstruct Gann legs and manage trailing buy-stop order
        if daily_high_breached and not setup_complete:
            # Check if working buy-stop filled on this bar
            if buy_stop_px is not None and bar["high"] >= buy_stop_px:
                entry_px = max(buy_stop_px, bar["open"])
                stop_px = bar["low"] - cfg.tick_size
                risk = max(entry_px - stop_px, cfg.tick_size)
                target_px = np.floor((entry_px + risk * cfg.profit_multiplier) / cfg.tick_size) * cfg.tick_size

                position = {
                    "entry": entry_px,
                    "stop": stop_px,
                    "target": target_px,
                    "entry_time": bar["datetime"],
                    "tf_min": selected_tf,
                }
                buy_stop_px = None
                setup_complete = True
                continue

            post_breach_bars.append(bar)
            legs = rebuild_gann_legs(post_breach_bars)
            pattern = find_valid_abc_pattern(legs, cfg.leg_c_max_ratio)

            if pattern is not None:
                _, _, leg_c = pattern
                candidate_stop = leg_c.low + cfg.tick_size
                # Trail the buy stop down as Leg C extends lower
                if buy_stop_px is None or candidate_stop < buy_stop_px:
                    buy_stop_px = candidate_stop
            else:
                buy_stop_px = None

    return pl.DataFrame(trades)


def generate_sample_ohlcv(n_days: int = 10, bars_per_day: int = 90, seed: int = 42) -> pl.DataFrame:
    """Generates multi-day intraday 5-minute OHLCV bars for standalone testing."""
    rng = np.random.default_rng(seed)
    records = []
    price = 21000.0

    for d in range(n_days):
        day_start = datetime(2025, 1, 2, 6, 30) + timedelta(days=d)
        for b in range(bars_per_day):
            ts = day_start + timedelta(minutes=5 * b)
            delta = rng.normal(loc=0.4, scale=6.0)
            open_px = price
            close_px = price + delta
            high_px = max(open_px, close_px) + rng.uniform(0.5, 5.0)
            low_px = min(open_px, close_px) - rng.uniform(0.5, 5.0)
            records.append({
                "datetime": ts,
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "volume": int(rng.integers(100, 2500)),
            })
            price = close_px

    return pl.DataFrame(records)


if __name__ == "__main__":
    sample_df = generate_sample_ohlcv()
    trade_log = run_gann_abc_backtest(sample_df)
    print(f"Simulated {len(trade_log)} Gann ABC trades:")
    print(trade_log.tail(10))
