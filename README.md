# quant-strategy-research

Quantitative research and backtesting framework for rule-based price action strategies. Focuses on short-term momentum, volatility setups, and swing trading mechanics—including classic patterns from *Street Smarts* (Raschke & Connors), multi-bar reversal patterns, and Gann bar-counting methodologies.

## Research Architecture

```text
+-------------------------------------------------------+
|  1. RAW INTRADAY DATA (1-Minute OHLCV Bars)           |
|     - Historical index futures bar & tick exports     |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|  2. MULTI-TIMEFRAME FEATURE ENGINEERING (Polars)      |
|     - Dynamic bar aggregation (group_by_dynamic)      |
|     - Vectorized volatility (ATR) & trend (EMA) state |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|  3. STRATEGY & PATTERN SIGNAL MODULES                 |
|     - Multi-bar compression & breakout structures     |
|     - Momentum / volatility swing setups              |
|     - Gann swing & bar-counting mechanics             |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|  4. EXCURSION & STATISTICAL EVALUATION (MFE / MAE)    |
|     - Path-dependent favorable vs. adverse excursion  |
|     - Asymmetric payoff & volatility regime analysis  |
+-------------------------------------------------------+
```
## Strategy & Research Modules
* **`hoagie_stats_logger.py`** — Vectorized multi-timeframe inside-bar compression ("Hoagie") detector. Identifies consecutive inside bars nested within a higher-timeframe mother bar, filters by rolling ATR-14 and EMA-20 regime state, and logs path-dependent MFE/MAE excursion telemetry.
* **`holy_grail_strategy.py`** — Event-driven implementation of the Raschke/Connors ADX(14) + EMA(20) trend-pullback strategy, complete with rolling stop-order management, Point A/B swing targets, single re-entry logic, and daily loss/drawdown circuit breakers.
* **`qte_scalper_strategy.py`** — High-frequency micro-scalping module combining vectorized EMA(10) pullback triggers, multi-session time-window filters (with overnight session wraparound), and asymmetric tick-level OCO bracket execution.
