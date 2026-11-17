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
