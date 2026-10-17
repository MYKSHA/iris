# Multi-strategy intraday backtester

Three modular strategies, run on their own or together, with one shared rule:
**only one position is open at a time across all modules**.

| Module | Key | Market | Signal TF |
|---|---|---|---|
| Govardhan Gajjala VCP + Momentum | `vcp` | NSE stocks, 9:15–15:30 IST | 5 / 15 min |
| Fabio Valentini ORB + Order Flow | `orb` | CME NQ futures, 9:30–15:30 ET | 5 min |
| Gajjala Horizontal Fade / Box Breakout | `box` | NSE stocks, 9:15–15:30 IST | 5 / 15 min |

```bash
cd trading
pip install -r requirements.txt

# 1. get data (or drop TradingView CSV exports into data/)
python -m algo.fetch --market nse    --interval 5min CDSL.NS KPITTECH.NS
python -m algo.fetch --market cme_nq --interval 1min NQ=F

# 2. edit config.toml (symbols, market caps, enabled modules)
# 3. backtest: the combined run plus each module on its own
python -m algo --config config.toml --separate
python -m algo --modules orb -v        # one module, with every entry/exit printed to the console
python -m pytest -q tests
```

Each run writes `results/<timestamp>/<run>/` containing `trades.csv`, `equity.csv` and
`report.json`, plus `backtest.log` and a copy of the config it ran with (`config.toml`) at the run root.

## Config (`config.toml`)

- `[modules.<key>].enabled` turns a module on or off. `--modules vcp,box` overrides it for one run.
- `[engine].priority` decides which module gets the trade when several signal on the same bar.
- `[engine].intrabar` decides what happens when one bar touches both the stop and the target. `stop_first` (the default) assumes the stop filled first. `open_proximity` fills whichever level is nearer the bar's open.
- `[markets.*]` holds the session hours, tick size, point value, FX rate to the base currency, commission and slippage for each market.
- If the config breaks the spec, loading fails. That covers `risk_pct` above 2 and `min_contractions`/`max_contractions` outside 3–4. Values outside the other spec ranges load with a warning: TP 8–13%, stop 5–8 ticks, target 10–15 ticks, position 10–20%.

## Rules as implemented

**VCP (`algo/strategies/vcp.py`)**
- **Universe:** market cap must be between ₹1000 and ₹5000 Cr. Set it in `market_caps_cr`, or set `market_cap_source = "yfinance"` to look it up. The 20-day average daily volume must be above 500k, measured on prior sessions only.
- **Momentum:** as of the prior session close, the stock must be up between +40% and +50% over a lookback of 20 to 25 trading days (any lookback in that window counts). Set `momentum_max_pct` very high to drop the 50% ceiling.
- **Trend:** close > EMA9 > EMA21 on the signal timeframe.
- **VCP pattern:**
  - The base must contain 3–4 successive pullbacks. A pullback runs from a swing high to the swing low after it.
  - Each pullback's depth % must be 50–70% of the depth of the one before it.
  - Mean volume must fall on each successive pullback.
  - The highs must slope down. The trendline joins the first and last contraction highs, and the highs in between may sit no more than `trendline_tolerance_pct` above it.
  - Swings are fractal pivots. A pivot only counts once `pivot_bars` bars have closed after it, so there is no lookahead.
- **Entry:** the bar closes above the trendline on volume above 1.5× the SMA of the previous 20 bars. It must come within `breakout_window_bars` bars of the first close above the line. Only one entry is taken per base.
- **Stop:** below the last swing low of the base (`stop_buffer_pct` sets the gap). The stop moves to break-even at +1R.
- **Exits:** a close below EMA9, or take-profit at `take_profit_pct` (10% by default; the spec range is 8–13%).
- **Sizing:** risk at most 2% of equity per trade, and keep the position at no more than 20% of equity. The trade is skipped if it would come out below 10%.

**ORB (`algo/strategies/orb.py`)**
- **Opening range:** the high and low of the 9:30–10:00 ET bars.
- **POC:** taken from the volume profile of the opening range. The profile is built from real trades when `data.source = "ticks"`. Otherwise each bar's volume is spread evenly across its high–low range.
- **Delta:** real delta when the data has a `delta` column (built from ticks by `orderflow.ticks_to_bars`). Otherwise it is estimated as `((close-open)/range) * volume`. Cumulative delta resets at 9:30 each session.
- **Long:** a fresh close above the OR high, with cumulative delta > 0 and POC below price. **Short** is the mirror image.
- **Stop and target:** the stop is 6 ticks beyond the ORB level (`stop_ticks`, spec range 5–8). The target is 12 ticks from entry (`target_ticks`, spec range 10–15).
- **Time stop:** exit if the trade has not moved +2 ticks within 3 minutes.
- **Hours:** entries only from 9:31 ET (the first 60 seconds are skipped) until 15:30. Positions are flattened at 15:30.
- **1-minute data:** load 1-min base bars (`data.interval = "1min"`) and keep `signal_timeframe = "5min"`. With 5-min base bars the 3-minute time stop can only be checked once the next 5-min bar closes. A 1.5-point stop and a 3-point target are also often both inside one 5-min bar.

**Box (`algo/strategies/box.py`)**
- **Box:** the longest stretch of 10–30 bars whose high–low range is 1–3%. Its mean volume must be below the 20-bar SMA volume measured just before the box started.
- **Entry:** a close above the box high on volume above 1.5× the 20-bar SMA volume.
- **Stop:** `stop_mode` picks the stop. Options are `box_low`, `ema21`, `tighter` (the nearer of the two) and `wider`.
- **Exits:** the spec gives no exit for this module. The defaults are a 2R target (`take_profit_r`, from the 2:1 baseline), break-even at +1R and an end-of-day flatten. All can be configured.

**Engine (`algo/engine.py`)**
- **Fills:** entries fill at the signal bar's close plus slippage.
- **Stops:** a gap through the stop fills at the open. Otherwise the stop fills at the stop price minus slippage.
- **Targets:** fill at the target price with no slippage.
- **Timeline:** all feeds share one UTC timeline, so NSE and CME sessions can run in one account. The FX rate converts results to the base currency.
- **Trade log columns:** strategy, symbol, side, entry/exit time and price, qty, initial and final stop, target, exit reason, P&L, R-multiple, return %, WIN/LOSS, MFE/MAE in R, bars held, equity before and after, and drawdown % at exit.

## Metrics and robustness

- **Metrics:** each run reports win rate, profit factor, average, median, win and loss R, realised R:R, total return, max drawdown on the mark-to-market equity curve, daily Sharpe (annualised √252), longest losing streak and a count of exit reasons.
- **Baseline check:** compares the run with the spec baseline of about 30% win rate and 2:1+ R:R. It needs at least 30 trades to give a verdict.
- **Monte Carlo** (`[montecarlo]`):
  - `bootstrap` resamples per-trade returns with replacement.
  - `shuffle` reorders the same trades, which tests path and drawdown risk.
  - It reports p5/p50/p95 final return, the drawdown distribution, P(loss) and P(drawdown > `ruin_drawdown_pct`).

## Data

- **CSV:** `data/<SYMBOL>.csv`, with `=` replaced by `_` and `^` dropped (`NQ=F` becomes `data/NQ_F.csv`). Columns are `time,open,high,low,close,volume[,delta]`. `time` can be Unix seconds (the TradingView "Export chart data" format), ISO with an offset, or naive local time.
- **Ticks:** `data/ticks/<SYMBOL>.csv` with `time,price,size[,side]`. `side` is the aggressor, `B`/`S` or ±1; without it the tick rule is used. Export these from Rithmic, CQG or IB time & sales to get real delta and a trade-based POC.
- **Yahoo limits:** Yahoo only serves the last 60 days of 5-min bars and the last 7 days of 1-min bars. `algo.fetch` merges each download into the existing CSV, so running it on a schedule builds up the 6–12 months of history needed for testing. For a 6–12 month test right away, use a TradingView export or a paid feed.
- **Synthetic:** `data.source = "synthetic"` generates random-walk sessions for smoke-testing the pipeline only.
