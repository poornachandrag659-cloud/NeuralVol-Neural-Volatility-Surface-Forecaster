# NeuralVol — Neural Volatility Surface Forecaster

NeuralVol is an educational quantitative-finance research project for **SPY**. It collects free options-chain snapshots, computes Black–Scholes implied volatility, builds an implied-volatility (IV) surface, and provides a chronological machine-learning workflow for next-snapshot surface forecasting.

> Research/education only. Options data from free providers may be delayed, incomplete, or subject to rate limits. This is not investment advice.

## Pipeline

1. **Collect** SPY option chains across available expiries using `yfinance`.
2. **Clean** quotes (positive bid/ask, non-crossed markets, spread guard).
3. **Invert** Black–Scholes prices to implied volatility.
4. **Build** a common log-moneyness × maturity grid.
5. **Visualize** the 3D surface, smile/skew, and term structure.
6. **Forecast** the next surface using a compact MLP and time-ordered evaluation.

## Quick start (Windows PowerShell)

Use Python 3.11 or 3.12 (standard, non-free-threaded build recommended):

```powershell
cd path\to\NeuralVol-GitHub-Ready
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

If `py -3.11` is unavailable, install standard 64-bit Python 3.11/3.12 from python.org, then recreate the virtual environment. On managed Windows devices, Application Control may block native NumPy/SciPy DLLs; if so, use Google Colab or an administrator-approved Python environment rather than disabling security controls.

## Commands

```powershell
neuralvol collect --ticker SPY
neuralvol build --ticker SPY
neuralvol visualize --ticker SPY
neuralvol train --ticker SPY
neuralvol forecast --ticker SPY
neuralvol report --ticker SPY
```

Run without installing the console script:

```powershell
python -m neuralvol.cli collect --ticker SPY
```

## Free data and forecasting expectations

- `collect` writes timestamped CSV files under `data/raw/`.
- `build` writes IV rows and a gridded surface under `data/processed/`.
- Each daily snapshot should be collected at a consistent time when possible.
- A one-day snapshot is enough to construct and visualize a surface, **not** enough to train a day-ahead model.
- `train --ticker SPY` requires at least 20 distinct surface dates and five usable consecutive-date pairs; it reports a chronological holdout metric. More observations (ideally 100+) are strongly preferred. The model and metadata are saved per ticker under `reports/`.
- The MLP predicts the next observed surface on a fixed grid. Forecast input includes the median calendar-day gap seen in the training pairs, so irregular snapshot spacing is represented consistently. It is a baseline forecasting experiment, not proof of tradable alpha.
- No future dates are shuffled into training. Scaling is fitted only on the training period.

## Outputs

| Output | Location |
|---|---|
| Raw option chain | `data/raw/` |
| IV observations | `data/processed/` |
| Surface grid CSV/NPZ | `data/processed/` |
| 3D HTML surface | `reports/` |
| Skew and term slices | `reports/` |
| Forecast metrics | `reports/forecast_metrics.json` |
| Trained model | `reports/` |

## Data quality and caveats

The code uses mid-quote when bid and ask are valid, otherwise the last traded price only when available. Quotes with invalid prices, crossed markets, very wide relative spreads, or prices outside no-arbitrage bounds are rejected. Risk-free rate is a configurable constant defaulting to 4%; replace it with a dated market curve for serious calibration. American-style SPY options are approximated with European Black–Scholes, so deep ITM/near-expiry values can be biased. Free Yahoo Finance data may omit quote sizes and timestamps.

## Research references

- Horvath, Muguruza & Tomas (2021), *Deep Learning Volatility*, Quantitative Finance: https://arxiv.org/abs/1901.09647
- Ackerer, Tagasovska & Vatter (2020), *Deep Smoothing of the Implied Volatility Surface*: https://arxiv.org/abs/1906.05065
- Companion code: https://github.com/amuguruza/NN-StochVol-Calibrations

## Repository checklist

- [ ] Collect multiple daily SPY snapshots.
- [ ] Review quote filters and risk-free rate assumptions.
- [ ] Run all tests.
- [ ] Generate visualizations and report.
- [ ] Add your measured forecast results (never copy example metrics).
- [ ] Push source, README, and selected small figures to GitHub; keep raw option-chain files out of Git.
