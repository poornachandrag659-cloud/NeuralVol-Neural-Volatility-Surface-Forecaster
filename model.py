"""Chronological next-surface MLP forecasting with leakage-safe scaling."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump, load

def _load_surface_files(processed_dir, ticker=None):
    pattern = f"{ticker.upper()}_surface_*.csv" if ticker else "*_surface_*.csv"
    paths = sorted(Path(processed_dir).glob(pattern))
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        required = {"snapshot_date","log_moneyness","maturity","iv"}
        if required.issubset(d.columns):
            frames.append(d[list(required)].copy())
    if not frames:
        raise FileNotFoundError(f"No *_surface_*.csv files with required columns in {processed_dir}")
    data = pd.concat(frames, ignore_index=True)
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")
    data = data.dropna(subset=["snapshot_date","log_moneyness","maturity","iv"])
    return data

def _make_pairs(data):
    dates = sorted(data["snapshot_date"].dt.normalize().unique())
    surfaces = {}
    for date in dates:
        d = data[data["snapshot_date"].dt.normalize() == date]
        surfaces[date] = d.groupby(["log_moneyness","maturity"])["iv"].median()
    pairs = []
    for prev, nxt in zip(dates[:-1], dates[1:]):
        common = surfaces[prev].index.intersection(surfaces[nxt].index)
        if len(common) < 20:
            continue
        a = surfaces[prev].loc[common].to_numpy()
        b = surfaces[nxt].loc[common].to_numpy()
        coords = np.asarray(list(common), dtype=float)
        X = np.column_stack([a, coords[:,0], coords[:,1], np.full(len(a), (pd.Timestamp(nxt)-pd.Timestamp(prev)).days)])
        pairs.append((prev, nxt, X, b))
    return pairs

def train_forecaster(processed_dir="data/processed", reports_dir="reports",
                     min_dates=20, random_state=42, ticker="SPY"):
    ticker = ticker.upper().strip()
    data = _load_surface_files(processed_dir, ticker=ticker)
    pairs = _make_pairs(data)
    dates = sorted(data["snapshot_date"].dt.normalize().unique())
    if len(dates) < min_dates or len(pairs) < 5:
        raise ValueError(
            f"Need >= {min_dates} distinct surface dates and >=5 usable day-pairs; "
            f"found {len(dates)} dates and {len(pairs)} pairs. Collect daily snapshots first."
        )
    # Hold out the final 20% of date-pairs, never random-shuffle dates.
    split = max(1, int(len(pairs)*0.8))
    train_pairs, test_pairs = pairs[:split], pairs[split:]
    X_train = np.vstack([p[2] for p in train_pairs]); y_train = np.concatenate([p[3] for p in train_pairs])
    if not test_pairs:
        raise ValueError("Not enough usable day-pairs for a chronological test holdout.")
    X_test = np.vstack([p[2] for p in test_pairs]); y_test = np.concatenate([p[3] for p in test_pairs])
    model = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(64,32), activation="relu", solver="adam",
        alpha=1e-3, learning_rate_init=1e-3, max_iter=600,
        early_stopping=False, random_state=random_state))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "ticker":ticker, "model":"MLPRegressor(64,32)",
        "distinct_dates":int(len(dates)), "usable_pairs":int(len(pairs)),
        "train_pairs":int(len(train_pairs)), "test_pairs":int(len(test_pairs)),
        "test_start":str(test_pairs[0][1].date()),
        "test_end":str(test_pairs[-1][1].date()),
        "mae_iv":float(mean_absolute_error(y_test,pred)),
        "rmse_iv":float(mean_squared_error(y_test,pred)**0.5),
        "baseline_persistence_mae_iv":float(mean_absolute_error(
            y_test, X_test[:,0])),
        "note":"Chronological holdout; metrics are not trading performance."
    }
    out = Path(reports_dir); out.mkdir(parents=True, exist_ok=True)
    dump(model, out/f"{ticker.lower()}_neuralvol_mlp.joblib")
    metadata = {"ticker": ticker, "median_training_gap_days": float(np.median([(pd.Timestamp(p[1])-pd.Timestamp(p[0])).days for p in train_pairs]))}
    (out/f"{ticker.lower()}_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out/f"{ticker.lower()}_forecast_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return metrics

def forecast_latest(processed_dir="data/processed", reports_dir="reports", ticker="SPY"):
    ticker = ticker.upper().strip()
    data = _load_surface_files(processed_dir, ticker=ticker)
    dates = sorted(data["snapshot_date"].dt.normalize().unique())
    latest = dates[-1]
    d = data[data["snapshot_date"].dt.normalize()==latest]
    surface = d.groupby(["log_moneyness","maturity"])["iv"].median().reset_index()
    out = Path(reports_dir)
    model_path = out/f"{ticker.lower()}_neuralvol_mlp.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model missing for {ticker}. Run neuralvol train --ticker {ticker} first.")
    model = load(model_path)
    metadata_path = out/f"{ticker.lower()}_model_metadata.json"
    gap_days = 1.0
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        gap_days = float(metadata.get("median_training_gap_days", 1.0))
    if not np.isfinite(gap_days) or gap_days <= 0:
        raise ValueError("Stored forecast interval must be a positive finite number of days.")
    X = np.column_stack([surface["iv"].to_numpy(),
                         surface["log_moneyness"].to_numpy(),
                         surface["maturity"].to_numpy(),
                         np.full(len(surface), gap_days)])
    surface["forecast_iv"] = np.clip(model.predict(X), 0.03, 3.0)
    surface["snapshot_date"] = str(pd.Timestamp(latest).date())
    out.mkdir(parents=True, exist_ok=True)
    path = out/f"{ticker}_next_surface_forecast.csv"
    surface.to_csv(path,index=False)
    return path
