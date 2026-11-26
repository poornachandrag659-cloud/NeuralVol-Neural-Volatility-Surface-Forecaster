import json
import numpy as np
import pandas as pd
import pytest

from neuralvol.black_scholes import bs_price
from neuralvol.surface import calculate_iv, build_surface
from neuralvol.model import train_forecaster, forecast_latest


def test_calculate_iv_without_optional_last_price(tmp_path):
    snapshot = pd.Timestamp("2025-01-02T15:00:00")
    expiry = (snapshot + pd.Timedelta(days=90)).date().isoformat()
    price = bs_price(100, 100, 90 / 365, 0.04, 0.25, "call")
    raw = pd.DataFrame([{
        "snapshot_time": snapshot.isoformat(), "snapshot_date": str(snapshot.date()),
        "ticker": "SPY", "underlying_price": 100, "strike": 100,
        "expiration": expiry, "option_type": "call",
        "bid": price - 0.01, "ask": price + 0.01,
    }])
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    result = calculate_iv(raw_path, output_dir=tmp_path)
    iv = pd.read_csv(result)
    assert len(iv) == 1
    assert iv.iloc[0]["iv"] == pytest.approx(0.25, abs=2e-4)
    surface_csv, surface_npz = build_surface(result, output_dir=tmp_path)
    surface = pd.read_csv(surface_csv)
    assert len(surface) == 31 * 12
    assert np.isfinite(surface["iv"]).all()
    assert surface_npz.exists()


def test_train_and_forecast_end_to_end_for_ticker(tmp_path):
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    processed.mkdir()
    x_values = np.linspace(-0.1, 0.1, 5)
    t_values = np.linspace(0.1, 1.0, 5)
    for day in range(20):
        date = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=2 * day)).date()
        rows = []
        for x in x_values:
            for maturity in t_values:
                iv = 0.22 + 0.08 * x + 0.015 * maturity + 0.001 * day
                rows.append({
                    "snapshot_date": str(date), "ticker": "QQQ",
                    "log_moneyness": x, "maturity": maturity, "iv": iv,
                })
        pd.DataFrame(rows).to_csv(processed / f"QQQ_surface_{date:%Y%m%d}.csv", index=False)

    metrics = train_forecaster(
        processed_dir=processed, reports_dir=reports, min_dates=20, ticker="QQQ"
    )
    assert metrics["ticker"] == "QQQ"
    assert metrics["usable_pairs"] == 19
    metadata = json.loads((reports / "qqq_model_metadata.json").read_text())
    assert metadata["median_training_gap_days"] == 2.0

    forecast_path = forecast_latest(processed_dir=processed, reports_dir=reports, ticker="QQQ")
    forecast = pd.read_csv(forecast_path)
    assert len(forecast) == 25
    assert forecast["forecast_iv"].between(0.03, 3.0).all()
    assert set(forecast["snapshot_date"]) == {"2025-02-08"}
