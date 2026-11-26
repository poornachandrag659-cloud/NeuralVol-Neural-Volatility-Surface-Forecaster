"""Quote filtering, IV calculation, and common-grid surface construction."""
from pathlib import Path
import numpy as np
import pandas as pd
from .black_scholes import implied_volatility

def calculate_iv(raw_csv, output_dir="data/processed", rate=0.04,
                 max_relative_spread=0.50, min_days=7, max_days=730):
    df = pd.read_csv(raw_csv)
    required = {"underlying_price","strike","expiration","option_type","bid","ask"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
    if "snapshot_time" not in df:
        df["snapshot_time"] = pd.Timestamp.now()
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")
    snap = df["snapshot_time"].dropna()
    valuation = snap.iloc[0].tz_localize(None).normalize() if len(snap) else pd.Timestamp.now().normalize()
    df["tau"] = (df["expiration"].dt.normalize() - valuation).dt.days / 365.0
    bid = pd.to_numeric(df["bid"], errors="coerce").fillna(0)
    ask = pd.to_numeric(df["ask"], errors="coerce").fillna(0)
    mid = (bid + ask) / 2
    spread_ratio = (ask - bid) / mid.replace(0, np.nan)
    valid_quote = (bid > 0) & (ask >= bid) & (mid > 0) & (spread_ratio <= max_relative_spread)
    last = pd.to_numeric(df["last_price"], errors="coerce").fillna(0) if "last_price" in df else pd.Series(0.0, index=df.index)
    df["market_price"] = np.where(valid_quote, mid, last)
    df["quote_source"] = np.where(valid_quote, "mid", "last")
    df["relative_spread"] = spread_ratio
    df = df[(df["tau"] >= min_days/365) & (df["tau"] <= max_days/365)]
    records = []
    for row in df.itertuples(index=False):
        price = float(row.market_price)
        spot = float(row.underlying_price)
        strike = float(row.strike)
        tau = float(row.tau)
        iv = implied_volatility(price, spot, strike, tau, rate, row.option_type)
        if iv is None or not np.isfinite(iv) or iv < 0.03 or iv > 3.0:
            continue
        records.append({
            "snapshot_date": str(getattr(row, "snapshot_date", valuation.date())),
            "snapshot_time": str(getattr(row, "snapshot_time", "")),
            "ticker": str(getattr(row, "ticker", "SPY")),
            "spot": spot, "strike": strike, "maturity": tau,
            "days_to_expiry": tau*365,
            "log_moneyness": np.log(strike/spot),
            "option_type": row.option_type, "market_price": price,
            "iv": iv, "quote_source": row.quote_source,
            "relative_spread": row.relative_spread,
        })
    ivdf = pd.DataFrame(records)
    if ivdf.empty:
        raise RuntimeError("No valid IVs. Check quote quality, date parsing, and filters.")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = Path(raw_csv).stem.replace("_options_", "_iv_")
    path = out / f"{stamp}.csv"
    ivdf.to_csv(path, index=False)
    return path

def build_surface(iv_csv, output_dir="data/processed", moneyness_points=31,
                 maturity_points=12, moneyness_min=-0.15, moneyness_max=0.15,
                 maturity_min=0.02, maturity_max=2.0):
    """Build a filled common grid using maturity-wise 1D interpolation."""
    df = pd.read_csv(iv_csv)
    if df.empty:
        raise ValueError("IV file is empty")
    # Calls/puts at similar coordinates are combined robustly by median.
    grouped = df.groupby(["maturity","log_moneyness"], as_index=False)["iv"].median()
    xgrid = np.linspace(moneyness_min, moneyness_max, moneyness_points)
    tgrid = np.linspace(maturity_min, maturity_max, maturity_points)
    matrix = np.full((len(tgrid), len(xgrid)), np.nan, dtype=float)
    for j, t in enumerate(tgrid):
        near = grouped[np.abs(grouped["maturity"]-t) <= max(0.04, maturity_max/maturity_points)]
        if len(near) < 2:
            # nearest expiry slice; interpolate only inside observed range
            nearest_t = grouped["maturity"].iloc[np.argmin(np.abs(grouped["maturity"].to_numpy()-t))]
            near = grouped[grouped["maturity"] == nearest_t]
        near = near.sort_values("log_moneyness").drop_duplicates("log_moneyness")
        xs = near["log_moneyness"].to_numpy()
        ys = near["iv"].to_numpy()
        if len(xs) >= 2:
            matrix[j] = np.interp(xgrid, xs, ys, left=ys[0], right=ys[-1])
        elif len(xs) == 1:
            matrix[j] = ys[0]
    # Keep matrix finite and flag cells filled by nearest available value.
    if not np.isfinite(matrix).all():
        finite = np.isfinite(matrix)
        if not finite.any():
            raise RuntimeError("Could not construct a finite surface grid")
        matrix[~finite] = float(np.nanmedian(matrix))
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    date = str(df["snapshot_date"].iloc[0]).replace("-", "") if "snapshot_date" in df else Path(iv_csv).stem[-8:]
    ticker = str(df["ticker"].iloc[0]) if "ticker" in df else "SPY"
    csv_path = out / f"{ticker}_surface_{date}.csv"
    rows = [{"snapshot_date":date, "ticker":ticker, "log_moneyness":float(x),
             "maturity":float(t), "iv":float(matrix[i,j])}
            for i,t in enumerate(tgrid) for j,x in enumerate(xgrid)]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    npz_path = out / f"{ticker}_surface_{date}.npz"
    np.savez_compressed(npz_path, moneyness=xgrid, maturity=tgrid,
                        iv_surface=matrix, underlying_price=float(df["spot"].median()))
    return csv_path, npz_path
