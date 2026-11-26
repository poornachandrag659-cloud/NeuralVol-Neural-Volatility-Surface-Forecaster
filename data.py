"""Free SPY option-chain collection using yfinance."""
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

def collect_options(ticker="SPY", output_dir="data/raw", max_expiries=None):
    """Download option chain snapshot; returns output CSV path."""
    import yfinance as yf  # lazy import keeps core pricing/tests lightweight
    symbol = ticker.upper().strip()
    instrument = yf.Ticker(symbol)
    spot_hist = instrument.history(period="5d", auto_adjust=False)
    if spot_hist.empty:
        raise RuntimeError(f"No underlying price history returned for {symbol}")
    spot = float(spot_hist["Close"].dropna().iloc[-1])
    expiries = list(instrument.options)
    if max_expiries:
        expiries = expiries[:int(max_expiries)]
    if not expiries:
        raise RuntimeError(f"No option expiries returned for {symbol}")
    stamp = datetime.now(timezone.utc).astimezone()
    rows = []
    for expiry in expiries:
        chain = instrument.option_chain(expiry)
        expiry_date = pd.Timestamp(expiry)
        for kind, frame in (("call", chain.calls), ("put", chain.puts)):
            if frame is None or frame.empty:
                continue
            for _, q in frame.iterrows():
                rows.append({
                    "snapshot_time": stamp.isoformat(),
                    "snapshot_date": stamp.date().isoformat(),
                    "ticker": symbol,
                    "underlying_price": spot,
                    "expiration": expiry,
                    "option_type": kind,
                    "strike": float(q.get("strike", float("nan"))),
                    "bid": float(q.get("bid", 0) or 0),
                    "ask": float(q.get("ask", 0) or 0),
                    "last_price": float(q.get("lastPrice", 0) or 0),
                    "volume": int(q.get("volume", 0) or 0),
                    "open_interest": int(q.get("openInterest", 0) or 0),
                    "days_to_expiry": (expiry_date - pd.Timestamp(stamp.date())).days,
                })
    if not rows:
        raise RuntimeError("Downloaded chains contained no option rows")
    df = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{symbol}_options_{stamp.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(path, index=False)
    return path
