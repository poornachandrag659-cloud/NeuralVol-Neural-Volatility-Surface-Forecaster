"""Command-line interface."""
import argparse
import json
from pathlib import Path
from .data import collect_options
from .surface import calculate_iv, build_surface
from .visualize import visualize_surface
from .model import train_forecaster, forecast_latest

def main():
    p = argparse.ArgumentParser(prog="neuralvol", description="Implied-volatility surface construction and forecasting research")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("collect","build","visualize","train","forecast","report"):
        s = sub.add_parser(name)
        s.add_argument("--ticker", default="SPY")
        if name == "collect": s.add_argument("--max-expiries", type=int, default=None)
    args = p.parse_args()
    ticker = args.ticker.upper()
    if args.command == "collect":
        print(collect_options(ticker, max_expiries=args.max_expiries))
    elif args.command == "build":
        raw = sorted(Path("data/raw").glob(f"{ticker}_options_*.csv"))
        if not raw: raise SystemExit("No raw option snapshot found. Run collect first.")
        iv = calculate_iv(raw[-1])
        surface_csv, surface_npz = build_surface(iv)
        print("IV:", iv); print("Surface CSV:", surface_csv); print("Surface NPZ:", surface_npz)
    elif args.command == "visualize":
        surfaces = sorted(Path("data/processed").glob(f"{ticker}_surface_*.csv"))
        if not surfaces: raise SystemExit("No surface CSV found. Run build first.")
        print(*visualize_surface(surfaces[-1]), sep="\n")
    elif args.command == "train":
        print(json.dumps(train_forecaster(ticker=ticker), indent=2))
    elif args.command == "forecast":
        print(forecast_latest(ticker=ticker))
    elif args.command == "report":
        metrics = Path("reports") / f"{ticker.lower()}_forecast_metrics.json"
        print(metrics.read_text(encoding="utf-8") if metrics.exists() else
              "No metrics yet. Collect multiple daily surfaces, then run train.")
if __name__ == "__main__":
    main()
