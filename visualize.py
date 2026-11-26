"""Interactive 3D surface and skew/term-structure plots."""
from pathlib import Path
import numpy as np
import pandas as pd

def visualize_surface(surface_csv, reports_dir="reports"):
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    df = pd.read_csv(surface_csv)
    out = Path(reports_dir); out.mkdir(parents=True, exist_ok=True)
    ticker = str(df["ticker"].iloc[0]).upper() if "ticker" in df else "SPY"
    pivot = df.pivot_table(index="maturity", columns="log_moneyness", values="iv", aggfunc="mean").sort_index()
    x = pivot.columns.to_numpy(); t = pivot.index.to_numpy(); z = pivot.to_numpy()
    fig = go.Figure(data=[go.Surface(x=x, y=t, z=z, colorscale="Viridis", colorbar={"title":"IV"})])
    fig.update_layout(title=f"{ticker} Implied Volatility Surface", scene={
        "xaxis_title":"Log-moneyness ln(K/S)", "yaxis_title":"Maturity (years)",
        "zaxis_title":"Implied volatility"}, template="plotly_white")
    html = out/f"{ticker}_iv_surface.html"; fig.write_html(html, include_plotlyjs="cdn")
    # Skew at nearest 30-day maturity and term structure near ATM.
    atm_idx = int(np.argmin(np.abs(x)))
    t_idx = int(np.argmin(np.abs(t-30/365)))
    plt.figure(figsize=(8,4.5)); plt.plot(x,z[t_idx],marker="o")
    plt.xlabel("Log-moneyness ln(K/S)"); plt.ylabel("Implied volatility")
    plt.title(f"{ticker} volatility skew (maturity ≈ {t[t_idx]*365:.0f} days)")
    plt.grid(True,alpha=.3); plt.tight_layout()
    skew = out/f"{ticker}_skew.png"; plt.savefig(skew,dpi=160); plt.close()
    plt.figure(figsize=(8,4.5)); plt.plot(t*365,z[:,atm_idx],marker="o")
    plt.xlabel("Maturity (days)"); plt.ylabel("ATM implied volatility")
    plt.title(f"{ticker} ATM term structure"); plt.grid(True,alpha=.3)
    plt.tight_layout(); term = out/f"{ticker}_term_structure.png"; plt.savefig(term,dpi=160); plt.close()
    return html, skew, term
