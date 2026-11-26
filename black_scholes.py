"""Black-Scholes pricing and robust implied-volatility inversion (stdlib only)."""
from math import erf, exp, log, sqrt, isfinite

def _cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_price(spot: float, strike: float, maturity: float, rate: float,
             volatility: float, option_type: str = "call") -> float:
    """European Black-Scholes price. maturity and volatility must be positive."""
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if spot <= 0 or strike <= 0 or maturity <= 0 or volatility <= 0:
        raise ValueError("spot, strike, maturity and volatility must be positive")
    if not all(isfinite(v) for v in (spot, strike, maturity, rate, volatility)):
        raise ValueError("inputs must be finite")
    root_t = sqrt(maturity)
    d1 = (log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    if kind == "call":
        return spot * _cdf(d1) - strike * exp(-rate * maturity) * _cdf(d2)
    return strike * exp(-rate * maturity) * _cdf(-d2) - spot * _cdf(-d1)

def implied_volatility(price: float, spot: float, strike: float, maturity: float,
                      rate: float, option_type: str = "call",
                      low: float = 1e-6, high: float = 5.0,
                      tolerance: float = 1e-7, max_iterations: int = 200):
    """Bisection IV solver; returns None for prices outside no-arbitrage bounds."""
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if price <= 0 or spot <= 0 or strike <= 0 or maturity <= 0:
        return None
    disc_k = strike * exp(-rate * maturity)
    lower = max(spot - disc_k, 0.0) if kind == "call" else max(disc_k - spot, 0.0)
    upper = spot if kind == "call" else disc_k
    if price < lower - tolerance or price > upper + tolerance:
        return None
    lo, hi = low, high
    p_lo = bs_price(spot, strike, maturity, rate, lo, kind)
    p_hi = bs_price(spot, strike, maturity, rate, hi, kind)
    if price < p_lo - tolerance or price > p_hi + tolerance:
        return None
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        value = bs_price(spot, strike, maturity, rate, mid, kind)
        if abs(value - price) <= tolerance or hi - lo <= tolerance:
            return mid
        if value < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
