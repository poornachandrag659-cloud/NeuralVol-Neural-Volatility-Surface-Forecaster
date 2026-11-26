import pytest
from neuralvol.black_scholes import bs_price, implied_volatility

@pytest.mark.parametrize("kind", ["call","put"])
def test_iv_round_trip(kind):
    s,k,t,r,v = 100.0,105.0,0.75,0.03,0.27
    price = bs_price(s,k,t,r,v,kind)
    recovered = implied_volatility(price,s,k,t,r,kind)
    assert recovered == pytest.approx(v, abs=2e-5)

def test_invalid_option_type():
    with pytest.raises(ValueError):
        bs_price(100,100,1,.03,.2,"other")

def test_invalid_price_returns_none():
    assert implied_volatility(-1,100,100,1,.03,"call") is None


def test_call_price_is_above_intrinsic_value():
    price = bs_price(100, 95, 0.5, 0.02, 0.25, "call")
    assert price >= max(100 - 95 * __import__("math").exp(-0.02 * 0.5), 0)
