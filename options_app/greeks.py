"""
greeks.py — Black-Scholes pricing and Greeks calculation.

Uses the closed-form Black-Scholes model for European options.
Scalar inputs only; vectorize at the call site with np.vectorize
or a DataFrame.apply if you need whole-chain coverage.

Public API:
    calculate_greeks(S, K, T, r, sigma, option_type) -> dict
    bs_price(S, K, T, r, sigma, option_type)         -> float
    implied_volatility(market_price, S, K, T, r, option_type) -> float
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    """Return (d1, d2) for the Black-Scholes formula.

    S     — spot price of the underlying
    K     — option strike price
    T     — time to expiry in years  (must be > 0)
    r     — continuously compounded risk-free rate (decimal, e.g. 0.05)
    sigma — annualized implied / historical volatility (decimal, e.g. 0.20)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Return the Black-Scholes theoretical price for a European option.

    Parameters
    ----------
    S, K, T, r, sigma : float
        See _d1_d2 for definitions.
    option_type : {'call', 'put'}

    Returns
    -------
    float
        Theoretical option price (same currency units as S and K).

    Raises
    ------
    ValueError
        If T <= 0 or option_type is not 'call'/'put'.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    discount = np.exp(-r * T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * discount * norm.cdf(d2)
    else:
        return K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    """Compute Black-Scholes price and all first-order Greeks.

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to expiry in years (e.g. 30 days → 30/365).
    r : float
        Continuously compounded risk-free rate as a decimal (e.g. 0.05).
    sigma : float
        Annualized volatility as a decimal (e.g. 0.20 for 20%).
    option_type : {'call', 'put'}
        Type of the option contract.

    Returns
    -------
    dict with keys:
        price  — Black-Scholes fair value
        delta  — rate of change of price w.r.t. S  (call: 0–1, put: -1–0)
        gamma  — rate of change of delta w.r.t. S  (always positive)
        theta  — price decay per calendar day  (almost always negative)
        vega   — price change per +1 percentage-point move in vol
        rho    — price change per +1 percentage-point move in rate

    Convention notes
    ----------------
    * Theta is divided by 365 (calendar days), not 252, matching most
      retail broker displays.
    * Vega is divided by 100 so it reads as "$ change per 1-vol-point",
      matching the Bloomberg / Thinkorswim convention.
    * Rho is divided by 100 for the same reason.

    Raises
    ------
    ValueError
        If T <= 0 or option_type is unrecognized.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    discount = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)  # standard normal PDF at d1 (shared by gamma/vega/theta)
    sqrt_T = np.sqrt(T)

    # --- price ---
    price = bs_price(S, K, T, r, sigma, option_type)

    # --- delta ---
    # Call: N(d1)   Put: N(d1) - 1
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # --- gamma ---
    # Identical for calls and puts; measures delta's curvature w.r.t. S.
    gamma = pdf_d1 / (S * sigma * sqrt_T)

    # --- theta ---
    # Rate of time decay; the two option types differ only in the
    # signed N(d2) term that captures the carry on the strike.
    common_theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T)
    if option_type == "call":
        theta = (common_theta - r * K * discount * norm.cdf(d2)) / 365
    else:
        theta = (common_theta + r * K * discount * norm.cdf(-d2)) / 365

    # --- vega ---
    # Sensitivity to a 1% (0.01) move in vol; divide by 100 for the
    # per-vol-point convention used by most retail platforms.
    vega = S * sqrt_T * pdf_d1 / 100

    # --- rho ---
    # Sensitivity to a 1% (0.01) move in the risk-free rate.
    if option_type == "call":
        rho = K * T * discount * norm.cdf(d2) / 100
    else:
        rho = -K * T * discount * norm.cdf(-d2) / 100

    return {
        "price": round(price, 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega, 4),
        "rho":   round(rho, 4),
    }


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
) -> float:
    """Invert bs_price via Brent's method to recover implied volatility.

    Parameters
    ----------
    market_price : float
        Observed mid-price of the option in the market.
    S, K, T, r : float
        See bs_price.
    option_type : {'call', 'put'}
    tol : float
        Convergence tolerance passed to brentq (default 1e-6).

    Returns
    -------
    float
        Implied volatility as a decimal (e.g. 0.25 for 25%).
        Returns np.nan when the solver cannot converge — this happens for
        deep ITM/OTM options whose market prices violate no-arbitrage bounds
        or when bid/ask spreads produce an infeasible input.
    """
    if T <= 0:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        return brentq(objective, a=1e-4, b=20.0, xtol=tol)
    except (ValueError, RuntimeError):
        # brentq raises ValueError when f(a) and f(b) have the same sign,
        # which means market_price is outside the model's reachable range.
        return np.nan


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ATM call: SPY-like inputs, 30 DTE, 20% vol, 5% rate
    S, K, T, r, sigma = 500.0, 500.0, 30 / 365, 0.05, 0.20

    print("=== ATM Call ===")
    call = calculate_greeks(S, K, T, r, sigma, "call")
    for k, v in call.items():
        print(f"  {k:>6}: {v}")

    print("\n=== ATM Put ===")
    put = calculate_greeks(S, K, T, r, sigma, "put")
    for k, v in put.items():
        print(f"  {k:>6}: {v}")

    # Put-call parity check: call - put ≈ S - K·e^(-rT)
    parity = S - K * np.exp(-r * T)
    diff = call["price"] - put["price"]
    print(f"\nPut-call parity — expected: {parity:.4f}, got: {diff:.4f}")
    assert abs(diff - parity) < 0.01, "Put-call parity violated!"
    print("Put-call parity check passed.")

    # IV round-trip: recover sigma from the theoretical price
    iv = implied_volatility(call["price"], S, K, T, r, "call")
    print(f"\nIV round-trip — input sigma: {sigma:.4f}, recovered IV: {iv:.4f}")
    assert abs(iv - sigma) < 1e-4, "IV round-trip failed!"
    print("IV round-trip check passed.")

    # OTM put (deep OTM → IV should still converge)
    otm_put = calculate_greeks(S, 450.0, T, r, sigma, "put")
    print(f"\n=== OTM Put (K=450) ===")
    for k, v in otm_put.items():
        print(f"  {k:>6}: {v}")
