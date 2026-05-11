"""
chain.py — yfinance option chain fetching and enrichment.

Fetches raw call/put DataFrames from yfinance, then cleans and
enriches them with derived columns so downstream modules (greeks,
charts) receive a consistent, ready-to-use DataFrame.

Public API:
    get_spot_price(ticker)                        -> float
    get_expirations(ticker)                       -> list[str]
    get_chain(ticker, expiration)                 -> tuple[pd.DataFrame, pd.DataFrame]
    enrich_chain(df, S, r, expiration_date)       -> pd.DataFrame
"""

from __future__ import annotations

import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

# Columns we keep from the raw yfinance payload, in display order.
_KEEP_COLS = [
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "volume",
    "openInterest",
    "impliedVolatility",
]


# ---------------------------------------------------------------------------
# Spot price
# ---------------------------------------------------------------------------

def get_spot_price(ticker: str) -> float:
    """Return the current spot price for the underlying.

    Tries fast_info.last_price first (works during market hours and
    reflects real-time data). Falls back to the most recent daily close
    from a 5-day history download when fast_info is unavailable (e.g.
    weekends, delisted tickers during testing).

    Parameters
    ----------
    ticker : str
        Exchange ticker symbol, e.g. "SPY".

    Returns
    -------
    float
        Last traded or most recent closing price.

    Raises
    ------
    ValueError
        If no price can be determined (unknown / invalid ticker).
    """
    t = yf.Ticker(ticker)

    # fast_info is a lightweight call that avoids a full info dict fetch.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        price = getattr(t.fast_info, "last_price", None)

    if price and not np.isnan(price):
        return float(price)

    # Fallback: pull the last close from recent history.
    hist = t.history(period="5d")
    if hist.empty:
        raise ValueError(
            f"Could not determine spot price for '{ticker}'. "
            "Check that the ticker is valid."
        )
    return float(hist["Close"].iloc[-1])


# ---------------------------------------------------------------------------
# Expirations
# ---------------------------------------------------------------------------

def get_expirations(ticker: str) -> list[str]:
    """Return available option expiration dates for the given ticker.

    Parameters
    ----------
    ticker : str
        Exchange ticker symbol.

    Returns
    -------
    list[str]
        ISO-8601 date strings (YYYY-MM-DD), sorted ascending.
        Only future expirations (>= today) are included.

    Raises
    ------
    ValueError
        If the ticker has no listed options or is unrecognised.
    """
    t = yf.Ticker(ticker)
    raw: tuple[str, ...] = t.options  # yfinance returns a tuple of strings

    if not raw:
        raise ValueError(
            f"No option expirations found for '{ticker}'. "
            "The ticker may be invalid or options may not be listed."
        )

    today = date.today()
    # yfinance already returns dates as 'YYYY-MM-DD' strings; filter stale ones.
    future = [
        d for d in raw
        if datetime.strptime(d, "%Y-%m-%d").date() >= today
    ]

    if not future:
        raise ValueError(
            f"All option expirations for '{ticker}' are in the past."
        )

    return sorted(future)


# ---------------------------------------------------------------------------
# Chain fetching
# ---------------------------------------------------------------------------

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Select, coerce, and fill the standard columns from a raw yfinance chain.

    - Keeps only _KEEP_COLS (adds any missing ones as NaN).
    - Coerces numeric columns; non-parseable values become NaN.
    - Fills volume / openInterest NaN → 0 (they are absent for illiquid strikes).
    - Clips impliedVolatility to [0, 20] to remove obviously bad values
      (yfinance sometimes returns 0.0 or >10 for deep OTM options).
    - Resets the index so callers get a clean 0-based integer index.
    """
    # Ensure all expected columns exist before selecting.
    for col in _KEEP_COLS:
        if col not in df.columns:
            df[col] = np.nan

    out = df[_KEEP_COLS].copy()

    # Coerce every column to numeric; silence the PerformanceWarning.
    for col in _KEEP_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Zero-fill count columns: missing = no activity, not unknown.
    out["volume"] = out["volume"].fillna(0).astype(int)
    out["openInterest"] = out["openInterest"].fillna(0).astype(int)

    # Clip IV: 0 and >20 (2000%) are data artifacts.
    out["impliedVolatility"] = out["impliedVolatility"].clip(lower=0.0, upper=20.0)
    out.loc[out["impliedVolatility"] == 0, "impliedVolatility"] = np.nan

    # Drop rows where the strike itself is missing (can't use them at all).
    out = out.dropna(subset=["strike"])

    return out.reset_index(drop=True)


def get_chain(
    ticker: str,
    expiration: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and clean calls and puts for a single expiration date.

    Parameters
    ----------
    ticker : str
        Exchange ticker symbol.
    expiration : str
        Expiration date as 'YYYY-MM-DD'. Must be a date returned by
        get_expirations(ticker).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (calls_df, puts_df), each with the columns defined in _KEEP_COLS:
        strike, lastPrice, bid, ask, volume, openInterest, impliedVolatility.

    Raises
    ------
    ValueError
        If yfinance cannot find a chain for the given ticker/expiration pair.
    """
    t = yf.Ticker(ticker)

    try:
        chain = t.option_chain(expiration)
    except Exception as exc:
        raise ValueError(
            f"Could not fetch option chain for '{ticker}' expiring {expiration}: {exc}"
        ) from exc

    calls = _clean(chain.calls)
    puts = _clean(chain.puts)

    if calls.empty and puts.empty:
        raise ValueError(
            f"Option chain for '{ticker}' on {expiration} returned no usable data."
        )

    return calls, puts


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def enrich_chain(
    df: pd.DataFrame,
    S: float,
    r: float,
    expiration_date: str,
) -> pd.DataFrame:
    """Attach derived analytics columns to a cleaned chain DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of get_chain (calls or puts).
    S : float
        Current spot price of the underlying.
    r : float
        Risk-free rate as a decimal (e.g. 0.05 for 5%).
    expiration_date : str
        Expiration date as 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        New DataFrame (input is not mutated) with these additional columns:

        mid          — (bid + ask) / 2; falls back to lastPrice when
                       bid/ask are both NaN (e.g. never traded).
        spread       — ask - bid (NaN when either leg is missing).
        moneyness    — S / strike; > 1 means ITM for a call, OTM for a put.
        dte          — calendar days to expiration from today (int).
        T            — dte / 365.0, annualized time to expiry (float).
    """
    out = df.copy()

    # mid: prefer (bid+ask)/2; fall back to lastPrice if both are absent.
    mid = (out["bid"] + out["ask"]) / 2
    fallback = out["lastPrice"]
    out["mid"] = np.where(mid.isna(), fallback, mid)

    out["spread"] = out["ask"] - out["bid"]

    out["moneyness"] = S / out["strike"]

    expiry = datetime.strptime(expiration_date, "%Y-%m-%d").date()
    dte = (expiry - date.today()).days
    out["dte"] = max(dte, 0)          # never negative; same for all rows
    out["T"] = out["dte"] / 365.0

    return out


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TICKER = "SPY"
    print(f"=== {TICKER} spot price ===")
    spot = get_spot_price(TICKER)
    print(f"  ${spot:.2f}\n")

    print(f"=== {TICKER} expirations (first 5) ===")
    exps = get_expirations(TICKER)
    for e in exps[:5]:
        print(f"  {e}")
    print(f"  … ({len(exps)} total)\n")

    expiry = exps[2]  # pick the third expiration for a meaningful chain
    print(f"=== Chain for {TICKER} — {expiry} ===")
    calls, puts = get_chain(TICKER, expiry)
    print(f"  Calls: {len(calls)} strikes  |  Puts: {len(puts)} strikes")
    print("\n  Calls (first 5 rows):")
    print(calls.head().to_string(index=False))
    print("\n  Puts (first 5 rows):")
    print(puts.head().to_string(index=False))

    print(f"\n=== Enriched calls (first 5 rows) ===")
    enriched = enrich_chain(calls, spot, 0.05, expiry)
    extra_cols = ["strike", "mid", "spread", "moneyness", "dte", "T"]
    print(enriched[extra_cols].head().to_string(index=False))

    # Spot-check: mid should be non-negative, DTE should be positive.
    assert (enriched["mid"].dropna() >= 0).all(), "Negative mid prices found"
    assert (enriched["dte"] >= 0).all(), "Negative DTE found"
    assert (enriched["T"] >= 0).all(), "Negative T found"
    print("\nSanity checks passed.")
