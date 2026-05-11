"""
app.py — Streamlit entry point for the Options Analytics app.

Run with:
    streamlit run options_app/app.py

Layout:
    Sidebar  — ticker input only (global)
    Tab 1    — Chain:  full calls + puts table for the nearest expiry
    Tab 2    — Greeks: inline expiry / option-type / strike controls + metric cards
    Tab 3    — Payoff: inline expiry / option-type / strike / position / rate + diagram

Session-state key namespacing:
    greeks_*   — isolated to the Greeks tab
    payoff_*   — isolated to the Payoff tab
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import streamlit as st

import chain as ch
import charts
import greeks as gr
from charts import _breakeven

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Options Analytics",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "ticker": "SPY",
    # Greeks tab — contract selectors
    "greeks_expiry":         None,
    "greeks_option_type":    "call",
    "greeks_strike":         None,
    # Greeks tab — scenario sliders (reset when contract changes)
    "greeks_last_contract":  None,
    "greeks_slider_S":       0.0,
    "greeks_slider_dte":     30,
    "greeks_slider_iv":      20.0,
    # Payoff tab
    "payoff_expiry":       None,
    "payoff_option_type":  "call",
    "payoff_strike":       None,
    "payoff_position":     "long",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def _load_expirations(ticker: str) -> list[str]:
    return ch.get_expirations(ticker)


@st.cache_data(ttl=60, show_spinner=False)
def _load_spot(ticker: str) -> float:
    return ch.get_spot_price(ticker)


@st.cache_data(ttl=60, show_spinner=False)
def _load_chain(ticker: str, expiration: str) -> tuple:
    """Return (calls_df, puts_df, spot) — enriched, r-independent."""
    calls_raw, puts_raw = ch.get_chain(ticker, expiration)
    spot = ch.get_spot_price(ticker)
    calls = ch.enrich_chain(calls_raw, spot, r=0.0, expiration_date=expiration)
    puts  = ch.enrich_chain(puts_raw,  spot, r=0.0, expiration_date=expiration)
    return calls, puts, spot

# ---------------------------------------------------------------------------
# Sidebar — ticker only
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📈 Options Analytics")
    st.divider()

    raw_input = st.text_input(
        "Ticker symbol",
        value=st.session_state.ticker,
        key="_ticker_raw",
    ).upper().strip()

    if raw_input and raw_input != st.session_state.ticker:
        st.session_state.ticker = raw_input
        # Reset all tab-specific state so stale strikes / expiries are cleared.
        for _k in ("greeks_expiry", "greeks_strike", "payoff_expiry", "payoff_strike"):
            st.session_state[_k] = None
        st.rerun()

    st.caption("Data: Yahoo Finance · 60 s cache")

ticker: str = st.session_state.ticker

# ---------------------------------------------------------------------------
# Load expirations (needed by all three tabs)
# ---------------------------------------------------------------------------

expirations: list[str] = []
exp_error: str | None = None

with st.spinner(f"Loading {ticker} expirations…"):
    try:
        expirations = _load_expirations(ticker)
    except ValueError as e:
        exp_error = str(e)

# Spot price (lightweight; used in header and per-tab calculations)
spot: float | None = None
if not exp_error:
    try:
        spot = _load_spot(ticker)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title(f"{ticker}")

if exp_error:
    st.error(exp_error)
    st.stop()

if spot:
    st.metric("Spot price", f"${spot:,.2f}")

st.divider()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DISPLAY_COLS = [
    "strike", "lastPrice", "bid", "ask", "mid",
    "spread", "volume", "openInterest", "impliedVolatility", "moneyness",
]

_FORMAT = {
    "strike":            "${:,.2f}",
    "lastPrice":         "${:.2f}",
    "bid":               "${:.2f}",
    "ask":               "${:.2f}",
    "mid":               "${:.2f}",
    "spread":            "${:.2f}",
    "volume":            "{:,.0f}",
    "openInterest":      "{:,.0f}",
    "impliedVolatility": "{:.1%}",
    "moneyness":         "{:.3f}",
}


def _render_chain_table(df, highlight_strike=None, height=480):
    """Style and render one side of the chain as a Streamlit dataframe."""
    if df is None or df.empty:
        st.info("No data.")
        return

    # Only keep columns that are actually present.
    cols = [c for c in _DISPLAY_COLS if c in df.columns]
    fmt  = {k: v for k, v in _FORMAT.items() if k in cols}

    styled = df[cols].style.format(fmt, na_rep="—")

    if highlight_strike is not None:
        def _highlight(row):
            color = "background-color: #1e3a5f" if row["strike"] == highlight_strike else ""
            return [color] * len(row)
        styled = styled.apply(_highlight, axis=1)

    st.dataframe(styled, use_container_width=True, height=height)


def _atm_strike(strikes: list[float], spot: float | None) -> float | None:
    if not strikes or spot is None:
        return None
    return min(strikes, key=lambda k: abs(k - spot))


def _expiry_selector(label: str, state_key: str, options: list[str]) -> str | None:
    """Render a selectbox that persists its value in session_state[state_key]."""
    if not options:
        return None
    current = st.session_state[state_key]
    idx = options.index(current) if current in options else 0
    chosen = st.selectbox(label, options, index=idx, key=f"_widget_{state_key}")
    st.session_state[state_key] = chosen
    return chosen


def _strike_selector(
    label: str,
    state_key: str,
    ticker: str,
    expiry: str | None,
    option_type: str,
) -> float | None:
    """Load strikes for the given expiry/type and render a selectbox."""
    if expiry is None:
        return None

    try:
        calls, puts, _ = _load_chain(ticker, expiry)
    except Exception:
        return None

    df = calls if option_type == "call" else puts
    if df is None or df.empty:
        return None

    strikes = sorted(df["strike"].dropna().unique().tolist())
    if not strikes:
        return None

    current = st.session_state[state_key]
    if current not in strikes:
        current = _atm_strike(strikes, spot)
        st.session_state[state_key] = current

    idx = strikes.index(current) if current in strikes else 0
    chosen = st.selectbox(
        label, strikes, index=idx,
        format_func=lambda x: f"${x:,.2f}",
        key=f"_widget_{state_key}",
    )
    st.session_state[state_key] = chosen
    return chosen


# ---------------------------------------------------------------------------
# Greek plain-English interpreter
# ---------------------------------------------------------------------------

def _greek_plain_english(result: dict, tkr: str, opt_type: str, S: float, K: float) -> None:
    """Render one st.info() box per Greek with a plain-English interpretation.

    Assumes 1 contract = 100 shares throughout.
    Dollar signs are escaped with \\$ to prevent Streamlit treating them as
    LaTeX math delimiters.
    """
    SHARES = 100
    delta = result["delta"]
    gamma = result["gamma"]
    theta = result["theta"]
    vega  = result["vega"]
    rho   = result["rho"]

    def _d(v: float) -> str:
        """Format a dollar amount, escaping $ for Streamlit markdown."""
        return f"\\${abs(v):.2f}"

    # --- Delta ---
    delta_contract = abs(delta) * SHARES
    pnl_dir = "gain" if delta > 0 else "lose"
    exposure = "same direction as" if opt_type == "call" else "opposite direction to"
    st.info(
        f"**Delta** — A \\$1 move up in {tkr} would **{pnl_dir} {_d(delta_contract)}** "
        f"on this 1-contract position. Delta of {delta:.4f} means the option moves "
        f"in the {exposure} the stock, at {abs(delta):.1%} of a full share."
    )

    # --- Gamma ---
    gamma_contract = gamma * SHARES
    atm = abs(S - K) / K < 0.05
    gamma_context = (
        f"highest near the strike — small moves in {tkr} around {_d(K)} accelerate P&L quickly"
        if atm else
        f"lower here since the option is away from the money; delta responds more slowly to price moves"
    )
    st.info(
        f"**Gamma** — For every extra \\$1 move in {tkr}, delta shifts by **{gamma:.4f}** "
        f"({_d(gamma_contract)} per contract). Gamma is {gamma_context}."
    )

    # --- Theta ---
    theta_contract = theta * SHARES
    decay_verb = "loses" if theta_contract < 0 else "gains"
    weekly = abs(theta_contract) * 5
    st.info(
        f"**Theta** — This position **{decay_verb} {_d(theta_contract)} per calendar day** "
        f"from time decay, all else equal. Held over a 5-trading-day week, "
        f"that is {_d(weekly)} of decay."
    )

    # --- Vega ---
    vega_contract = vega * SHARES
    vega_dir = "adds" if vega_contract > 0 else "subtracts"
    vega_5pt = abs(vega_contract) * 5
    st.info(
        f"**Vega** — A 1 vol-point (1%) rise in implied volatility **{vega_dir} "
        f"{_d(vega_contract)}** to this contract's value. A sudden 5-point vol "
        f"spike would {'increase' if vega_contract > 0 else 'decrease'} value by {_d(vega_5pt)}."
    )

    # --- Rho ---
    rho_contract = rho * SHARES
    rho_dir  = "adds" if rho_contract > 0 else "subtracts"
    rate_why = (
        "calls benefit as higher rates increase the cost of carrying the underlying"
        if opt_type == "call" else
        "puts are hurt as higher rates reduce the present value of the strike payoff"
    )
    st.info(
        f"**Rho** — A 1 percentage-point rise in the risk-free rate **{rho_dir} "
        f"{_d(rho_contract)}** to value. {rate_why.capitalize()}."
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chain, tab_greeks, tab_payoff = st.tabs(["Chain", "Greeks", "Payoff"])

# ===========================================================================
# Tab 1 — Chain
# No controls. Shows calls + puts for the nearest available expiry.
# ===========================================================================

with tab_chain:
    if not expirations:
        st.info("No expirations available for this ticker.")
    else:
        nearest_expiry = expirations[0]

        chain_error: str | None = None
        calls_df = puts_df = None

        with st.spinner(f"Loading chain for {nearest_expiry}…"):
            try:
                calls_df, puts_df, _ = _load_chain(ticker, nearest_expiry)
            except Exception as e:
                chain_error = str(e)

        st.caption(f"Nearest expiry: **{nearest_expiry}** · {calls_df['dte'].iloc[0] if calls_df is not None and not calls_df.empty else '—'} DTE")

        if chain_error:
            st.error(chain_error)
        else:
            col_calls, col_puts = st.columns(2)
            with col_calls:
                st.markdown("**Calls**")
                _render_chain_table(calls_df)
            with col_puts:
                st.markdown("**Puts**")
                _render_chain_table(puts_df)

# ===========================================================================
# Tab 2 — Greeks
# Inline controls: expiry · option type · strike
# ===========================================================================

with tab_greeks:
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 2])

    with ctrl1:
        g_expiry = _expiry_selector("Expiration", "greeks_expiry", expirations)

    with ctrl2:
        g_opt_type = st.radio(
            "Type", ["call", "put"],
            index=0 if st.session_state.greeks_option_type == "call" else 1,
            horizontal=True,
            key="_greeks_opt_radio",
        )
        if g_opt_type != st.session_state.greeks_option_type:
            st.session_state.greeks_option_type = g_opt_type
            st.session_state.greeks_strike = None   # reset strike on type change

    with ctrl3:
        g_strike = _strike_selector(
            "Strike", "greeks_strike", ticker, g_expiry, g_opt_type
        )

    st.divider()

    if g_expiry is None or g_strike is None:
        st.info("Select an expiration and strike above to view Greeks.")
    else:
        g_calls, g_puts, g_spot = _load_chain(ticker, g_expiry)
        g_chain = g_calls if g_opt_type == "call" else g_puts

        row = g_chain[g_chain["strike"] == g_strike]
        if row.empty:
            st.warning("No data for the selected contract.")
        else:
            row = row.iloc[0]

            # --- Derive chain defaults for the three sliders ---
            chain_dte  = max(int(row.get("dte", 1)), 1)
            chain_iv   = row.get("impliedVolatility", np.nan)
            chain_iv   = float(chain_iv) if (chain_iv and not np.isnan(float(chain_iv))) else 0.20

            # --- Three parameter sliders ---
            # Each resets to its chain default when the contract changes.
            # We detect a contract change by comparing a stored "last contract"
            # key against the current (ticker, expiry, strike, option_type) tuple.
            contract_id = (ticker, g_expiry, g_strike, g_opt_type)
            if st.session_state.get("greeks_last_contract") != contract_id:
                st.session_state["greeks_last_contract"] = contract_id
                st.session_state["greeks_slider_S"]   = float(g_spot)
                st.session_state["greeks_slider_dte"] = max(chain_dte, 1)
                # Clamp IV to slider range [1, 150] before storing.
                st.session_state["greeks_slider_iv"]  = float(
                    np.clip(round(chain_iv * 100, 1), 1.0, 150.0)
                )

            sl1, sl2, sl3 = st.columns(3)

            with sl1:
                S_lo = round(g_spot * 0.50, 2)
                S_hi = round(g_spot * 1.50, 2)
                # Clamp stored value to current range in case spot moved.
                st.session_state["greeks_slider_S"] = float(
                    np.clip(st.session_state["greeks_slider_S"], S_lo, S_hi)
                )
                g_S = st.slider(
                    "Underlying price (S)",
                    min_value=S_lo,
                    max_value=S_hi,
                    step=round((S_hi - S_lo) / 200, 2),
                    format="$%.2f",
                    key="greeks_slider_S",
                )

            with sl2:
                g_dte = st.slider(
                    "Days to expiration (DTE)",
                    min_value=1,
                    max_value=365,
                    step=1,
                    format="%d d",
                    key="greeks_slider_dte",
                )

            with sl3:
                st.session_state["greeks_slider_iv"] = float(
                    np.clip(st.session_state["greeks_slider_iv"], 1.0, 150.0)
                )
                g_iv_pct = st.slider(
                    "Implied volatility (%)",
                    min_value=1.0,
                    max_value=150.0,
                    step=0.5,
                    format="%.1f%%",
                    key="greeks_slider_iv",
                )

            # Convert slider values to BS inputs
            g_T     = g_dte / 365.0
            g_sigma = g_iv_pct / 100.0

            st.divider()

            st.subheader(f"{g_opt_type.title()} · K={g_strike} · {g_expiry}")

            result = gr.calculate_greeks(g_S, g_strike, g_T, r=0.05, sigma=g_sigma, option_type=g_opt_type)

            st.markdown("##### Black-Scholes values")
            m0, m1, m2, m3, m4, m5 = st.columns(6)
            m0.metric("Price",  f"${result['price']:.4f}")
            m1.metric("Delta",  f"{result['delta']:.4f}")
            m2.metric("Gamma",  f"{result['gamma']:.6f}")
            m3.metric("Theta",  f"{result['theta']:.4f}")
            m4.metric("Vega",   f"{result['vega']:.4f}")
            m5.metric("Rho",    f"{result['rho']:.4f}")

            st.caption(
                f"S = ${g_S:,.2f}  ·  K = ${g_strike:,.2f}  ·  "
                f"DTE = {g_dte}  ·  σ = {g_sigma:.1%}  ·  r = 5.00%"
            )

            st.divider()
            st.markdown("##### What this means")
            _greek_plain_english(result, ticker, g_opt_type, g_S, g_strike)

            st.divider()

            # ------------------------------------------------------------------
            # Greeks vs. Price chart
            # ------------------------------------------------------------------
            st.markdown("##### Greeks vs. Underlying Price")

            gvp_col, _ = st.columns([2, 2])
            with gvp_col:
                gvp_choice = st.radio(
                    "Show",
                    ["all", "delta", "gamma", "theta", "vega"],
                    horizontal=True,
                    key="greeks_vs_price_radio",
                    format_func=lambda x: x.title(),
                )

            with st.spinner("Building chart…"):
                fig_gvp = charts.greeks_vs_price(
                    K=g_strike,
                    T=g_T,
                    r=0.05,
                    sigma=g_sigma,
                    option_type=g_opt_type,
                    S_current=g_S,
                    selected=gvp_choice,
                )
            st.plotly_chart(fig_gvp, use_container_width=True)

            st.divider()
            st.markdown("##### Greek surface")

            surf_col, _ = st.columns([3, 1])
            with surf_col:
                greek_choice = st.selectbox(
                    "Greek", ["delta", "gamma", "theta", "vega", "rho"],
                    key="greeks_surface_select",
                )

            with st.spinner("Rendering surface…"):
                fig_surface = charts.greek_surface(
                    greek_choice, g_opt_type, S=g_S, r=0.05, sigma=g_sigma
                )
            st.plotly_chart(fig_surface, use_container_width=True)

# ===========================================================================
# Tab 3 — Payoff
# Inline controls: expiry · option type · strike · position · rate
# ===========================================================================

with tab_payoff:
    p_ctrl1, p_ctrl2, p_ctrl3 = st.columns([2, 1, 1])

    with p_ctrl1:
        p_expiry = _expiry_selector("Expiration", "payoff_expiry", expirations)

    with p_ctrl2:
        p_opt_type = st.radio(
            "Type", ["call", "put"],
            index=0 if st.session_state.payoff_option_type == "call" else 1,
            horizontal=True,
            key="_payoff_opt_radio",
        )
        if p_opt_type != st.session_state.payoff_option_type:
            st.session_state.payoff_option_type = p_opt_type
            st.session_state.payoff_strike = None

    with p_ctrl3:
        p_position = st.radio(
            "Position", ["long", "short"],
            index=0 if st.session_state.payoff_position == "long" else 1,
            horizontal=True,
            key="_payoff_pos_radio",
        )
        st.session_state.payoff_position = p_position

    p_ctrl4, p_ctrl5 = st.columns([2, 2])

    with p_ctrl4:
        p_strike = _strike_selector(
            "Strike", "payoff_strike", ticker, p_expiry, p_opt_type
        )

    with p_ctrl5:
        p_rate = st.slider(
            "Risk-free rate (%)",
            min_value=0.0, max_value=10.0, value=5.0, step=0.25,
            key="payoff_rate_slider",
        ) / 100.0

    st.divider()

    if p_expiry is None or p_strike is None:
        st.info("Select an expiration and strike above to view the payoff diagram.")
    else:
        p_calls, p_puts, p_spot = _load_chain(ticker, p_expiry)
        p_chain = p_calls if p_opt_type == "call" else p_puts

        p_row = p_chain[p_chain["strike"] == p_strike]
        if p_row.empty:
            st.warning("No data for the selected contract.")
        else:
            p_row = p_row.iloc[0]
            mid_px  = p_row.get("mid", np.nan)
            last_px = p_row.get("lastPrice", np.nan)

            def _to_float(v):
                try:
                    f = float(v)
                    return f if not np.isnan(f) else None
                except Exception:
                    return None

            auto_premium = _to_float(mid_px) or _to_float(last_px) or 0.0

            st.subheader(
                f"{p_position.title()} {p_opt_type.title()} · K={p_strike} · {p_expiry}"
            )

            premium = st.number_input(
                "Premium per share ($)",
                min_value=0.0,
                value=round(auto_premium, 2),
                step=0.01,
                format="%.2f",
                help="Pre-filled from the chain mid price. Edit to model a different fill.",
                key="payoff_premium_input",
            )

            S_range = np.linspace(p_spot * 0.70, p_spot * 1.30, 400)

            fig_payoff = charts.plot_payoff(
                option_type=p_opt_type,
                position=p_position,
                strike=p_strike,
                premium=premium,
                S_range=S_range,
            )
            st.plotly_chart(fig_payoff, use_container_width=True)

            # Contract summary stats
            be = _breakeven(p_opt_type, p_position, p_strike, premium)

            if p_opt_type == "call" and p_position == "long":
                max_profit, max_loss = "Unlimited", f"${premium:.2f}"
            elif p_opt_type == "call" and p_position == "short":
                max_profit, max_loss = f"${premium:.2f}", "Unlimited"
            elif p_opt_type == "put" and p_position == "long":
                max_profit = f"${max(p_strike - premium, 0):.2f}"
                max_loss   = f"${premium:.2f}"
            else:  # short put
                max_profit = f"${premium:.2f}"
                max_loss   = f"${max(p_strike - premium, 0):.2f}"

            st.markdown("##### Contract summary")
            cs0, cs1, cs2, cs3 = st.columns(4)
            cs0.metric("Premium",    f"${premium:.2f}")
            cs1.metric("Breakeven",  f"${be:.2f}" if be is not None else "N/A")
            cs2.metric("Max profit", max_profit)
            cs3.metric("Max loss",   max_loss)
