"""
charts.py — Plotly-based payoff diagrams and Greek surface charts.

All functions return a plotly.graph_objects.Figure so the caller
can render it directly (st.plotly_chart) or add further traces.

Public API:
    plot_payoff(option_type, position, strike, premium, S_range) -> go.Figure
    payoff_diagram(legs, S, title)                               -> go.Figure
    greek_surface(greek, option_type, S, r, sigma)               -> go.Figure
    iv_smile(chain_df, option_type)                              -> go.Figure
    open_interest_chart(chain_df)                                -> go.Figure
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Shared colour palette so all charts feel consistent.
_GREEN = "#26a69a"
_RED = "#ef5350"
_BLUE = "#5c6bc0"
_GREY = "#90a4ae"
_YELLOW = "#ffa726"

_BG = "#0e1117"          # Streamlit dark background
_PAPER = "#1a1d23"       # slightly lighter panel background
_GRID = "#2a2d35"        # subtle grid lines


def _base_layout(**overrides) -> dict:
    """Return a dark-theme layout dict, optionally overriding keys."""
    layout = dict(
        paper_bgcolor=_PAPER,
        plot_bgcolor=_BG,
        font=dict(color="#e0e0e0", size=13),
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GREY),
        yaxis=dict(gridcolor=_GRID, zerolinecolor=_GREY),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin=dict(l=60, r=30, t=60, b=50),
        hovermode="x unified",
    )
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# Single-leg payoff
# ---------------------------------------------------------------------------

def _leg_payoff(
    S_range: np.ndarray,
    option_type: str,
    position: str,
    strike: float,
    premium: float,
) -> np.ndarray:
    """Compute per-share P&L at expiry for one option leg.

    Intrinsic value at expiry (before cost basis):
        call: max(S - K, 0)
        put:  max(K - S, 0)

    Long pays the premium; short receives it.
    """
    option_type = option_type.lower()
    position = position.lower()

    if option_type == "call":
        intrinsic = np.maximum(S_range - strike, 0)
    elif option_type == "put":
        intrinsic = np.maximum(strike - S_range, 0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if position == "long":
        return intrinsic - premium
    elif position == "short":
        return premium - intrinsic
    else:
        raise ValueError(f"position must be 'long' or 'short', got '{position}'")


def _breakeven(
    option_type: str, position: str, strike: float, premium: float
) -> float | None:
    """Return the exact breakeven price, or None for undefined cases."""
    ot = option_type.lower()
    pos = position.lower()

    if ot == "call" and pos == "long":
        return strike + premium
    if ot == "call" and pos == "short":
        return strike + premium          # same point, mirrored P&L
    if ot == "put" and pos == "long":
        return strike - premium
    if ot == "put" and pos == "short":
        return strike - premium
    return None


def plot_payoff(
    option_type: str,
    position: str,
    strike: float,
    premium: float,
    S_range: np.ndarray,
) -> go.Figure:
    """Plot the P&L payoff diagram for a single European option leg at expiry.

    Parameters
    ----------
    option_type : {'call', 'put'}
        The type of option contract.
    position : {'long', 'short'}
        Long = bought the option; short = sold (wrote) it.
    strike : float
        The option's strike price.
    premium : float
        Price paid (long) or received (short) per share.
    S_range : np.ndarray
        Array of underlying prices at expiration to evaluate P&L over.
        Tip: np.linspace(spot * 0.7, spot * 1.3, 300) gives a clean range.

    Returns
    -------
    plotly.graph_objects.Figure
        Dark-themed payoff chart with:
        - Filled profit region (green) and loss region (red)
        - Horizontal zero line
        - Vertical dashed line at the breakeven price
        - Hover labels showing the exact P&L at each price point
    """
    pnl = _leg_payoff(S_range, option_type, position, strike, premium)
    be = _breakeven(option_type, position, strike, premium)

    profit = np.where(pnl >= 0, pnl, np.nan)
    loss = np.where(pnl < 0, pnl, np.nan)

    label = f"{position.title()} {option_type.title()}"
    hover = "S=%{x:.2f}<br>P&L=%{y:.2f}<extra></extra>"

    fig = go.Figure()

    # --- profit fill (above zero) ---
    fig.add_trace(go.Scatter(
        x=S_range, y=profit,
        mode="lines",
        name="Profit",
        line=dict(color=_GREEN, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(38,166,154,0.18)",
        hovertemplate=hover,
    ))

    # --- loss fill (below zero) ---
    fig.add_trace(go.Scatter(
        x=S_range, y=loss,
        mode="lines",
        name="Loss",
        line=dict(color=_RED, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(239,83,80,0.18)",
        hovertemplate=hover,
    ))

    # --- combined line on top so hover is clean ---
    fig.add_trace(go.Scatter(
        x=S_range, y=pnl,
        mode="lines",
        name=label,
        line=dict(color=_BLUE, width=1.5, dash="dot"),
        hovertemplate=hover,
        showlegend=False,
    ))

    # --- zero line ---
    fig.add_hline(
        y=0,
        line=dict(color=_GREY, width=1, dash="solid"),
    )

    # --- strike marker ---
    fig.add_vline(
        x=strike,
        line=dict(color=_YELLOW, width=1, dash="dash"),
        annotation=dict(
            text=f"Strike {strike:.2f}",
            font=dict(color=_YELLOW, size=11),
            yref="paper", y=1.0, yanchor="bottom",
        ),
    )

    # --- breakeven marker ---
    if be is not None and S_range[0] <= be <= S_range[-1]:
        fig.add_vline(
            x=be,
            line=dict(color=_GREEN, width=1.5, dash="dash"),
            annotation=dict(
                text=f"B/E {be:.2f}",
                font=dict(color=_GREEN, size=11),
                yref="paper", y=0.88, yanchor="bottom",
            ),
        )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"{label} — Payoff at Expiry  |  K={strike}  Premium={premium}",
                font=dict(size=15),
            ),
            xaxis_title="Underlying Price at Expiry",
            yaxis_title="P&L per Share ($)",
        )
    )

    return fig


# ---------------------------------------------------------------------------
# Multi-leg payoff diagram
# ---------------------------------------------------------------------------

def payoff_diagram(
    legs: list[dict],
    S: float,
    title: str = "Strategy Payoff at Expiry",
) -> go.Figure:
    """Plot combined P&L at expiry for one or more option/stock legs.

    Each leg dict:
        {
            "option_type": "call" | "put" | "stock",
            "position":    "long" | "short",
            "strike":      float,   # ignored for stock legs
            "premium":     float,   # cost basis per share
            "quantity":    int,     # number of contracts (1 contract = 100 shares)
        }

    Parameters
    ----------
    legs : list[dict]
        Option/stock legs to combine.
    S : float
        Current spot price (used to centre the x-axis range).
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Combined payoff diagram with individual leg traces (thin dashed)
        and a bold combined P&L trace.
    """
    S_range = np.linspace(S * 0.6, S * 1.4, 400)
    combined = np.zeros_like(S_range)
    leg_colors = [_BLUE, _YELLOW, _GREY, "#ce93d8", "#80cbc4"]

    fig = go.Figure()

    for i, leg in enumerate(legs):
        ot = leg.get("option_type", "call").lower()
        pos = leg.get("position", "long").lower()
        strike = float(leg.get("strike", S))
        premium = float(leg.get("premium", 0.0))
        qty = int(leg.get("quantity", 1))

        if ot == "stock":
            intrinsic = S_range - strike
            pnl = (intrinsic - premium) * qty if pos == "long" else (premium - intrinsic) * qty
        else:
            pnl = _leg_payoff(S_range, ot, pos, strike, premium) * qty

        combined += pnl
        color = leg_colors[i % len(leg_colors)]
        leg_label = f"Leg {i+1}: {pos.title()} {ot.title()}"
        if ot != "stock":
            leg_label += f" K={strike}"

        fig.add_trace(go.Scatter(
            x=S_range, y=pnl,
            mode="lines",
            name=leg_label,
            line=dict(color=color, width=1.2, dash="dot"),
            hovertemplate="S=%{x:.2f}<br>P&L=%{y:.2f}<extra>" + leg_label + "</extra>",
        ))

    # Combined P&L with profit/loss shading
    profit = np.where(combined >= 0, combined, np.nan)
    loss = np.where(combined < 0, combined, np.nan)

    fig.add_trace(go.Scatter(
        x=S_range, y=profit, mode="lines", name="Combined (profit)",
        line=dict(color=_GREEN, width=3),
        fill="tozeroy", fillcolor="rgba(38,166,154,0.15)",
        hovertemplate="S=%{x:.2f}<br>Combined=%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=S_range, y=loss, mode="lines", name="Combined (loss)",
        line=dict(color=_RED, width=3),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
        hovertemplate="S=%{x:.2f}<br>Combined=%{y:.2f}<extra></extra>",
    ))

    fig.add_hline(y=0, line=dict(color=_GREY, width=1))

    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Underlying Price at Expiry",
        yaxis_title="P&L per Share ($)",
    ))

    return fig


# ---------------------------------------------------------------------------
# Greek surface
# ---------------------------------------------------------------------------

def greek_surface(
    greek: str,
    option_type: str = "call",
    S: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.20,
) -> go.Figure:
    """3-D surface of a single Greek across strike and DTE.

    Parameters
    ----------
    greek : {'delta', 'gamma', 'theta', 'vega', 'rho'}
    option_type : {'call', 'put'}
    S, r, sigma : float
        Held constant; only strike and time-to-expiry vary.

    Returns
    -------
    go.Figure
        Interactive 3-D surface rendered with go.Surface.
    """
    # Import here to avoid a circular import when greeks.py imports charts.
    from greeks import calculate_greeks  # noqa: PLC0415

    strikes = np.linspace(S * 0.7, S * 1.3, 40)
    dtes = np.arange(1, 91)          # 1–90 calendar days
    T_vals = dtes / 365.0

    Z = np.zeros((len(dtes), len(strikes)))
    for i, T in enumerate(T_vals):
        for j, K in enumerate(strikes):
            try:
                result = calculate_greeks(S, K, T, r, sigma, option_type)
                Z[i, j] = result[greek]
            except Exception:
                Z[i, j] = np.nan

    fig = go.Figure(data=go.Surface(
        x=strikes,
        y=dtes,
        z=Z,
        colorscale="RdYlGn",
        colorbar=dict(title=greek.title(), tickfont=dict(color="#e0e0e0")),
        hovertemplate=(
            "Strike=%{x:.1f}<br>DTE=%{y}d<br>"
            + greek.title() + "=%{z:.4f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text=f"{greek.title()} Surface — {option_type.title()}  "
                 f"(S={S}, σ={sigma:.0%}, r={r:.1%})",
            font=dict(size=15, color="#e0e0e0"),
        ),
        scene=dict(
            xaxis=dict(title="Strike", backgroundcolor=_BG, gridcolor=_GRID),
            yaxis=dict(title="DTE (days)", backgroundcolor=_BG, gridcolor=_GRID),
            zaxis=dict(title=greek.title(), backgroundcolor=_BG, gridcolor=_GRID),
            bgcolor=_BG,
        ),
        paper_bgcolor=_PAPER,
        font=dict(color="#e0e0e0"),
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return fig


# ---------------------------------------------------------------------------
# IV smile
# ---------------------------------------------------------------------------

def iv_smile(
    chain_df: pd.DataFrame,
    option_type: str = "call",
) -> go.Figure:
    """Scatter + spline of implied volatility vs. strike for one expiration.

    Parameters
    ----------
    chain_df : pd.DataFrame
        Enriched chain DataFrame; must contain 'strike' and 'impliedVolatility'.
    option_type : {'call', 'put'}
        Used only for the chart title label.

    Returns
    -------
    go.Figure
    """
    df = chain_df.dropna(subset=["impliedVolatility"]).copy()
    df = df[df["impliedVolatility"] > 0].sort_values("strike")

    color = _BLUE if option_type.lower() == "call" else _RED

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["strike"],
        y=df["impliedVolatility"] * 100,     # display as percentage
        mode="lines+markers",
        name=f"IV ({option_type.title()})",
        line=dict(color=color, width=2, shape="spline", smoothing=1.0),
        marker=dict(size=5, color=color),
        hovertemplate="Strike=%{x:.2f}<br>IV=%{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        title=dict(
            text=f"IV Smile — {option_type.title()}s",
            font=dict(size=15),
        ),
        xaxis_title="Strike",
        yaxis_title="Implied Volatility (%)",
    ))

    return fig


# ---------------------------------------------------------------------------
# Open interest / volume
# ---------------------------------------------------------------------------

def open_interest_chart(chain_df: pd.DataFrame) -> go.Figure:
    """Bar chart of open interest with volume overlaid as a line.

    Parameters
    ----------
    chain_df : pd.DataFrame
        Enriched chain DataFrame; must contain 'strike', 'openInterest', 'volume'.

    Returns
    -------
    go.Figure
        Dual-axis chart: bars = open interest (left axis),
        line = volume (right axis).
    """
    df = chain_df.sort_values("strike")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["strike"],
        y=df["openInterest"],
        name="Open Interest",
        marker_color=_BLUE,
        opacity=0.75,
        hovertemplate="Strike=%{x:.2f}<br>OI=%{y:,}<extra></extra>",
        yaxis="y1",
    ))

    fig.add_trace(go.Scatter(
        x=df["strike"],
        y=df["volume"],
        name="Volume",
        mode="lines+markers",
        line=dict(color=_YELLOW, width=2),
        marker=dict(size=4),
        hovertemplate="Strike=%{x:.2f}<br>Vol=%{y:,}<extra></extra>",
        yaxis="y2",
    ))

    fig.update_layout(**_base_layout(
        title=dict(text="Open Interest & Volume by Strike", font=dict(size=15)),
        xaxis_title="Strike",
        yaxis=dict(title="Open Interest", gridcolor=_GRID),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        hovermode="x unified",
    ))

    return fig


# ---------------------------------------------------------------------------
# Greeks vs. underlying price
# ---------------------------------------------------------------------------

# Per-Greek display metadata: colour and y-axis label.
_GREEK_META: dict[str, dict] = {
    "delta": {"color": _BLUE,   "label": "Delta (0–1 / -1–0)"},
    "gamma": {"color": _GREEN,  "label": "Gamma"},
    "theta": {"color": _RED,    "label": "Theta ($/day)"},
    "vega":  {"color": _YELLOW, "label": "Vega ($/vol-pt)"},
}

_GREEKS_TO_PLOT = list(_GREEK_META.keys())


def greeks_vs_price(
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    S_current: float,
    selected: str = "all",
    n_points: int = 200,
) -> go.Figure:
    """Plot one or all Greeks against a range of underlying prices.

    Parameters
    ----------
    K : float
        Strike price.
    T : float
        Time to expiry in years (from the DTE slider).
    r : float
        Risk-free rate as a decimal.
    sigma : float
        Implied volatility as a decimal (from the IV slider).
    option_type : {'call', 'put'}
    S_current : float
        Current underlying price — used for the x-axis range and
        the vertical marker.
    selected : str
        One of 'all', 'delta', 'gamma', 'theta', 'vega'. Controls which
        traces are shown. When 'all', each Greek gets its own secondary
        y-axis so the different scales don't compress each other.
    n_points : int
        Number of price steps across the 70%–130% range.

    Returns
    -------
    go.Figure
        Dark-themed line chart with a vertical dashed marker at S_current.
    """
    from greeks import calculate_greeks  # avoid circular import at module level

    S_lo = S_current * 0.70
    S_hi = S_current * 1.30
    S_range = np.linspace(S_lo, S_hi, n_points)

    # Build a dict of arrays: {greek_name: [values across S_range]}
    series: dict[str, np.ndarray] = {g: np.empty(n_points) for g in _GREEKS_TO_PLOT}
    for i, S in enumerate(S_range):
        try:
            res = calculate_greeks(S, K, T, r, sigma, option_type)
            for g in _GREEKS_TO_PLOT:
                series[g][i] = res[g]
        except Exception:
            for g in _GREEKS_TO_PLOT:
                series[g][i] = np.nan

    greeks_to_show = _GREEKS_TO_PLOT if selected == "all" else [selected]

    fig = go.Figure()

    if selected == "all":
        # Normalise each Greek to [-1, 1] so all four fit on one axis without
        # the tiny gamma trace being invisible next to delta.
        for g in greeks_to_show:
            arr = series[g]
            lo, hi = np.nanmin(arr), np.nanmax(arr)
            span = hi - lo if (hi - lo) > 1e-10 else 1.0
            normalised = (arr - lo) / span * 2 - 1   # maps to [-1, 1]
            meta = _GREEK_META[g]
            fig.add_trace(go.Scatter(
                x=S_range,
                y=normalised,
                mode="lines",
                name=g.title(),
                line=dict(color=meta["color"], width=2),
                customdata=arr,
                hovertemplate=f"S=%{{x:.2f}}<br>{g.title()}=%{{customdata:.4f}}<extra></extra>",
            ))
        y_title = "Normalised value (raw shown on hover)"
    else:
        g = greeks_to_show[0]
        meta = _GREEK_META[g]
        fig.add_trace(go.Scatter(
            x=S_range,
            y=series[g],
            mode="lines",
            name=g.title(),
            line=dict(color=meta["color"], width=2.5),
            hovertemplate=f"S=%{{x:.2f}}<br>{g.title()}=%{{y:.4f}}<extra></extra>",
        ))
        y_title = meta["label"]

    # Zero reference line (only useful when a Greek crosses zero)
    fig.add_hline(y=0, line=dict(color=_GREY, width=1, dash="dot"))

    # Vertical marker at the current price
    fig.add_vline(
        x=S_current,
        line=dict(color=_GREY, width=1.5, dash="dash"),
        annotation=dict(
            text=f"S = ${S_current:.2f}",
            font=dict(color=_GREY, size=11),
            yref="paper", y=1.0, yanchor="bottom",
        ),
    )

    greek_label = "All Greeks (normalised)" if selected == "all" else selected.title()
    fig.update_layout(
        **_base_layout(
            title=dict(
                text=(
                    f"{greek_label} vs. Underlying Price — "
                    f"{option_type.title()}  K={K:.0f}  DTE={round(T*365)}  σ={sigma:.1%}"
                ),
                font=dict(size=14),
            ),
            xaxis_title="Underlying Price",
            yaxis_title=y_title,
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ),
        )
    )

    return fig


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # --- plot_payoff: all four combinations ---
    spot = 500.0
    S_range = np.linspace(spot * 0.7, spot * 1.3, 300)

    cases = [
        ("call", "long",  500.0, 12.50),
        ("call", "short", 500.0, 12.50),
        ("put",  "long",  500.0, 11.00),
        ("put",  "short", 500.0, 11.00),
    ]

    for ot, pos, K, prem in cases:
        fig = plot_payoff(ot, pos, K, prem, S_range)
        assert fig is not None
        traces = [t.name for t in fig.data]
        assert any("Profit" in (n or "") for n in traces), "Missing profit trace"
        assert any("Loss"   in (n or "") for n in traces), "Missing loss trace"
        print(f"  plot_payoff({pos} {ot}) OK — {len(fig.data)} traces")

    # --- breakeven sanity ---
    # Long call breakeven = strike + premium
    be = _breakeven("call", "long", 500.0, 12.50)
    assert abs(be - 512.50) < 1e-6, f"Wrong breakeven: {be}"
    # Long put breakeven = strike - premium
    be = _breakeven("put", "long", 500.0, 11.00)
    assert abs(be - 489.00) < 1e-6, f"Wrong breakeven: {be}"
    print("  Breakeven calculations OK")

    # --- payoff_diagram (multi-leg): long straddle ---
    straddle_legs = [
        {"option_type": "call", "position": "long", "strike": 500, "premium": 12.50, "quantity": 1},
        {"option_type": "put",  "position": "long", "strike": 500, "premium": 11.00, "quantity": 1},
    ]
    fig2 = payoff_diagram(straddle_legs, spot, title="Long Straddle")
    assert len(fig2.data) >= 3, "Expected leg traces + combined"
    print(f"  payoff_diagram (straddle) OK — {len(fig2.data)} traces")

    # --- greek_surface ---
    fig3 = greek_surface("delta", "call", S=500, r=0.05, sigma=0.20)
    surface_trace = fig3.data[0]
    assert surface_trace.type == "surface"
    assert surface_trace.z.shape == (90, 40), f"Unexpected shape: {surface_trace.z.shape}"
    print(f"  greek_surface (delta/call) OK — Z shape {surface_trace.z.shape}")

    print("\nAll charts.py checks passed.")
