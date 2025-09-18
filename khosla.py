# ai_valuation_fund_return_simulator.py
# Streamlit app to stress‑test whether current AI entry valuations can plausibly return a venture fund.
# Now supports Monte Carlo (50–1000 runs) and **absolute exit modeling in dollars** so entry PMV materially affects results.

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Helpers
# -----------------------------

def fmt_money(x):
    return f"${x:,.0f}" if pd.notnull(x) else "—"

# Single‑run simulator (one portfolio draw)
@st.cache_data
def simulate_portfolio(
    fund_size,
    invest_pct,
    n_initial,
    entry_pmv,
    target_ownership,  # kept for heatmap/sensitivity; check sizing remains budget‑driven by default
    reserve_ratio,
    followon_rounds,
    per_round_dilution,
    win_rate,
    loser_exit_dollars,
    tail_alpha,
    winner_exit_min,
    winner_exit_cap,
    target_tvpi,
    horizon_years,
    exit_timing_mode,
    exit_years_fixed,
    seed=42,
    initial_check_override=None,  # NEW: if provided, force initial check (ownership-driven grid)
):
    rng = np.random.default_rng(seed)

    # Capital staging
    invest_cap = fund_size * invest_pct
    if initial_check_override is not None:
        initial_check = float(initial_check_override)
    else:
        initial_check = invest_cap / (1 + reserve_ratio) / n_initial

    # Initial ownership from post‑money valuation (PMV)
    init_own = initial_check / entry_pmv

    # Follow‑on model: if reserves > 0, defend pro‑rata; else dilute
    if reserve_ratio > 0:
        post_own = init_own
        followon_spend = initial_check * reserve_ratio
    else:
        post_own = init_own * ((1 - per_round_dilution) ** followon_rounds)
        followon_spend = 0.0

    cost_per_deal = initial_check + followon_spend
    total_deployed = cost_per_deal * n_initial

    # Outcomes in ABSOLUTE DOLLARS (no MOIC-on-entry):
    # Winners: Pareto tail on exit enterprise value in $; Losers: small $ exit/recovery
    winners = rng.binomial(1, win_rate, size=n_initial).astype(bool)
    U = rng.random(size=n_initial)
    winner_exits = winner_exit_min * (U ** (-1.0 / tail_alpha))
    if np.isfinite(winner_exit_cap) and winner_exit_cap > 0:
        winner_exits = np.minimum(winner_exits, winner_exit_cap)
    # ensure min bound
    winner_exits = np.clip(winner_exits, winner_exit_min, None)

    loser_exits = np.full(n_initial, loser_exit_dollars)
    exit_values = np.where(winners, winner_exits, loser_exits)

    # Deal‑level cash back and MOIC to the fund now depend on ownership and absolute exits
    cash_back = post_own * exit_values
    deal_moic = cash_back / cost_per_deal

    df = pd.DataFrame({
        'deal_id': np.arange(1, n_initial+1),
        'winner': winners,
        'exit_value_$': exit_values,
        'ownership_final': post_own,
        'entry_pmv': entry_pmv,
        'initial_check': initial_check,
        'followon_spend': followon_spend,
        'cost_per_deal': cost_per_deal,
        'deal_moic': deal_moic,
        'cash_back': cash_back,
    })

    tvpi = df['cash_back'].sum() / total_deployed

    # Exit timing
    if exit_timing_mode == 'Front‑loaded (early partials, late big wins)':
        exit_years = np.where(df['winner'], horizon_years, rng.integers(1, max(2, horizon_years-1), size=n_initial))
    elif exit_timing_mode == 'Uniform exits across horizon':
        exit_years = rng.integers(1, horizon_years+1, size=n_initial)
    else:  # Fixed
        exit_years = np.full(n_initial, exit_years_fixed)
    df['exit_year'] = exit_years

    yearly = (
        df.groupby('exit_year')['cash_back']
        .sum()
        .reindex(range(1, horizon_years+1), fill_value=0.0)
        .rename('distributions')
        .to_frame()
    )
    yearly['cumulative_dpi'] = yearly['distributions'].cumsum() / total_deployed

    # Required **winner exit $** to hit target TVPI (deterministic algebra):
    # TVPI = [ post_own * ( p * W_exit + (1-p) * L_exit ) ] / cost_per_deal
    # Solve for W_exit
    p = win_rate
    L = loser_exit_dollars
    if post_own > 0 and p > 0:
        required_winner_exit = (target_tvpi * cost_per_deal / post_own - (1 - p) * L) / p
    else:
        required_winner_exit = np.nan

    insights = {
        'invest_capital': invest_cap,
        'total_deployed': total_deployed,
        'initial_check': initial_check,
        'ownership_initial': init_own,
        'ownership_final': post_own,
        'tvpi': tvpi,
        'required_winner_exit_$': required_winner_exit,
    }

    return df, yearly, insights

# Multi‑run Monte Carlo wrapper
@st.cache_data
def run_simulations(n_sims, seed_base, **kwargs):
    tvpis = []
    dpi_curves = []
    dfs = []
    yearlies = []
    insights_list = []

    for i in range(n_sims):
        df_i, yearly_i, ins_i = simulate_portfolio(seed=seed_base + i, **kwargs)
        tvpis.append(ins_i['tvpi'])
        dpi_curves.append(yearly_i['cumulative_dpi'].values)
        dfs.append(df_i)
        yearlies.append(yearly_i)
        insights_list.append(ins_i)

    tvpis = np.array(tvpis)
    dpi_curves = np.vstack(dpi_curves)

    tvpi_mean = float(tvpis.mean())
    tvpi_med = float(np.median(tvpis))
    tvpi_p5, tvpi_p95 = float(np.percentile(tvpis, 5)), float(np.percentile(tvpis, 95))

    dpi_mean = dpi_curves.mean(axis=0)
    dpi_p10 = np.percentile(dpi_curves, 10, axis=0)
    dpi_p90 = np.percentile(dpi_curves, 90, axis=0)

    med_idx = int(np.argsort(tvpis)[len(tvpis)//2])

    agg = {
        'tvpis': tvpis,
        'tvpi_mean': tvpi_mean,
        'tvpi_median': tvpi_med,
        'tvpi_p5': tvpi_p5,
        'tvpi_p95': tvpi_p95,
        'dpi_mean': dpi_mean,
        'dpi_p10': dpi_p10,
        'dpi_p90': dpi_p90,
        'median_run_df': dfs[med_idx],
        'median_run_yearly': yearlies[med_idx],
        'median_run_insights': insights_list[med_idx],
    }
    return agg

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="AI Valuation Fund‑Return Simulator", page_icon="📈", layout="wide")

st.title("📈 AI Valuation Fund‑Return Simulator — Absolute Exit Mode")
st.markdown(
    """
This simulator now models **absolute exit values in dollars** (not MOIC‑on‑entry). As a result, higher entry PMVs
reduce ownership for the same check and **degrade TVPI/DPI**, matching real valuation sensitivity.
"""
)

with st.sidebar:
    st.header("Portfolio & Entry Terms")
    fund_size = st.number_input("Fund size ($)", value=200_000_000, step=10_000_000, min_value=10_000_000)
    invest_pct = st.slider("% of fund deployed into companies (after fees)", 0.5, 1.0, 0.9)
    n_initial = st.slider("Initial positions", 10, 80, 30)
    entry_pmv = st.number_input("Typical entry post‑money valuation ($)", value=150_000_000, step=10_000_000, min_value=10_000_000)
    target_ownership = st.number_input("Target ownership at entry (%) (for sensitivity only)", value=7.0, step=0.5, min_value=1.0, max_value=25.0) / 100

    st.divider()
    st.header("Reserves & Dilution")
    reserve_ratio = st.slider("Reserves as a multiple of initial check", 0.0, 2.0, 1.0, 0.1)
    followon_rounds = st.slider("# of follow‑on rounds (if no reserves)", 0, 5, 2)
    per_round_dilution = st.slider("Per‑round dilution if you don’t defend (%)", 0, 40, 20) / 100

    st.divider()
    st.header("Exit Model (absolute $)")
    win_rate = st.slider("Winner rate (%)", 1, 30, 5) / 100
    tail_alpha = st.slider("Tail thickness α (lower = fatter tail)", 0.5, 3.0, 1.2, 0.1)
    winner_exit_min = st.number_input("Min winner exit ($)", value=300_000_000, step=50_000_000, min_value=10_000_000)
    winner_exit_cap = st.number_input("Max winner exit cap ($, 0 = no cap)", value=100_000_000_000, step=10_000_000)
    loser_exit_dollars = st.number_input("Loser exit / recovery ($)", value=5_000_000, step=1_000_000, min_value=0)

    st.divider()
    st.header("DPI vs TVPI Timing")
    horizon_years = st.slider("Fund horizon (years)", 5, 12, 10)
    exit_timing_mode = st.selectbox("Exit timing", [
        'Front‑loaded (early partials, late big wins)',
        'Uniform exits across horizon',
        'Fixed year for all exits',
    ])
    exit_years_fixed = st.slider("If fixed, exit year", 1, 12, 8)

    st.divider()
    st.header("Targets & Simulations")
    target_tvpi = st.slider("Target gross TVPI to test (x)", 1.0, 5.0, 4.0, 0.1)
    n_sims = st.slider("# simulations (Monte Carlo)", 50, 1000, 200, step=10)
    seed = st.number_input("Base random seed", value=42)

# -----------------------------
# Run simulations
# -----------------------------

with st.spinner("Running Monte Carlo simulations…"):
    agg = run_simulations(
        n_sims=n_sims,
        seed_base=seed,
        fund_size=fund_size,
        invest_pct=invest_pct,
        n_initial=n_initial,
        entry_pmv=entry_pmv,
        target_ownership=target_ownership,
        reserve_ratio=reserve_ratio,
        followon_rounds=followon_rounds,
        per_round_dilution=per_round_dilution,
        win_rate=win_rate,
        loser_exit_dollars=loser_exit_dollars,
        tail_alpha=tail_alpha,
        winner_exit_min=winner_exit_min,
        winner_exit_cap=(0 if winner_exit_cap == 0 else winner_exit_cap),
        target_tvpi=target_tvpi,
        horizon_years=horizon_years,
        exit_timing_mode=exit_timing_mode,
        exit_years_fixed=exit_years_fixed,
    )

# -----------------------------
# KPIs (averaged across simulations)
# -----------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Simulations", f"{n_sims}")
col2.metric("Gross TVPI (mean)", f"{agg['tvpi_mean']:.2f}×")
col3.metric("Gross TVPI (median)", f"{agg['tvpi_median']:.2f}×")
col4.metric("TVPI 5–95%", f"{agg['tvpi_p5']:.2f}× – {agg['tvpi_p95']:.2f}×")

# Deterministic required winner exit $
_, _, _ins = simulate_portfolio(
    fund_size=fund_size,
    invest_pct=invest_pct,
    n_initial=n_initial,
    entry_pmv=entry_pmv,
    target_ownership=target_ownership,
    reserve_ratio=reserve_ratio,
    followon_rounds=followon_rounds,
    per_round_dilution=per_round_dilution,
    win_rate=win_rate,
    loser_exit_dollars=loser_exit_dollars,
    tail_alpha=tail_alpha,
    winner_exit_min=winner_exit_min,
    winner_exit_cap=(0 if winner_exit_cap == 0 else winner_exit_cap),
    target_tvpi=target_tvpi,
    horizon_years=horizon_years,
    exit_timing_mode=exit_timing_mode,
    exit_years_fixed=exit_years_fixed,
    seed=seed,
)
req_exit = _ins['required_winner_exit_$']
if np.isfinite(req_exit) and req_exit > 0:
    st.info(f"To hit {target_tvpi:.1f}× TVPI with these terms, each *winner* must average **{fmt_money(req_exit)}** exit value.")
else:
    st.info("Insufficient ownership or winner rate is zero; cannot compute a finite required winner exit.")

# -----------------------------
# Visuals
# -----------------------------

# A) Distribution of portfolio‑level TVPI across simulations
fig_tvpi = px.histogram(x=agg['tvpis'], nbins=30, labels={'x':'Portfolio TVPI (×)'}, title='Distribution of Portfolio TVPI Across Simulations')
fig_tvpi.update_layout(margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_tvpi, use_container_width=True)

# B) Mean DPI curve with uncertainty band
years = np.arange(1, horizon_years+1)
fig_dpi = go.Figure()
fig_dpi.add_trace(go.Scatter(x=years, y=agg['dpi_mean'], mode='lines+markers', name='Mean DPI'))
fig_dpi.add_trace(go.Scatter(x=years, y=agg['dpi_p10'], mode='lines', name='P10', line=dict(dash='dash')))
fig_dpi.add_trace(go.Scatter(x=years, y=agg['dpi_p90'], mode='lines', name='P90', line=dict(dash='dash'), fill='tonexty'))
fig_dpi.update_layout(title='Cumulative DPI Over Time (Mean with P10–P90 Band)', xaxis_title='Year', yaxis_title='DPI (Distributions / Paid‑In)', margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_dpi, use_container_width=True)

# C) Representative waterfall from the median TVPI run
rep_df = agg['median_run_df'].sort_values('cash_back', ascending=False).reset_index(drop=True)
fig_wf = go.Figure()
fig_wf.add_trace(go.Bar(x=rep_df.index+1, y=rep_df['cash_back'], name='Cash Back'))
fig_wf.update_layout(title='Cash Back per Deal (Representative Median Run)', xaxis_title='Deal rank', yaxis_title='Cash returned ($)', margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_wf, use_container_width=True)

# D) Sensitivity — Ownership‑driven grid (re-coded)
st.subheader("Sensitivity: Mean TVPI across 10 sims (Ownership‑driven grid)")

pmv_vals = np.linspace(10_000_000, 100_000_000, 10)  # $10MM → $100MM
own_vals = np.linspace(0.01, 0.15, 15)               # 1% → 15%
Z = np.full((len(own_vals), len(pmv_vals)), np.nan)

# For this grid we *force* ownership by overriding the initial check: check = own × PMV
# Note: This grid isolates valuation/ownership effects and does not enforce portfolio budget feasibility per cell.

for i, own in enumerate(own_vals):
    for j, pmv in enumerate(pmv_vals):
        initial_check_override = own * pmv
        tvpis = []
        for k in range(10):
            _, _, ins_k = simulate_portfolio(
                fund_size=fund_size,
                invest_pct=invest_pct,
                n_initial=n_initial,
                entry_pmv=pmv,
                target_ownership=target_ownership,
                reserve_ratio=reserve_ratio,
                followon_rounds=followon_rounds,
                per_round_dilution=per_round_dilution,
                win_rate=win_rate,
                loser_exit_dollars=loser_exit_dollars,
                tail_alpha=tail_alpha,
                winner_exit_min=winner_exit_min,
                winner_exit_cap=(0 if winner_exit_cap == 0 else winner_exit_cap),
                target_tvpi=target_tvpi,
                horizon_years=horizon_years,
                exit_timing_mode=exit_timing_mode,
                exit_years_fixed=exit_years_fixed,
                seed=seed + k + i*100 + j*1000,
                initial_check_override=initial_check_override,
            )
            tvpis.append(ins_k['tvpi'])
        Z[i, j] = float(np.mean(tvpis))

heat_df = pd.DataFrame(Z, index=np.round(own_vals*100,1), columns=np.round(pmv_vals/1e6,1))
fig_heat = px.imshow(
    heat_df,
    aspect='auto',
    origin='lower',
    labels=dict(x='Entry PMV ($MM)', y='Ownership (%)', color='Mean TVPI (×)'),
    title='Mean Portfolio TVPI across 10 sims — Ownership-driven grid',
    zmin=np.nanpercentile(Z, 5),
    zmax=np.nanpercentile(Z, 95),
)

# Invert the scale so high values map to dark colors (or vice versa)
fig_heat.update_coloraxes(reversescale=True)

st.plotly_chart(fig_heat, use_container_width=True)


st.caption("Grid forces check = ownership × PMV for each cell (ignoring fund budget) to isolate valuation/ownership effects on mean TVPI.")
