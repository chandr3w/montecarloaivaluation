# ai_valuation_fund_return_simulator.py
# Streamlit app to stress‑test whether current AI entry valuations can plausibly return a venture fund.
# Now supports running 50–1000 Monte Carlo simulations and averaging results.

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
    target_ownership,
    reserve_ratio,
    followon_rounds,
    per_round_dilution,
    win_rate,
    lose_recovery,
    tail_alpha,
    tail_scale,
    target_tvpi,
    horizon_years,
    exit_timing_mode,
    exit_years_fixed,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # Capital staging
    invest_cap = fund_size * invest_pct
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

    # Outcomes
    winners = rng.binomial(1, win_rate, size=n_initial).astype(bool)

    # Power‑law on company value MOIC relative to entry PMV
    U = rng.random(size=n_initial)
    pareto_draw = tail_scale * (U ** (-1.0 / tail_alpha))
    pareto_draw = np.clip(pareto_draw, tail_scale, 10_000)
    company_moic = np.where(winners, pareto_draw, lose_recovery)

    # Deal‑level MOIC to the fund
    deal_moic = post_own * company_moic * (entry_pmv / cost_per_deal)

    df = pd.DataFrame({
        'deal_id': np.arange(1, n_initial+1),
        'winner': winners,
        'company_moic': company_moic,
        'ownership_final': post_own,
        'entry_pmv': entry_pmv,
        'initial_check': initial_check,
        'followon_spend': followon_spend,
        'cost_per_deal': cost_per_deal,
        'deal_moic': deal_moic,
        'cash_back': deal_moic * cost_per_deal,
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

    # Required winner company MOIC for target TVPI (deterministic scalar)
    p = win_rate
    L = lose_recovery
    scalar = post_own * (entry_pmv / cost_per_deal)
    required_winner_company_moic = max(0.0, (target_tvpi / scalar - (1-p)*L) / p) if p > 0 else np.nan

    insights = {
        'invest_capital': invest_cap,
        'total_deployed': total_deployed,
        'initial_check': initial_check,
        'ownership_initial': init_own,
        'ownership_final': post_own,
        'tvpi': tvpi,
        'required_winner_company_moic_for_target_tvpi': required_winner_company_moic,
    }

    return df, yearly, insights

# Multi‑run Monte Carlo wrapper
@st.cache_data
def run_simulations(n_sims, seed_base, **kwargs):
    tvpis = []
    dpi_curves = []  # list of arrays length horizon_years
    dfs = []  # store per‑run dfs for later inspection (kept modest size)
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

    # Aggregate stats
    tvpi_mean = float(tvpis.mean())
    tvpi_med = float(np.median(tvpis))
    tvpi_p5, tvpi_p95 = float(np.percentile(tvpis, 5)), float(np.percentile(tvpis, 95))

    dpi_mean = dpi_curves.mean(axis=0)
    dpi_p10 = np.percentile(dpi_curves, 10, axis=0)
    dpi_p90 = np.percentile(dpi_curves, 90, axis=0)

    # Choose the median‑TVPI run for representative deal table/waterfall visuals
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

st.title("📈 AI Valuation Fund‑Return Simulator")
st.markdown(
    """
This simulator lets you stress‑test whether **today's entry valuations** can plausibly return a fund. Now with
**Monte Carlo** (50–1000 runs) to average outcomes and show uncertainty bands.
"""
)

with st.sidebar:
    st.header("Portfolio & Entry Terms")
    fund_size = st.number_input("Fund size ($)", value=200_000_000, step=10_000_000, min_value=10_000_000)
    invest_pct = st.slider("% of fund deployed into companies (after fees)", 0.5, 1.0, 0.9)
    n_initial = st.slider("Initial positions", 10, 80, 30)
    entry_pmv = st.number_input("Typical entry post‑money valuation ($)", value=30_000_000, step=5_000_000, min_value=10_000_000)
    target_ownership = st.number_input("Target ownership at entry (%)", value=7.0, step=0.5, min_value=1.0, max_value=25.0) / 100

    st.divider()
    st.header("Reserves & Dilution")
    reserve_ratio = st.slider("Reserves as a multiple of initial check", 0.0, 2.0, 1.0, 0.1)
    followon_rounds = st.slider("# of follow‑on rounds (if no reserves)", 0, 5, 2)
    per_round_dilution = st.slider("Per‑round dilution if you don’t defend (%)", 0, 40, 20) / 100

    st.divider()
    st.header("Outcome Model (Power‑Law)")
    win_rate = st.slider("Winner rate (%)", 1, 30, 5) / 100
    lose_recovery = st.slider("Loser recovery (company MOIC)", 0.0, 0.5, 0.1, 0.05)
    tail_alpha = st.slider("Tail thickness α (lower = fatter tail)", 0.5, 3.0, 1.2, 0.1)
    tail_scale = st.slider("Tail scale (min company MOIC for winners)", 2.0, 10.0, 4.0, 0.5)

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
        lose_recovery=lose_recovery,
        tail_alpha=tail_alpha,
        tail_scale=tail_scale,
        target_tvpi=target_tvpi,
        horizon_years=horizon_years,
        exit_timing_mode=exit_timing_mode,
        exit_years_fixed=exit_years_fixed,
    )

# -----------------------------
# KPIs (averaged across simulations)
# -----------------------------

# Fix: previously we tried to index the tuple returned by simulate_portfolio with a string key, causing a TypeError.
# Updated to unpack the tuple properly and extract the insights dict.

# ... (rest of the file unchanged above)

# -----------------------------
# KPIs (averaged across simulations)
# -----------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Simulations", f"{n_sims}")
col2.metric("Gross TVPI (mean)", f"{agg['tvpi_mean']:.2f}×")
col3.metric("Gross TVPI (median)", f"{agg['tvpi_median']:.2f}×")
col4.metric("TVPI 5–95%", f"{agg['tvpi_p5']:.2f}× – {agg['tvpi_p95']:.2f}×")

# Required winner MOIC (deterministic given entry terms)
_, _, insights_single = simulate_portfolio(
    fund_size,
    invest_pct,
    n_initial,
    entry_pmv,
    target_ownership,
    reserve_ratio,
    followon_rounds,
    per_round_dilution,
    win_rate,
    lose_recovery,
    tail_alpha,
    tail_scale,
    target_tvpi,
    horizon_years,
    exit_timing_mode,
    exit_years_fixed,
    seed,
)

# Compute deterministic required winner MOIC once (no tuple-as-dict bug)
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
    lose_recovery=lose_recovery,
    tail_alpha=tail_alpha,
    tail_scale=tail_scale,
    target_tvpi=target_tvpi,
    horizon_years=horizon_years,
    exit_timing_mode=exit_timing_mode,
    exit_years_fixed=exit_years_fixed,
    seed=seed,
)
_required = _ins['required_winner_company_moic_for_target_tvpi']
scalar_note = (
    f"To hit {target_tvpi:.1f}× TVPI at these entry terms, each *winner* would need to average "
    f"**{_required:.1f}× company MOIC**."
)
st.info(scalar_note)

# -----------------------------
# Visuals
# -----------------------------

# (rest of file continues unchanged)

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

# D) Sensitivity heatmap (unchanged; deterministic mapping of entry terms)
pmv_vals = np.linspace(entry_pmv*0.5, entry_pmv*2.5, 25)
own_vals = np.linspace(max(0.02, target_ownership*0.5), min(0.30, target_ownership*1.5), 25)
Z = np.zeros((len(own_vals), len(pmv_vals)))

for i, own in enumerate(own_vals):
    initial_check_grid = own * pmv_vals
    followon_spend_grid = np.where(reserve_ratio > 0, initial_check_grid * reserve_ratio, 0.0)
    cost_grid = initial_check_grid + followon_spend_grid
    scalar_grid = own * (pmv_vals / cost_grid)
    p = win_rate
    L = lose_recovery
    with np.errstate(divide='ignore', invalid='ignore'):
        Z[i, :] = np.where(p > 0, np.maximum(0.0, (target_tvpi / scalar_grid - (1-p)*L) / p), np.nan)

heat_df = pd.DataFrame(Z, index=np.round(own_vals*100,2), columns=np.round(pmv_vals/1e6,1))
fig_heat = px.imshow(heat_df, aspect='auto', origin='lower', labels=dict(x='Entry PMV ($MM)', y='Ownership (%)', color='Req. Winner Company MOIC (×)'), title='Sensitivity: Required Winner Outcomes vs Entry Terms')
fig_heat.update_layout(margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_heat, use_container_width=True)

# -----------------------------
# Table & Download (median run)
# -----------------------------

with st.expander("Show deal table (median TVPI run)"):
    st.dataframe(
        agg['median_run_df'].assign(
            entry_pmv=agg['median_run_df']['entry_pmv'].map(fmt_money),
            initial_check=agg['median_run_df']['initial_check'].map(fmt_money),
            followon_spend=agg['median_run_df']['followon_spend'].map(fmt_money),
            cost_per_deal=agg['median_run_df']['cost_per_deal'].map(fmt_money),
            cash_back=agg['median_run_df']['cash_back'].map(fmt_money),
        )
    )

csv = agg['median_run_df'].to_csv(index=False).encode('utf-8')
st.download_button("Download deal‑level CSV (median run)", data=csv, file_name="deal_outcomes_median_run.csv", mime="text/csv")

# -----------------------------
# Narrative Insight Block
# -----------------------------

st.markdown("---")
st.subheader("Interpretation")
st.markdown(
    """
**Monte Carlo view:** averaging across many draws stabilizes TVPI and DPI expectations and shows how unforgiving
entry terms are. When entry PMVs are high and ownership is modest, the mean/median TVPI tends to sit far below
fund targets unless winner MOICs are extreme—validating the thesis that the math breaks without outlier outcomes.
"""
)

st.caption("Adjust the # of simulations to trade off speed vs stability. 200–500 runs usually gives a tight picture.")
