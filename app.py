import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings

# --- 1. SETTINGS ---
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="IRIS | Global Risk Intelligence",
    layout="wide",
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500&display=swap');

:root {
    --iris-dark:   #0a0e1a;
    --iris-card:   #111827;
    --iris-border: #1e293b;
    --iris-gold:   #c9a84c;
    --iris-gold2:  #f0c96a;
    --iris-red:    #ef4444;
    --iris-green:  #22c55e;
    --iris-blue:   #3b82f6;
    --iris-text:   #e2e8f0;
    --iris-muted:  #64748b;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1525 50%, #0a1020 100%);
    color: var(--iris-text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1525 0%, #111827 100%);
    border-right: 1px solid var(--iris-border);
}
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--iris-muted);
    font-size: 0.78rem;
    letter-spacing: 0.04em;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--iris-gold) !important;
    letter-spacing: -0.01em;
}
h1 { font-size: 2.4rem !important; font-weight: 900 !important; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.2rem !important; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827 0%, #1a2540 100%);
    border: 1px solid var(--iris-border);
    border-radius: 12px;
    padding: 20px 24px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(201,168,76,0.15);
    border-color: var(--iris-gold);
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    color: var(--iris-muted) !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    color: var(--iris-gold2) !important;
    font-size: 1.8rem !important;
}

[data-testid="stTabs"] [role="tablist"] {
    background: #0d1525;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--iris-border);
    gap: 2px;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    color: var(--iris-muted) !important;
    border-radius: 8px;
    padding: 8px 16px;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1e293b, #1a2540) !important;
    color: var(--iris-gold) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--iris-border);
    border-radius: 8px;
    overflow: hidden;
}

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #1a2540, #1e3a5f) !important;
    color: var(--iris-gold) !important;
    border: 1px solid var(--iris-gold) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(135deg, var(--iris-gold), #b8932e) !important;
    color: #0a0e1a !important;
    box-shadow: 0 4px 16px rgba(201,168,76,0.4) !important;
}

[data-testid="stSlider"] [role="slider"] {
    background: var(--iris-gold) !important;
}
.stSelectbox label, .stSlider label, .stNumberInput label, .stFileUploader label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    color: var(--iris-muted) !important;
    text-transform: uppercase;
}

hr { border-color: var(--iris-border) !important; }
.stCaption { color: var(--iris-muted) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.7rem !important; }
.js-plotly-plot .plotly .bg { fill: transparent !important; }
.stImage img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# --- 3. CORE DATA ENGINE ---
def process_data(df, year_col, val_col):
    df = df.rename(columns={year_col: 'year', val_col: 'inflation'})
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['inflation'] = pd.to_numeric(df['inflation'], errors='coerce')
    df = df.dropna(subset=['year', 'inflation']).sort_values('year').reset_index(drop=True)
    mu = df['inflation'].mean()
    sigma = df['inflation'].std()
    df['z_score'] = 0 if sigma == 0 else (df['inflation'] - mu) / sigma
    return df, mu, sigma

def styled_plotly(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,21,37,0.6)',
        font=dict(family='IBM Plex Mono', color='#94a3b8', size=11),
        title_font=dict(family='Playfair Display', color='#c9a84c', size=16),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e293b', borderwidth=1),
        xaxis=dict(gridcolor='#1e293b', zerolinecolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b', zerolinecolor='#1e293b'),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

def styled_mpl_fig():
    fig, ax = plt.subplots(facecolor='#0d1525')
    ax.set_facecolor('#111827')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
    ax.tick_params(colors='#64748b', labelsize=8)
    ax.xaxis.label.set_color('#64748b')
    ax.yaxis.label.set_color('#64748b')
    return fig, ax


# --- 4. SIDEBAR ---
st.sidebar.markdown("""
<div style="text-align:center;padding:12px 0 20px 0;">
<div style="font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:900;color:#c9a84c;letter-spacing:0.08em;">IRIS</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#64748b;letter-spacing:0.2em;text-transform:uppercase;">Global Risk Intelligence</div>
<div style="margin-top:8px;font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#374151;background:#111827;border:1px solid #1e293b;border-radius:6px;padding:4px 10px;display:inline-block;">IRIS_v2_2026</div>
</div>
<hr style="border-color:#1e293b;margin-bottom:16px;">
""", unsafe_allow_html=True)

st.sidebar.title("📥 Data Ingestion Hub")
st.sidebar.markdown("Upload inflation data in **CSV** or **JSON** format.")
uploaded_file = st.sidebar.file_uploader("Primary Dataset", type=['csv', 'json'])

st.sidebar.markdown("""
<hr style="border-color:#1e293b;margin:20px 0 12px 0;">
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#374151;text-align:center;line-height:1.8;">
LEAD SCIENTIST<br>
<span style="color:#c9a84c;font-size:0.68rem;">Stephen Munene</span><br><br>
Inflation Risk Intelligence System<br>
© 2026 — All Rights Reserved
</div>
""", unsafe_allow_html=True)


# --- 5. LANDING SCREEN ---
if not uploaded_file:

    st.markdown("""
<style>
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 40px rgba(201,168,76,0.15), inset 0 0 20px rgba(201,168,76,0.04); }
    50%       { box-shadow: 0 0 80px rgba(201,168,76,0.30), inset 0 0 40px rgba(201,168,76,0.08); }
}
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center;padding:60px 20px 20px 20px;">
<div style="width:120px;height:120px;border-radius:50%;background:radial-gradient(circle,rgba(201,168,76,0.18) 0%,rgba(201,168,76,0.04) 60%,transparent 100%);border:2px solid rgba(201,168,76,0.35);display:flex;align-items:center;justify-content:center;margin:0 auto 32px auto;animation:pulse 3s ease-in-out infinite;">
<span style="font-size:3rem;">🏛️</span>
</div>
<div style="font-family:'Playfair Display',serif;font-size:4rem;font-weight:900;color:#c9a84c;letter-spacing:-0.02em;line-height:1.1;margin-bottom:8px;">IRIS</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;letter-spacing:0.35em;color:#64748b;text-transform:uppercase;margin-bottom:28px;">Inflation Risk Intelligence System</div>
<div style="width:60px;height:1px;background:linear-gradient(90deg,transparent,#c9a84c,transparent);margin:0 auto 28px auto;"></div>
<div style="font-family:'Inter',sans-serif;font-size:0.95rem;color:#94a3b8;max-width:520px;line-height:1.7;margin:0 auto 48px auto;">
A professional-grade platform for macroeconomic risk assessment, anomaly detection, and predictive modeling of inflation dynamics.
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:center;margin-bottom:40px;">
<span style="background:#111827;border:1px solid #1e293b;border-radius:20px;padding:6px 16px;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#64748b;letter-spacing:0.08em;">📈 TREND ANALYSIS</span>
<span style="background:#111827;border:1px solid #1e293b;border-radius:20px;padding:6px 16px;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#64748b;letter-spacing:0.08em;">⚠️ ANOMALY DETECTION</span>
<span style="background:#111827;border:1px solid #1e293b;border-radius:20px;padding:6px 16px;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#64748b;letter-spacing:0.08em;">🔮 ARIMA FORECASTING</span>
<span style="background:#111827;border:1px solid #1e293b;border-radius:20px;padding:6px 16px;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#64748b;letter-spacing:0.08em;">🌐 COMPARATIVE AUDIT</span>
<span style="background:#111827;border:1px solid #1e293b;border-radius:20px;padding:6px 16px;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#64748b;letter-spacing:0.08em;">📄 EXECUTIVE REPORT</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center;margin-bottom:16px;">
<div style="display:inline-block;background:linear-gradient(135deg,#111827,#1a2540);border:1px solid rgba(201,168,76,0.3);border-radius:12px;padding:18px 32px;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#c9a84c;letter-spacing:0.1em;box-shadow:0 0 30px rgba(201,168,76,0.08);">
← UPLOAD A CSV OR JSON DATASET IN THE SIDEBAR TO BEGIN
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center;margin-top:40px;font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#374151;letter-spacing:0.12em;">
LEAD SCIENTIST: STEPHEN MUNENE &nbsp;·&nbsp; IRIS_v2_2026
</div>
""", unsafe_allow_html=True)

    st.stop()


# --- 6. MAIN APP ---
try:
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_json(uploaded_file)

    col_year = st.sidebar.selectbox("Year Column", df_raw.columns, index=0)
    col_val  = st.sidebar.selectbox("Inflation Column", df_raw.columns, index=1)

    df, mu, sigma = process_data(df_raw, col_year, col_val)

    # --- HEADER ---
    st.markdown(f"""
<div style="border-bottom:1px solid #1e293b;padding-bottom:20px;margin-bottom:28px;display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:12px;">
<div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#64748b;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:6px;">Global Risk Intelligence Platform</div>
<div style="font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:900;color:#c9a84c;line-height:1.1;">Inflation Risk Intelligence System</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#64748b;margin-top:8px;">
Dataset: <span style="color:#94a3b8;">{uploaded_file.name}</span>
&nbsp;·&nbsp; Span: <span style="color:#94a3b8;">{int(df['year'].min())} – {int(df['year'].max())}</span>
&nbsp;·&nbsp; Observations: <span style="color:#94a3b8;">{len(df)}</span>
</div>
</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#374151;background:#111827;border:1px solid #1e293b;border-radius:8px;padding:6px 14px;text-align:right;line-height:1.8;">
LEAD SCIENTIST<br><span style="color:#c9a84c;">Stephen Munene</span><br>IRIS_v2_2026
</div>
</div>
""", unsafe_allow_html=True)

    # --- KPIs ---
    latest_val = df['inflation'].iloc[-1]
    latest_z   = df['z_score'].iloc[-1]
    min_val    = df['inflation'].min()
    max_val    = df['inflation'].max()

    if abs(latest_z) > 2:
        status = "🔴 CRITICAL"
    elif abs(latest_z) > 1:
        status = "🟡 WARNING"
    else:
        status = "🟢 STABLE"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Historical Mean (μ)", f"{mu:.2f}%")
    k2.metric("Volatility (σ)",      f"{sigma:.2f}%")
    k3.metric("Min / Max",           f"{min_val:.1f}% / {max_val:.1f}%")
    k4.metric("Current Inflation",   f"{latest_val:.2f}%", delta=f"{latest_z:+.2f} σ")
    k5.metric("Risk Status",         status)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TABS ---
    t1, t2, t3, t4, t5 = st.tabs([
        "📈  Dashboard & Trends",
        "⚠️  Risk Intelligence",
        "🔮  ARIMA Forecasting",
        "🌐  Comparative Audit",
        "📄  Executive Report"
    ])

    # TAB 1
    with t1:
        st.subheader("Market DNA — Trend & Distribution")
        c1, c2 = st.columns([2, 1])

        with c1:
            window = st.slider("Smoothing Window (Years)", 1, 10, 3)
            df['ma'] = df['inflation'].rolling(window=window).mean()
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df['year'], y=df['inflation'], name="Actual", line=dict(color='rgba(148,163,184,0.35)', width=1.5)))
            fig_trend.add_trace(go.Scatter(x=df['year'], y=df['ma'], name=f"{window}-yr Moving Avg", line=dict(color='#c9a84c', width=3)))
            fig_trend.update_layout(title="Inflation Trajectory")
            st.plotly_chart(styled_plotly(fig_trend), use_container_width=True)

        with c2:
            fig_dist, ax_dist = styled_mpl_fig()
            sns.histplot(df['inflation'], kde=True, color="#c9a84c", alpha=0.6, ax=ax_dist)
            ax_dist.axvline(mu, color='#ef4444', linestyle='--', linewidth=1.5, label=f'μ = {mu:.2f}%')
            ax_dist.legend(fontsize=7, facecolor='#111827', edgecolor='#1e293b', labelcolor='#94a3b8')
            ax_dist.set_title("Distribution", color='#c9a84c', fontsize=10, pad=8)
            st.pyplot(fig_dist)
            plt.close()

            skew = df['inflation'].skew()
            skew_label = "Right-skewed (upside shock bias)" if skew > 0.5 else "Left-skewed (downside bias)" if skew < -0.5 else "Approximately symmetric"
            st.markdown(f"""
<div style="background:#111827;border:1px solid #1e293b;border-radius:8px;padding:14px 18px;margin-top:8px;">
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Skewness</div>
<div style="font-family:'Playfair Display',serif;font-size:1.5rem;color:#c9a84c;">{skew:.3f}</div>
<div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">{skew_label}</div>
</div>
""", unsafe_allow_html=True)

    # TAB 2
    with t2:
        st.subheader("Statistical Anomaly Detection")
        z_thresh = st.slider("Risk Threshold (Sigma)", 1.0, 3.0, 2.0, step=0.25)
        outliers = df[df['z_score'].abs() > z_thresh]

        fig_risk = go.Figure()
        y_upper = mu + z_thresh * sigma
        y_lower = mu - z_thresh * sigma
        fig_risk.add_hrect(y0=y_upper, y1=max(df['inflation'].max()*1.05, y_upper+1), fillcolor="rgba(239,68,68,0.06)", line_width=0)
        fig_risk.add_hrect(y0=min(df['inflation'].min()*1.05, y_lower-1), y1=y_lower, fillcolor="rgba(239,68,68,0.06)", line_width=0)
        fig_risk.add_trace(go.Scatter(x=df['year'], y=df['inflation'], name="Inflation Path", line=dict(color='#94a3b8', width=1.5)))
        fig_risk.add_trace(go.Scatter(x=outliers['year'], y=outliers['inflation'], mode='markers',
                                      marker=dict(color='#ef4444', size=12, symbol='x', line=dict(width=2, color='#ef4444')),
                                      name=f"Outlier (>{z_thresh}σ)"))
        fig_risk.add_hline(y=y_upper, line_dash="dot", line_color="#ef4444",
                           annotation_text=f"+{z_thresh}σ = {y_upper:.1f}%", annotation_font_color="#ef4444", annotation_font_size=10)
        fig_risk.add_hline(y=y_lower, line_dash="dot", line_color="#ef4444",
                           annotation_text=f"−{z_thresh}σ = {y_lower:.1f}%", annotation_font_color="#ef4444", annotation_font_size=10)
        fig_risk.add_hline(y=mu, line_dash="dash", line_color="#c9a84c",
                           annotation_text=f"μ = {mu:.1f}%", annotation_font_color="#c9a84c", annotation_font_size=10)
        fig_risk.update_layout(title="Anomaly Detection — Z-Score Corridor")
        st.plotly_chart(styled_plotly(fig_risk, height=440), use_container_width=True)

        st.markdown(f"### Identified Risk Events &nbsp; <span style='font-family:IBM Plex Mono;font-size:0.75rem;color:#ef4444;'>({len(outliers)} events)</span>", unsafe_allow_html=True)
        if len(outliers) > 0:
            st.dataframe(
                outliers[['year','inflation','z_score']]
                .sort_values('z_score', key=abs, ascending=False)
                .reset_index(drop=True)
                .rename(columns={'year':'Year','inflation':'Inflation (%)','z_score':'Z-Score'}),
                use_container_width=True
            )
        else:
            st.success("No anomalies detected at the current threshold.")

    # TAB 3
    with t3:
        st.subheader("ARIMA Predictive Corridor")
        fc1, fc2 = st.columns([1, 3])
        with fc1:
            f_years = st.number_input("Forecast Horizon (Years)", 1, 15, 5)
            arima_p = st.number_input("ARIMA p", 0, 5, 1)
            arima_d = st.number_input("ARIMA d", 0, 2, 1)
            arima_q = st.number_input("ARIMA q", 0, 5, 1)

        with fc2:
            series = df.set_index('year')['inflation']
            try:
                model    = ARIMA(series, order=(int(arima_p), int(arima_d), int(arima_q))).fit()
                forecast = model.get_forecast(steps=f_years).summary_frame()
                forecast.index = np.arange(df['year'].max()+1, df['year'].max()+1+f_years)

                fig_f = go.Figure()
                hist_view = series.tail(20)
                fig_f.add_trace(go.Scatter(x=hist_view.index, y=hist_view.values, name="Recent History", line=dict(color='#94a3b8', width=2)))
                fig_f.add_trace(go.Scatter(x=forecast.index, y=forecast['mean'], name="ARIMA Forecast", line=dict(color='#c9a84c', width=2.5, dash='dot')))
                fig_f.add_trace(go.Scatter(
                    x=np.concatenate([forecast.index, forecast.index[::-1]]),
                    y=np.concatenate([forecast['mean_ci_upper'], forecast['mean_ci_lower'][::-1]]),
                    fill='toself', fillcolor='rgba(201,168,76,0.08)',
                    line=dict(color='rgba(0,0,0,0)'), name='95% Confidence Corridor'
                ))
                fig_f.update_layout(title=f"ARIMA({int(arima_p)},{int(arima_d)},{int(arima_q)}) — {f_years}-Year Outlook")
                st.plotly_chart(styled_plotly(fig_f, height=420), use_container_width=True)

                st.markdown("**Forecast Values**")
                fc_table = forecast[['mean','mean_ci_lower','mean_ci_upper']].copy()
                fc_table.index.name = 'Year'
                fc_table.columns = ['Forecast (%)', 'Lower 95%', 'Upper 95%']
                st.dataframe(fc_table.style.format("{:.2f}"), use_container_width=True)

            except Exception as e:
                st.error(f"Forecasting Error: {e}")
                st.info("Try adjusting the ARIMA parameters (p, d, q) above.")

    # TAB 4
    with t4:
        st.subheader("Geopolitical Comparative Audit")
        comp_file = st.file_uploader("Upload Comparison Dataset (CSV/JSON)", type=['csv','json'], key="comp")

        if comp_file:
            df_c_raw = pd.read_csv(comp_file) if comp_file.name.endswith('.csv') else pd.read_json(comp_file)
            c_y = st.selectbox("Year Column (Comparison)", df_c_raw.columns, key="cy")
            c_v = st.selectbox("Inflation Column (Comparison)", df_c_raw.columns, key="cv")
            df_c, mu_c, sigma_c = process_data(df_c_raw, c_y, c_v)

            ca1, ca2 = st.columns(2)
            with ca1:
                fig_c1 = go.Figure()
                fig_c1.add_trace(go.Scatter(x=df['year'],   y=df['inflation'],   name=uploaded_file.name.replace('.csv','').replace('.json',''), line=dict(color='#c9a84c', width=2)))
                fig_c1.add_trace(go.Scatter(x=df_c['year'], y=df_c['inflation'], name=comp_file.name.replace('.csv','').replace('.json',''),    line=dict(color='#3b82f6', width=2)))
                fig_c1.update_layout(title="Side-by-Side Inflation Trajectories")
                st.plotly_chart(styled_plotly(fig_c1, height=380), use_container_width=True)

            with ca2:
                fig_c2, ax_c2 = styled_mpl_fig()
                sns.kdeplot(df['inflation'],   fill=True, color='#c9a84c', alpha=0.5, label="Primary",    ax=ax_c2)
                sns.kdeplot(df_c['inflation'], fill=True, color='#3b82f6', alpha=0.5, label="Comparison", ax=ax_c2)
                ax_c2.legend(fontsize=7, facecolor='#111827', edgecolor='#1e293b', labelcolor='#94a3b8')
                ax_c2.set_title("Density Comparison", color='#c9a84c', fontsize=10, pad=8)
                st.pyplot(fig_c2)
                plt.close()

            st.markdown("### Statistical Comparison")
            comp_summary = pd.DataFrame({
                "Metric":     ["Mean (%)", "Std Dev (%)", "Min (%)", "Max (%)", "Skewness", "Observations"],
                "Primary":    [f"{mu:.2f}", f"{sigma:.2f}", f"{df['inflation'].min():.2f}", f"{df['inflation'].max():.2f}", f"{df['inflation'].skew():.3f}", len(df)],
                "Comparison": [f"{mu_c:.2f}", f"{sigma_c:.2f}", f"{df_c['inflation'].min():.2f}", f"{df_c['inflation'].max():.2f}", f"{df_c['inflation'].skew():.3f}", len(df_c)],
            })
            st.dataframe(comp_summary, use_container_width=True, hide_index=True)

        else:
            st.markdown("""
<div style="background:#111827;border:1px dashed #1e293b;border-radius:10px;padding:40px;text-align:center;color:#374151;">
<div style="font-size:2rem;margin-bottom:12px;">🌐</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;letter-spacing:0.1em;">UPLOAD A SECOND DATASET ABOVE TO ENABLE COMPARATIVE ANALYSIS</div>
</div>
""", unsafe_allow_html=True)

    # TAB 5
    with t5:
        st.subheader("Strategic Risk Memorandum")

        try:
            report_outliers = outliers
        except NameError:
            report_outliers = df[df['z_score'].abs() > 2.0]
            z_thresh = 2.0

        report_text = f"""INSTITUTIONAL RISK AUDIT — STRATEGIC MEMORANDUM
{'='*60}

SUBJECT  : Automated Inflation Risk Assessment
SOURCE   : {uploaded_file.name}
ANALYST  : IRIS_v2_2026 | Lead Scientist: Stephen Munene
PERIOD   : {int(df['year'].min())} – {int(df['year'].max())}
GENERATED: 2026

{'='*60}
1. HISTORICAL VOLATILITY PROFILE
{'─'*60}
  Historical Average (μ)       : {mu:.4f}%
  Structural Volatility (σ)    : {sigma:.4f}%
  Minimum Observed              : {df['inflation'].min():.4f}%
  Maximum Observed              : {df['inflation'].max():.4f}%
  Skewness (Shock Bias)         : {df['inflation'].skew():.4f}
  Total Observations            : {len(df)}

{'='*60}
2. CURRENT RISK POSTURE
{'─'*60}
  Latest Inflation Reading      : {latest_val:.4f}%
  Z-Score (Standard Deviations) : {latest_z:+.4f}σ
  Risk Classification           : {status}

{'='*60}
3. ANOMALY / BLACK SWAN ANALYSIS
{'─'*60}
  Detection Threshold           : ±{z_thresh:.2f}σ
  Total Risk Breach Events      : {len(report_outliers)}

  Top Risk Events:
"""
        top_events = report_outliers.nlargest(5, 'z_score')[['year','inflation','z_score']]
        for _, row in top_events.iterrows():
            report_text += f"    Year {int(row['year'])}: {row['inflation']:.2f}% (z={row['z_score']:+.2f})\n"

        report_text += f"""
{'='*60}
4. POLICY VERDICT
{'─'*60}
  The current Z-score of {latest_z:+.2f} places the system at
  {status} status. {'Immediate policy review is recommended.' if abs(latest_z) > 2 else 'Continued monitoring advised.' if abs(latest_z) > 1 else 'System operating within normal parameters.'}

  Historical volatility of {sigma:.2f}% {'exceeds' if sigma > 5 else 'is within'} typical
  central bank tolerance bands.

{'='*60}
CONFIDENTIAL — IRIS AUTOMATED INTELLIGENCE OUTPUT
Lead Scientist: Stephen Munene | IRIS_v2_2026
"""

        st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1525,#111827);border:1px solid #1e293b;border-left:3px solid #c9a84c;border-radius:10px;padding:28px 32px;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#94a3b8;line-height:1.9;white-space:pre-wrap;margin-bottom:20px;">{report_text}</div>
""", unsafe_allow_html=True)

        st.download_button(
            "⬇️  Export Risk Memorandum (.txt)",
            data=report_text,
            file_name=f"IRIS_Risk_Report_{int(df['year'].max())}.txt",
            mime="text/plain"
        )

except Exception as e:
    st.error(f"⚠️ Data Processing Error: {e}")
    st.info("Ensure your CSV/JSON headers match the column selections in the sidebar.")