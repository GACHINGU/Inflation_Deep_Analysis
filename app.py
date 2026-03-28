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
        plt.style.use('ggplot') # Fallback to ggplot if all else fails
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings

# --- 1. SETTINGS ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="IRIS | Global Risk Intelligence", layout="wide", page_icon="🏛️")

# --- 2. CORE DATA ENGINE ---
def process_data(df, year_col, val_col):
    """Universal processor for any inflation dataset."""
    df = df.rename(columns={year_col: 'year', val_col: 'inflation'})
    # Ensure numeric types
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['inflation'] = pd.to_numeric(df['inflation'], errors='coerce')
    # Drop NAs and sort
    df = df.dropna(subset=['year', 'inflation']).sort_values('year').reset_index(drop=True)
    
    mu = df['inflation'].mean()
    sigma = df['inflation'].std()
    # Handle zero sigma to avoid division by zero
    if sigma == 0:
        df['z_score'] = 0
    else:
        df['z_score'] = (df['inflation'] - mu) / sigma
    return df, mu, sigma

# --- 3. SIDEBAR: DATA INGESTION ---
st.sidebar.title("📥 Data Ingestion Hub")
st.sidebar.markdown("Upload inflation data (CSV/JSON).")

uploaded_file = st.sidebar.file_uploader("Primary Dataset", type=['csv', 'json'])

# --- 4. APP LOGIC ---
if uploaded_file:
    try:
        # Load Data
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_json(uploaded_file)
        
        # Column Selection
        col_year = st.sidebar.selectbox("Year Column", df_raw.columns, index=0)
        col_val = st.sidebar.selectbox("Inflation Column", df_raw.columns, index=1)
        
        # Process
        df, mu, sigma = process_data(df_raw, col_year, col_val)
        
        # --- HEADER ---
        st.title("🏛️ Inflation Risk Intelligence System (IRIS)")
        st.caption(f"Analyzing Data DNA from {int(df['year'].min())} to {int(df['year'].max())}")
        
        # KPI Row
        latest_val = df['inflation'].iloc[-1]
        latest_z = df['z_score'].iloc[-1]
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Historical Mean (μ)", f"{mu:.2f}%")
        kpi2.metric("Volatility (σ)", f"{sigma:.2f}%")
        kpi3.metric("Current Inflation", f"{latest_val}%", delta=f"{latest_z:.2f} σ")
        
        status = "🔴 CRITICAL" if abs(latest_z) > 2 else "🟡 WARNING" if abs(latest_z) > 1 else "🟢 STABLE"
        kpi4.metric("Risk Status", status)

        # --- TABS ---
        t1, t2, t3, t4, t5 = st.tabs([
            "Dashboard & Trends", 
            "Risk Intelligence (2σ)", 
            "ARIMA Forecasting", 
            "Comparative Audit", 
            "Executive Report"
        ])

        # TAB 1: DASHBOARD & TRENDS
        with t1:
            st.subheader("Market DNA: Trend & Distribution")
            c1, c2 = st.columns([2, 1])
            
            with c1:
                window = st.slider("Smoothing Window (Years)", 1, 10, 3)
                df['ma'] = df['inflation'].rolling(window=window).mean()
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=df['year'], y=df['inflation'], name="Actual", line=dict(color='lightgrey')))
                fig_trend.add_trace(go.Scatter(x=df['year'], y=df['ma'], name=f"{window}yr Moving Avg", line=dict(color='#1f77b4', width=3)))
                fig_trend.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=400)
                st.plotly_chart(fig_trend, use_container_width=True)
                
            with c2:
                fig_dist, ax_dist = plt.subplots()
                sns.histplot(df['inflation'], kde=True, color="#1f77b4", ax=ax_dist)
                ax_dist.axvline(mu, color='red', linestyle='--')
                st.pyplot(fig_dist)
                st.write(f"**Skewness:** {df['inflation'].skew():.2f}")

        # TAB 2: RISK INTELLIGENCE
        with t2:
            st.subheader("Statistical Anomaly Detection")
            z_thresh = st.slider("Risk Threshold (Sigma)", 1.0, 3.0, 2.0)
            
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(x=df['year'], y=df['inflation'], name="Inflation Path", line=dict(color='black', width=1)))
            
            # Highlight Outliers
            outliers = df[df['z_score'].abs() > z_thresh]
            fig_risk.add_trace(go.Scatter(x=outliers['year'], y=outliers['inflation'], mode='markers', 
                                          marker=dict(color='red', size=12, symbol='x'), name="Outlier Event"))
            
            fig_risk.add_hline(y=mu + z_thresh*sigma, line_dash="dash", line_color="red")
            fig_risk.add_hline(y=mu - z_thresh*sigma, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig_risk, use_container_width=True)
            st.write("### Identified 'Black Swan' Events")
            st.dataframe(outliers[['year', 'inflation', 'z_score']].sort_values('z_score', ascending=False))

        # TAB 3: FORECASTING
        with t3:
            st.subheader("ARIMA Predictive Corridor")
            f_years = st.number_input("Forecast Years", 1, 10, 5)
            
            series = df.set_index('year')['inflation']
            try:
                model = ARIMA(series, order=(1,1,1)).fit()
                forecast = model.get_forecast(steps=f_years).summary_frame()
                forecast.index = np.arange(df['year'].max() + 1, df['year'].max() + 1 + f_years)
                
                fig_f = go.Figure()
                hist_view = series.tail(20)
                fig_f.add_trace(go.Scatter(x=hist_view.index, y=hist_view.values, name="Recent History", line=dict(color='blue')))
                fig_f.add_trace(go.Scatter(x=forecast.index, y=forecast['mean'], name="Forecast", line=dict(color='red', dash='dot')))
                fig_f.add_trace(go.Scatter(x=np.concatenate([forecast.index, forecast.index[::-1]]),
                                             y=np.concatenate([forecast['mean_ci_upper'], forecast['mean_ci_lower'][::-1]]),
                                             fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(0,0,0,0)'), name='95% Corridor'))
                st.plotly_chart(fig_f, use_container_width=True)
            except Exception as e:
                st.error(f"Forecasting Error: {e}")

        # TAB 4: COMPARATIVE AUDIT
        with t4:
            st.subheader("Geopolitical Comparative Audit")
            comp_file = st.file_uploader("Upload Comparison Dataset", type=['csv', 'json'], key="comp")
            
            if comp_file:
                df_c_raw = pd.read_csv(comp_file) if comp_file.name.endswith('.csv') else pd.read_json(comp_file)
                c_y = st.selectbox("Year Col (Comp)", df_c_raw.columns)
                c_v = st.selectbox("Inflation Col (Comp)", df_c_raw.columns)
                df_c, mu_c, sigma_c = process_data(df_c_raw, c_y, c_v)
                
                ca1, ca2 = st.columns(2)
                with ca1:
                    fig_c1 = go.Figure()
                    fig_c1.add_trace(go.Scatter(x=df['year'], y=df['inflation'], name="Primary"))
                    fig_c1.add_trace(go.Scatter(x=df_c['year'], y=df_c['inflation'], name="Comparison"))
                    st.plotly_chart(fig_c1, use_container_width=True)
                with ca2:
                    fig_c2, ax_c2 = plt.subplots()
                    sns.kdeplot(df['inflation'], fill=True, label="Primary", ax=ax_c2)
                    sns.kdeplot(df_c['inflation'], fill=True, label="Comparison", ax=ax_c2)
                    ax_c2.legend()
                    st.pyplot(fig_c2)

        # TAB 5: REPORT
        with t5:
            st.subheader("Strategic Risk Memorandum")
            report = f"""
            ### Institutional Risk Audit
            **Subject:** Automated Inflation Risk Assessment
            **Metric:** {uploaded_file.name}
            
            **1. Historical Volatility Profile**
            - Historical Average: {mu:.2f}%
            - Structural Volatility (Sigma): {sigma:.2f}%
            - Skewness (Shock Bias): {df['inflation'].skew():.2f}
            
            **2. Outlier Analysis**
            A total of {len(outliers)} risk breaches were detected using a {z_thresh}-sigma threshold.
            
            **3. Policy Verdict**
            The current Z-score of {latest_z:.2f} indicates that the system is **{status}**.
            """
            st.markdown(report)
            st.download_button("Export Report", report, file_name="risk_report.txt")

    except Exception as e:
        st.error(f"Data Processing Error: {e}")
        st.info("Check if your CSV headers match the selections in the sidebar.")

else:
    st.info("💡 Please upload an inflation dataset (CSV/JSON) in the sidebar to initialize IRIS.")
   