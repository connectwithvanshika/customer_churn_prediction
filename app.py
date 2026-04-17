import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
import json


# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Theme & CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0a0f1e;
    --surface:   #111827;
    --surface2:  #1a2235;
    --border:    rgba(255,255,255,0.07);
    --accent:    #6366f1;
    --accent2:   #06b6d4;
    --danger:    #f43f5e;
    --warn:      #f59e0b;
    --success:   #10b981;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --radius:    14px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stSidebar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 100% !important; }

/* ── Top Nav ── */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 2rem;
    background: rgba(17,24,39,0.95);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 999;
    backdrop-filter: blur(20px);
    margin: 0 -2rem 2rem -2rem;
}
.brand { display: flex; align-items: center; gap: 10px; }
.brand-icon {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.brand-name { font-size: 1.1rem; font-weight: 700; color: var(--text); }
.brand-sub { font-size: 0.7rem; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 22px;
    position: relative; overflow: hidden;
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent-color, var(--accent));
}
.kpi-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.kpi-value { font-size: 2rem; font-weight: 700; color: var(--text); line-height: 1; margin-bottom: 4px; }
.kpi-sub { font-size: 0.75rem; color: var(--muted); }
.kpi-icon { position: absolute; right: 18px; top: 18px; font-size: 1.6rem; opacity: 0.3; }

/* ── Charts ── */
.chart-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.chart-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
}
.chart-title { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; font-weight: 600; }

/* ── Insight Cards ── */
.insights-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 24px; }
.insight-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    border-left: 3px solid var(--ins-color, var(--accent));
}
.insight-title { font-size: 0.82rem; font-weight: 600; margin-bottom: 6px; color: var(--ins-color, var(--accent)); }
.insight-text { font-size: 0.78rem; color: var(--muted); line-height: 1.5; }

/* ── Page Titles ── */
.page-title { font-size: 1.6rem; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.page-subtitle { font-size: 0.85rem; color: var(--muted); margin-bottom: 28px; }

/* ── Form Sections ── */
.form-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 16px;
}
.form-section-title {
    font-size: 0.75rem; font-weight: 600; color: var(--accent2);
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 18px;
    padding-bottom: 10px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px;
}

/* ── Streamlit input overrides ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider { 
    background: var(--surface2) !important; 
    border-color: var(--border) !important;
    color: var(--text) !important;
}
label[data-testid="stWidgetLabel"] p { color: var(--muted) !important; font-size: 0.8rem !important; }

/* ── Primary Button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.9 !important; transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(99,102,241,0.35) !important;
}

/* ── Results Page ── */
.prob-hero {
    text-align: center;
    padding: 40px 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 16px;
}
.prob-pct { font-size: 4.5rem; font-weight: 800; line-height: 1; }
.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 50px;
    font-size: 0.8rem; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
    margin-top: 12px;
}
.risk-high { background: rgba(244,63,94,0.15); color: #f43f5e; border: 1px solid rgba(244,63,94,0.3); }
.risk-low  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.risk-med  { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }

.factor-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex; align-items: center; gap: 12px;
}
.factor-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.factor-name { font-size: 0.88rem; font-weight: 500; flex: 1; }
.factor-badge {
    font-size: 0.68rem; font-weight: 700; padding: 3px 10px;
    border-radius: 50px; text-transform: uppercase; letter-spacing: 0.5px;
}

.rec-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.rec-num { font-size: 0.7rem; color: var(--accent2); font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.rec-title { font-size: 0.9rem; font-weight: 600; margin-bottom: 4px; }
.rec-desc { font-size: 0.78rem; color: var(--muted); line-height: 1.5; }

.summary-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.summary-table tr { border-bottom: 1px solid var(--border); }
.summary-table td { padding: 9px 14px; }
.summary-table td:first-child { color: var(--muted); width: 45%; }
.summary-table td:last-child { font-weight: 500; color: var(--text); }

/* Back button style */
.back-btn > div[data-testid="stButton"] > button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "customer_data" not in st.session_state:
    st.session_state.customer_data = None

# ─── Data & Model Loading ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    paths = [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "Telco-Customer-Churn.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df.dropna(subset=["TotalCharges"], inplace=True)
            return df
    return None

@st.cache_resource
def load_models():
    pkl_dirs = [".", "notebook_&_otherpkl", "models", "pkl"]
    files = {
        "model": ["final_churn_model.pkl", "churn_model.pkl", "model.pkl"],
        "scaler": ["scaler.pkl"],
        "encoders": ["encoders.pkl", "label_encoders.pkl"],
        "threshold": ["threshold.pkl"],
        "feature_order": ["feature_order.pkl", "feature_names.pkl"],
    }
    loaded = {}
    for key, names in files.items():
        for d in pkl_dirs:
            for name in names:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        loaded[key] = pickle.load(f)
                    break
            if key in loaded:
                break
    return loaded

def load_retention_knowledge():
    for p in ["retention_knowledge.json", "data/retention_knowledge.json"]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}

df = load_data()
models = load_models()

# ─── Plotly Theme Helper ──────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94a3b8", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    showlegend=True,
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=10),
        orientation="h",
        yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
)
COLORS = ["#6366f1", "#f43f5e", "#06b6d4", "#10b981", "#f59e0b", "#a78bfa"]

# ─── Navigation ──────────────────────────────────────────────────────────────
def topnav():
    st.markdown("""
    <div class="topnav">
        <div class="brand">
            <div class="brand-icon">🛡️</div>
            <div>
                <div class="brand-name">ChurnGuard AI</div>
                <div class="brand-sub">Customer Intelligence &amp; Retention Platform</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, spacer = st.columns([1, 1, 1, 5])
    with c1:
        if st.button("📊 Analytics Dashboard",
                     type="primary" if st.session_state.page == "dashboard" else "secondary",
                     use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
    with c2:
        if st.button("🔮 Predict Churn",
                     type="primary" if st.session_state.page == "predict" else "secondary",
                     use_container_width=True):
            st.session_state.page = "predict"
            st.rerun()
    with c3:
        if st.button("📋 Results & Insights",
                     type="primary" if st.session_state.page == "results" else "secondary",
                     use_container_width=True):
            st.session_state.page = "results"
            st.rerun()

# ═══════════════════════════════════════════════════════════════
#  PAGE 1 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown('<div class="page-title">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time insights from the Telco Customer Churn dataset (7,043 customers)</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ Dataset not found. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root to enable charts.")
        st.info("Expected KPIs from repository:\n- Total Customers: 7,043\n- Churn Rate: 26.5%\n- Avg Tenure: 32.4 months\n- Month-to-Month customers: 55%")
        return

    churn_yes = df[df["Churn"] == "Yes"]
    churn_no  = df[df["Churn"] == "No"]
    churn_rate = len(churn_yes) / len(df) * 100
    avg_tenure = df["tenure"].mean()
    avg_monthly = df["MonthlyCharges"].mean()
    pct_m2m = (df["Contract"] == "Month-to-month").sum() / len(df) * 100

    # ── 4 KPI Cards ──
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    kpis = [
        ("Total Customers", f"{len(df):,}", "Active accounts in dataset", "🏢", "#6366f1"),
        ("Churn Rate", f"{churn_rate:.1f}%", f"{len(churn_yes):,} customers churned", "📉", "#f43f5e"),
        ("Avg Monthly Charges", f"${avg_monthly:.2f}", "Per customer per month", "💳", "#06b6d4"),
        ("Month-to-Month %", f"{pct_m2m:.1f}%", f"Highest churn-risk segment", "⚠️", "#f59e0b"),
    ]
    for label, val, sub, icon, color in kpis:
        st.markdown(f"""
        <div class="kpi-card" style="--accent-color:{color}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 1: 2 charts ──
    c1, c2 = st.columns(2)

    # Chart 1 — Churn Distribution (Donut)
    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title">Churn Distribution</div>', unsafe_allow_html=True)
        counts = df["Churn"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["No Churn", "Churned"],
            values=[counts.get("No", 0), counts.get("Yes", 0)],
            hole=0.65,
            marker=dict(colors=["#10b981", "#f43f5e"],
                        line=dict(color="rgba(0,0,0,0)", width=0)),
            textfont=dict(size=11, color="white"),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        ))
        fig.add_annotation(text=f"<b>{churn_rate:.1f}%</b><br>Churn",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="white"), xref="paper", yref="paper")
        fig.update_layout(**{**PLOT_LAYOUT, "height": 270,
                             "legend": dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=10),
                                            orientation="h", y=-0.05)})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Chart 2 — Churn by Contract Type
    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title">Churn Rate by Contract Type</div>', unsafe_allow_html=True)
        contract_churn = df.groupby("Contract")["Churn"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100
        ).reset_index()
        contract_churn.columns = ["Contract", "ChurnRate"]
        fig = go.Figure(go.Bar(
            x=contract_churn["Contract"],
            y=contract_churn["ChurnRate"],
            marker=dict(color=["#f43f5e", "#f59e0b", "#10b981"],
                        line=dict(width=0)),
            text=[f"{v:.1f}%" for v in contract_churn["ChurnRate"]],
            textposition="outside",
            textfont=dict(color="white", size=11),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(**{**PLOT_LAYOUT, "height": 270,
                             "showlegend": False,
                             "yaxis": {**PLOT_LAYOUT["yaxis"], "ticksuffix": "%"}})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: 2 charts ──
    c3, c4 = st.columns(2)

    # Chart 3 — Churn by Internet Service
    with c3:
        st.markdown('<div class="chart-card"><div class="chart-title">Churn Rate by Internet Service</div>', unsafe_allow_html=True)
        isp_churn = df.groupby("InternetService")["Churn"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100
        ).reset_index()
        isp_churn.columns = ["InternetService", "ChurnRate"]
        fig = go.Figure(go.Bar(
            x=isp_churn["ChurnRate"],
            y=isp_churn["InternetService"],
            orientation="h",
            marker=dict(color=["#6366f1", "#f43f5e", "#06b6d4"],
                        line=dict(width=0)),
            text=[f"{v:.1f}%" for v in isp_churn["ChurnRate"]],
            textposition="outside",
            textfont=dict(color="white", size=11),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(**{**PLOT_LAYOUT, "height": 270,
                             "showlegend": False,
                             "xaxis": {**PLOT_LAYOUT["xaxis"], "ticksuffix": "%"}})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Chart 4 — Churn by Payment Method
    with c4:
        st.markdown('<div class="chart-card"><div class="chart-title">Churn Rate by Payment Method</div>', unsafe_allow_html=True)
        pay_churn = df.groupby("PaymentMethod")["Churn"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100
        ).reset_index()
        pay_churn.columns = ["PaymentMethod", "ChurnRate"]
        pay_churn["PaymentMethod"] = pay_churn["PaymentMethod"].str.replace(" (automatic)", "", regex=False)
        fig = go.Figure(go.Bar(
            x=pay_churn["PaymentMethod"],
            y=pay_churn["ChurnRate"],
            marker=dict(color=COLORS[:len(pay_churn)],
                        line=dict(width=0)),
            text=[f"{v:.1f}%" for v in pay_churn["ChurnRate"]],
            textposition="outside",
            textfont=dict(color="white", size=10),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(**{**PLOT_LAYOUT, "height": 270,
                             "showlegend": False,
                             "yaxis": {**PLOT_LAYOUT["yaxis"], "ticksuffix": "%"},
                             "xaxis": {**PLOT_LAYOUT["xaxis"],
                                       "tickfont": dict(size=9)}})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: 2 charts ──
    c5, c6 = st.columns(2)

    # Chart 5 — Tenure Distribution by Churn (Histogram overlay)
    with c5:
        st.markdown('<div class="chart-card"><div class="chart-title">Tenure Distribution by Churn</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=churn_no["tenure"], name="No Churn",
            marker_color="#6366f1", opacity=0.7,
            xbins=dict(size=4),
            hovertemplate="Tenure: %{x}<br>Count: %{y}<extra></extra>",
        ))
        fig.add_trace(go.Histogram(
            x=churn_yes["tenure"], name="Churned",
            marker_color="#f43f5e", opacity=0.7,
            xbins=dict(size=4),
            hovertemplate="Tenure: %{x}<br>Count: %{y}<extra></extra>",
        ))
        fig.update_layout(**{**PLOT_LAYOUT, "height": 270,
                             "barmode": "overlay",
                             "xaxis": {**PLOT_LAYOUT["xaxis"], "title": "Tenure (months)"},
                             "legend": dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                                            orientation="h", y=1.05)})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Chart 6 — Monthly Charges vs Churn (Box plot)
    with c6:
        st.markdown('<div class="chart-card"><div class="chart-title">Monthly Charges vs Churn</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=churn_no["MonthlyCharges"], name="No Churn",
            marker_color="#6366f1", line_color="#6366f1",
            fillcolor="rgba(99,102,241,0.15)",
            hovertemplate="No Churn<br>%{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Box(
            y=churn_yes["MonthlyCharges"], name="Churned",
            marker_color="#f43f5e", line_color="#f43f5e",
            fillcolor="rgba(244,63,94,0.15)",
            hovertemplate="Churned<br>%{y:.2f}<extra></extra>",
        ))
        fig.update_layout(**{**PLOT_LAYOUT, "height": 270,
                             "yaxis": {**PLOT_LAYOUT["yaxis"], "tickprefix": "$"},
                             "legend": dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                                            orientation="h", y=1.05)})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Key Insights ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:12px;">🔍 Key Business Insights</div>', unsafe_allow_html=True)

    insights = [
        ("#f43f5e", "Month-to-Month Contract Risk",
         f"M2M customers churn at ~{df[df['Contract']=='Month-to-month']['Churn'].apply(lambda x: 1 if x=='Yes' else 0).mean()*100:.0f}% — nearly 3× higher than two-year plans."),
        ("#f43f5e", "Electronic Check Payment",
         "E-check customers show the highest churn rate among all payment methods (~45%)."),
        ("#f59e0b", "Low Tenure = High Risk",
         "Customers below 12 months are significantly more likely to churn than long-term customers."),
        ("#10b981", "Long-Term Contracts Retain",
         "Two-year contract customers have a churn rate below 5%, showing strong retention power."),
        ("#f43f5e", "No Tech Support Matters",
         "Customers without tech support churn at ~42% vs ~15% for those with support."),
        ("#f59e0b", "High Monthly Charges Hurt",
         "Customers paying above $70/month show elevated churn risk, especially without bundled services."),
    ]
    st.markdown('<div class="insights-grid">', unsafe_allow_html=True)
    for color, title, text in insights:
        st.markdown(f"""
        <div class="insight-card" style="--ins-color:{color}">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 — PREDICT CHURN
# ═══════════════════════════════════════════════════════════════
def page_predict():
    st.markdown('<div class="page-title">Predict Customer Churn</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Fill in the customer profile below to get an AI-powered churn prediction</div>', unsafe_allow_html=True)

    with st.form("churn_form"):
        # ── Demographics ──
        st.markdown('<div class="form-section"><div class="form-section-title">👤 Customer Demographics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: gender = st.selectbox("Gender", ["Male", "Female"])
        with c2: senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        with c3: partner = st.selectbox("Has Partner", ["Yes", "No"])
        with c4: dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Services ──
        st.markdown('<div class="form-section"><div class="form-section-title">📡 Services Subscribed</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        with c2: multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        with c3: internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        with c4: online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

        c5, c6, c7, c8 = st.columns(4)
        with c5: online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        with c6: device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        with c7: tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        with c8: streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

        c9, *_ = st.columns(4)
        with c9: streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Account Info ──
        st.markdown('<div class="form-section"><div class="form-section-title">🗂️ Account Information</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: tenure = st.slider("Tenure (months)", 0, 72, 12)
        with c2: contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with c3: paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with c4: payment = st.selectbox("Payment Method",
                                         ["Electronic check", "Mailed check",
                                          "Bank transfer (automatic)", "Credit card (automatic)"])

        c5, c6, *_ = st.columns(4)
        with c5: monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.01)
        with c6: total = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly), step=0.01)
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("🔮 Run Churn Analysis", use_container_width=True)

    if submitted:
        customer = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        }
        st.session_state.customer_data = customer

        # Try model prediction
        prob = None
        if "model" in models and "scaler" in models:
            try:
                feature_order = models.get("feature_order", list(customer.keys()))
                encoders = models.get("encoders", {})
                row = {}
                for feat in feature_order:
                    val = customer.get(feat, 0)
                    if feat in encoders and hasattr(encoders[feat], "transform"):
                        try:
                            val = int(encoders[feat].transform([val])[0])
                        except Exception:
                            val = 0
                    elif isinstance(val, str):
                        val = 1 if val in ["Yes", "Female"] else 0
                    row[feat] = val

                X = pd.DataFrame([row])[feature_order]
                X_scaled = models["scaler"].transform(X)
                threshold = models.get("threshold", 0.4)
                if hasattr(threshold, "__float__"):
                    threshold = float(threshold)
                else:
                    threshold = 0.4

                prob_arr = models["model"].predict_proba(X_scaled)[0]
                prob = float(prob_arr[1])
            except Exception as e:
                st.warning(f"Model prediction error: {e}. Using rule-based estimate.")

        if prob is None:
            # Rule-based fallback (calibrated from EDA findings)
            risk = 0.1
            if contract == "Month-to-month": risk += 0.35
            elif contract == "One year": risk += 0.1
            if payment == "Electronic check": risk += 0.2
            if tech_support == "No": risk += 0.12
            if internet_service == "Fiber optic": risk += 0.1
            if online_security == "No": risk += 0.08
            if tenure < 12: risk += 0.15
            elif tenure > 48: risk -= 0.1
            if monthly > 70: risk += 0.08
            prob = min(max(risk, 0.02), 0.98)

        st.session_state.prediction_result = {
            "prob": prob,
            "churns": prob >= 0.4,
        }
        st.session_state.page = "results"
        st.rerun()


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 — RESULTS & INSIGHTS
# ═══════════════════════════════════════════════════════════════
def page_results():
    result = st.session_state.prediction_result
    customer = st.session_state.customer_data

    if result is None or customer is None:
        st.info("No prediction yet. Please fill the form on the **Predict Churn** page.")
        if st.button("← Go to Predict Churn"):
            st.session_state.page = "predict"
            st.rerun()
        return

    prob = result["prob"]
    pct = int(prob * 100)
    churns = result["churns"]

    if pct >= 60:
        risk_label = "HIGH RISK"
        risk_class = "risk-high"
        prob_color = "#f43f5e"
    elif pct >= 35:
        risk_label = "MEDIUM RISK"
        risk_class = "risk-med"
        prob_color = "#f59e0b"
    else:
        risk_label = "LOW RISK"
        risk_class = "risk-low"
        prob_color = "#10b981"

    st.markdown('<div class="page-title">Prediction Results & Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">AI-powered churn analysis and retention recommendations</div>', unsafe_allow_html=True)

    # ── Hero + Gauge Row ──
    col_hero, col_gauge = st.columns([1, 1])

    with col_hero:
        churn_label = "Likely to Churn" if churns else "Likely to Stay"
        st.markdown(f"""
        <div class="prob-hero">
            <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">Churn Probability</div>
            <div class="prob-pct" style="color:{prob_color}">{pct}%</div>
            <div style="color:#94a3b8;font-size:0.95rem;margin-top:10px;">{churn_label}</div>
            <div><span class="risk-badge {risk_class}">{risk_label}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 36, "color": prob_color, "family": "DM Sans"}},
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#334155",
                          tickfont=dict(color="#64748b", size=10)),
                bar=dict(color=prob_color, thickness=0.25),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 35], color="rgba(16,185,129,0.12)"),
                    dict(range=[35, 60], color="rgba(245,158,11,0.12)"),
                    dict(range=[60, 100], color="rgba(244,63,94,0.12)"),
                ],
                threshold=dict(line=dict(color=prob_color, width=3), thickness=0.8, value=pct),
            ),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#94a3b8"),
            height=230, margin=dict(l=20, r=20, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Risk Factors + Recommendations ──
    col_risk, col_rec = st.columns(2)

    with col_risk:
        st.markdown('<div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:12px;">🔎 Risk Factors Identified</div>', unsafe_allow_html=True)
        
        factors = []
        if customer.get("Contract") == "Month-to-month":
            factors.append(("Month-to-Month Contract", "#f43f5e", "HIGH", "Highest churn-risk contract type"))
        if customer.get("PaymentMethod") == "Electronic check":
            factors.append(("Electronic Check Payment", "#f43f5e", "HIGH", "Associated with 45% churn rate"))
        if customer.get("TechSupport") == "No":
            factors.append(("No Tech Support", "#f59e0b", "MEDIUM", "Churn rate 42% vs 15% with support"))
        if customer.get("InternetService") == "Fiber optic":
            factors.append(("Fiber Optic Internet", "#f59e0b", "MEDIUM", "Higher charges drive dissatisfaction"))
        if customer.get("OnlineSecurity") == "No":
            factors.append(("No Online Security", "#94a3b8", "LOW", "Security bundle reduces churn"))
        if customer.get("PaperlessBilling") == "Yes":
            factors.append(("Paperless Billing", "#94a3b8", "LOW", "Correlated with higher churn"))
        if customer.get("tenure", 72) < 12:
            factors.append(("Low Tenure < 12 months", "#f43f5e", "HIGH", "New customers churn at highest rate"))

        if not factors:
            factors = [("Strong retention profile", "#10b981", "LOW", "This customer has low churn risk")]

        for name, color, level, detail in factors[:6]:
            level_bg = {"HIGH": "rgba(244,63,94,0.15)", "MEDIUM": "rgba(245,158,11,0.15)", "LOW": "rgba(148,163,184,0.1)"}
            st.markdown(f"""
            <div class="factor-card">
                <div class="factor-dot" style="background:{color}"></div>
                <div>
                    <div class="factor-name">{name}</div>
                    <div style="font-size:0.72rem;color:#64748b;margin-top:2px">{detail}</div>
                </div>
                <span class="factor-badge" style="background:{level_bg.get(level,'rgba(148,163,184,0.1)')};color:{color}">{level}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_rec:
        st.markdown('<div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:12px;">💡 Retention Recommendations</div>', unsafe_allow_html=True)

        recs = []
        if customer.get("Contract") == "Month-to-month":
            recs.append(("Offer Contract Upgrade", "Provide a 15–20% discount for switching to a 1 or 2-year contract. Long-term contracts reduce churn rate by over 80%."))
        if customer.get("PaymentMethod") == "Electronic check":
            recs.append(("Switch to Auto-Pay", "Offer a $5/month discount to switch to automatic bank transfer or credit card — reduces payment friction significantly."))
        if customer.get("TechSupport") == "No":
            recs.append(("Bundle Tech Support", "Offer tech support at a reduced add-on rate. Customers with support churn at ~15% vs 42% without."))
        if customer.get("OnlineSecurity") == "No":
            recs.append(("Add Security Bundle", "Offer online security as an affordable add-on. Bundled service customers show significantly lower churn rates."))
        if customer.get("tenure", 72) < 12:
            recs.append(("Early Loyalty Reward", "Introduce a loyalty bonus at the 6-month and 12-month marks to anchor new customers to the brand."))
        recs.append(("Personalized Outreach", "Assign a dedicated account manager for proactive check-ins before the next billing cycle."))

        for i, (title, desc) in enumerate(recs[:4], 1):
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-num">Recommendation {i}</div>
                <div class="rec-title">{title}</div>
                <div class="rec-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Customer Summary Table ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:12px;">📋 Customer Profile Summary</div>', unsafe_allow_html=True)

    rows = [
        ("Contract", customer.get("Contract", "—")),
        ("Tenure", f"{customer.get('tenure', '—')} months"),
        ("Monthly Charges", f"${customer.get('MonthlyCharges', 0):.2f}"),
        ("Payment Method", customer.get("PaymentMethod", "—")),
        ("Internet Service", customer.get("InternetService", "—")),
        ("Tech Support", customer.get("TechSupport", "—")),
        ("Online Security", customer.get("OnlineSecurity", "—")),
        ("Senior Citizen", "Yes" if customer.get("SeniorCitizen") == 1 else "No"),
        ("Partner", customer.get("Partner", "—")),
        ("Churn Probability", f"{pct}% ({risk_label})"),
    ]

    html = '<div class="chart-card"><table class="summary-table">'
    for k, v in rows:
        html += f"<tr><td>{k}</td><td>{v}</td></tr>"
    html += "</table></div>"
    st.markdown(html, unsafe_allow_html=True)

    # ── Action Buttons ──
    st.markdown("<br>", unsafe_allow_html=True)
    btn1, btn2, btn3 = st.columns([1, 1, 4])
    with btn1:
        if st.button("← Predict Another Customer", use_container_width=True):
            st.session_state.prediction_result = None
            st.session_state.customer_data = None
            st.session_state.page = "predict"
            st.rerun()
    with btn2:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()


# ═══════════════════════════════════════════════════════════════
#  RENDER
# ═══════════════════════════════════════════════════════════════
topnav()

if st.session_state.page == "dashboard":
    page_dashboard()
elif st.session_state.page == "predict":
    page_predict()
elif st.session_state.page == "results":
    page_results()
