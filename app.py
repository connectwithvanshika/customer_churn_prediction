import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Intelligence System",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%236d28d9'/><text x='50%' y='58%' dominant-baseline='middle' text-anchor='middle' font-size='18' fill='white'>C</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────
if "dark" not in st.session_state: st.session_state.dark = True
if "page" not in st.session_state: st.session_state.page = "Dashboard"

D = st.session_state.dark   # shorthand

# ─────────────────────────────────────────────────────────────────
#  DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────
if D:
    BG       = "#0a0b12"
    SURF     = "#111220"
    SURF2    = "#181a2e"
    BORDER   = "#222540"
    TEXT     = "#e4e6f0"
    SUB      = "#8890b0"
    SUBTLE   = "#4a5070"
    A1       = "#6d28d9"    # violet
    A2       = "#0891b2"    # cyan
    A3       = "#be185d"    # rose
    GRAD     = "linear-gradient(135deg,#6d28d9 0%,#0891b2 55%,#be185d 100%)"
    NAV_BG   = "rgba(10,11,18,0.88)"
    CARD_SHD = "0 0 0 1px #222540, 0 8px 32px rgba(0,0,0,.5)"
    CARD_HOV = "0 0 0 1px #6d28d988, 0 12px 40px rgba(109,40,217,.2)"
    PBG      = "#0a0b12"
    PPAPER   = "#111220"
    GRID     = "#1a1d30"
else:
    BG       = "#f5f6fc"
    SURF     = "#ffffff"
    SURF2    = "#eef0f8"
    BORDER   = "#dde0f0"
    TEXT     = "#111220"
    SUB      = "#5a607a"
    SUBTLE   = "#9298b0"
    A1       = "#6d28d9"
    A2       = "#0284c7"
    A3       = "#be185d"
    GRAD     = "linear-gradient(135deg,#6d28d9 0%,#0284c7 55%,#be185d 100%)"
    NAV_BG   = "rgba(255,255,255,0.92)"
    CARD_SHD = "0 0 0 1px #dde0f0, 0 4px 24px rgba(0,0,0,.07)"
    CARD_HOV = "0 0 0 1px #6d28d966, 0 8px 32px rgba(109,40,217,.12)"
    PBG      = "#f5f6fc"
    PPAPER   = "#ffffff"
    GRID     = "#e8eaf5"

# ─────────────────────────────────────────────────────────────────
#  DATA & MODEL
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():     return joblib.load("final_churn_model.pkl")
@st.cache_resource
def load_threshold(): return joblib.load("threshold.pkl")

@st.cache_data
def load_data():
    d = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    d["TotalCharges"] = d["TotalCharges"].fillna(d["TotalCharges"].median())
    d["ChurnBinary"]  = (d["Churn"] == "Yes").astype(int)
    return d

model     = load_model()
threshold = float(load_threshold())
df        = load_data()

total_cust = len(df)
churn_n    = (df["Churn"] == "Yes").sum()
churn_r    = churn_n / total_cust * 100
retain_r   = 100 - churn_r
avg_mth    = df["MonthlyCharges"].mean()
avg_ten    = df["tenure"].mean()

# ─────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700;800&display=swap');

/* ── Reset ── */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}

html,body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"]>.main,
section.main{{
  background:{BG}!important;
  color:{TEXT}!important;
  font-family:'Inter',sans-serif!important;
}}

/* ── Hide Streamlit chrome + sidebar arrow ── */
#MainMenu,footer,header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"]{{display:none!important;}}
[data-testid="collapsedControl"]{{display:none!important;}}

/* ── Main content padding ── */
.block-container{{
  padding:2.5rem 3rem 5rem!important;
  max-width:1480px!important;
}}

/* ═══════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════ */
[data-testid="stSidebar"]{{
  background:{SURF}!important;
  border-right:1px solid {BORDER}!important;
  min-width:240px!important;
  max-width:260px!important;
}}
[data-testid="stSidebar"] *{{
  font-family:'Inter',sans-serif!important;
}}

/* hide the radio widget — we use buttons instead */
[data-testid="stSidebar"] [data-testid="stRadio"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"]{{display:none!important;}}

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton>button{{
  width:100%!important;
  text-align:left!important;
  padding:.6rem 1rem!important;
  border-radius:10px!important;
  background:transparent!important;
  border:1px solid transparent!important;
  color:{TEXT}!important;
  font-family:'Inter',sans-serif!important;
  font-size:.85rem!important;
  font-weight:500!important;
  letter-spacing:.01em!important;
  text-transform:none!important;
  margin:0!important;
  margin-bottom:.15rem!important;
  animation:none!important;
  transition:background .18s,border-color .18s,color .18s!important;
  box-shadow:none!important;
}}
[data-testid="stSidebar"] .stButton>button:hover{{
  background:{"rgba(255,255,255,.08)" if D else "rgba(0,0,0,.06)"}!important;
  border-color:{BORDER}!important;
  color:{TEXT}!important;
  transform:none!important;
  box-shadow:none!important;
}}
[data-testid="stSidebar"] .stButton>button::after{{display:none!important;}}

/* Active sidebar nav button */
[data-testid="stSidebar"] .sb-active .stButton>button{{
  background:{"rgba(109,40,217,.18)" if D else "rgba(109,40,217,.1)"}!important;
  border-color:{"rgba(109,40,217,.4)" if D else "rgba(109,40,217,.28)"}!important;
  color:{"#c4b5fd" if D else A1}!important;
  font-weight:700!important;
  box-shadow:none!important;
}}
[data-testid="stSidebar"] .sb-active .stButton>button:hover{{
  transform:none!important;box-shadow:none!important;
  color:{"#c4b5fd" if D else A1}!important;
}}

/* Theme toggle button in sidebar */
[data-testid="stSidebar"] .sb-theme .stButton>button{{
  width:100%!important;
  padding:.55rem 1rem!important;
  border-radius:10px!important;
  background:{"rgba(109,40,217,.14)" if D else "rgba(109,40,217,.08)"}!important;
  border:1px solid {"rgba(109,40,217,.35)" if D else "rgba(109,40,217,.22)"}!important;
  color:{"#c4b5fd" if D else A1}!important;
  font-family:'Inter',sans-serif!important;
  font-size:.78rem!important;font-weight:600!important;
  text-transform:none!important;letter-spacing:.02em!important;
  margin-top:.5rem!important;
  animation:none!important;box-shadow:none!important;
}}
[data-testid="stSidebar"] .sb-theme .stButton>button:hover{{
  background:{"rgba(109,40,217,.25)" if D else "rgba(109,40,217,.14)"}!important;
  box-shadow:0 0 12px rgba(109,40,217,.3)!important;
  transform:none!important;
  color:{"#c4b5fd" if D else A1}!important;
}}
[data-testid="stSidebar"] .sb-theme .stButton>button::after{{display:none!important;}}

/* ════════════════════════════════════════════
   NAVBAR
════════════════════════════════════════════ */
.ccis-nav{{
  position:fixed;top:0;left:0;right:0;z-index:9999;
  height:60px;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 2.5rem;
  background:{NAV_BG};
  backdrop-filter:saturate(180%) blur(20px);
  -webkit-backdrop-filter:saturate(180%) blur(20px);
  border-bottom:1px solid {BORDER};
  box-shadow:0 1px 0 {BORDER},0 4px 24px rgba(0,0,0,{".4" if D else ".06"});
}}

/* Brand (left) */
.ccis-brand{{
  display:flex;align-items:center;gap:.75rem;
  text-decoration:none;flex-shrink:0;
}}
.ccis-brand-mark{{
  width:32px;height:32px;border-radius:8px;
  background:{GRAD};
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 0 16px rgba(109,40,217,{".55" if D else ".3"});
  flex-shrink:0;
}}
.ccis-brand-mark-inner{{
  width:14px;height:14px;border-radius:3px;
  border:2px solid rgba(255,255,255,.9);
  position:relative;
}}
.ccis-brand-mark-inner::after{{
  content:'';
  position:absolute;
  top:50%;left:50%;
  transform:translate(-50%,-50%);
  width:5px;height:5px;border-radius:50%;
  background:white;
}}
.ccis-brand-name{{
  font-family:'Poppins',sans-serif;
  font-size:.88rem;font-weight:700;
  color:{TEXT};letter-spacing:-.01em;white-space:nowrap;
  line-height:1.2;
}}
.ccis-brand-sub{{
  font-size:.58rem;font-weight:500;
  color:{SUB};letter-spacing:.04em;text-transform:uppercase;
}}

/* Nav links (centre) */
.ccis-links{{
  display:flex;align-items:center;gap:.2rem;
  position:absolute;left:50%;transform:translateX(-50%);
}}
.ccis-link{{
  padding:.42rem 1.1rem;
  border-radius:8px;
  font-family:'Inter',sans-serif;
  font-size:.8rem;font-weight:600;
  color:{"#ffffff" if D else "#000000"};
  transition:color .18s,background .18s,box-shadow .18s;
  white-space:nowrap;letter-spacing:.01em;
  border:1px solid transparent;
}}
.ccis-link:hover{{
  color:{"#000000" if D else "#ffffff"};
  background:{"rgba(255,255,255,.1)" if D else "rgba(0,0,0,.07)"};
  border-color:{BORDER};
}}
.ccis-link.active{{
  color:{"#ffffff" if D else "#000000"};
  font-weight:700;
  background:{"rgba(255,255,255,.14)" if D else "rgba(0,0,0,.1)"};
  border-color:{"rgba(255,255,255,.28)" if D else "rgba(0,0,0,.22)"};
  box-shadow:{"0 0 0 1px rgba(255,255,255,.12)" if D else "0 0 0 1px rgba(0,0,0,.1)"};
}}

/* Right controls */
.ccis-right{{
  display:flex;align-items:center;gap:.9rem;flex-shrink:0;
}}
.ccis-status{{
  display:flex;align-items:center;gap:.45rem;
  font-size:.7rem;font-weight:600;
  color:{"#4ade80" if D else "#16a34a"};
  letter-spacing:.04em;text-transform:uppercase;
}}
.ccis-status-dot{{
  width:7px;height:7px;border-radius:50%;
  background:{"#4ade80" if D else "#16a34a"};
  box-shadow:0 0 8px {"rgba(74,222,128,.7)" if D else "rgba(22,163,74,.5)"};
  animation:blink 2.4s ease-in-out infinite;
}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}

.ccis-toggle{{
  display:flex;align-items:center;gap:.5rem;
  padding:.34rem .85rem;
  border-radius:999px;
  border:1px solid {BORDER};
  background:{"rgba(255,255,255,.05)" if D else "rgba(0,0,0,.04)"};
  font-family:'Inter',sans-serif;
  font-size:.72rem;font-weight:600;
  color:{TEXT};cursor:pointer;
  transition:border-color .2s,background .2s,box-shadow .2s;
  white-space:nowrap;
}}
.ccis-toggle:hover{{
  border-color:{A1};
  box-shadow:0 0 12px rgba(109,40,217,.3);
  background:{"rgba(109,40,217,.12)" if D else "rgba(109,40,217,.07)"};
}}
.ccis-toggle-icon{{
  width:18px;height:10px;
  border-radius:999px;
  background:{"#6d28d9" if D else "#d1d5db"};
  position:relative;
  transition:background .3s;
}}
.ccis-toggle-icon::after{{
  content:'';
  position:absolute;
  top:2px;
  left:{"calc(100% - 8px - 2px)" if D else "2px"};
  width:6px;height:6px;border-radius:50%;
  background:#fff;
  transition:left .3s;
  box-shadow:0 1px 3px rgba(0,0,0,.3);
}}

/* ════════════════════════════════════════════
   ANIMATIONS
════════════════════════════════════════════ */
@keyframes gradmv{{0%,100%{{background-position:0% 50%}}50%{{background-position:100% 50%}}}}
@keyframes heroIn{{from{{opacity:0;transform:translateY(-20px)}}to{{opacity:1;transform:none}}}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:none}}}}
@keyframes shimmer{{0%{{background-position:-700px 0}}100%{{background-position:700px 0}}}}

/* ════════════════════════════════════════════
   HERO SECTION
════════════════════════════════════════════ */
.hero-section{{
  padding:2.5rem 0 2rem;
  text-align:center;
  animation:heroIn .65s cubic-bezier(.22,1,.36,1) both;
}}
.hero-eyebrow{{
  display:inline-block;
  font-family:'Inter',sans-serif;
  font-size:.62rem;font-weight:700;letter-spacing:.2em;
  text-transform:uppercase;color:{A2};
  margin-bottom:1.1rem;
  padding:.3rem .85rem;
  border:1px solid {"rgba(8,145,178,.3)" if D else "rgba(2,132,199,.25)"};
  border-radius:4px;
  background:{"rgba(8,145,178,.08)" if D else "rgba(2,132,199,.06)"};
}}
.hero-heading{{
  font-family:'Poppins',sans-serif;
  font-size:clamp(1.9rem,3.6vw,3rem);
  font-weight:800;line-height:1.1;
  color:{TEXT};
  margin-bottom:.75rem;
  letter-spacing:-.03em;
}}
.hero-heading span{{
  background:{GRAD};background-size:200% 200%;
  animation:gradmv 5s ease infinite;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.hero-sub{{
  font-size:.94rem;font-weight:400;
  color:{SUB};max-width:480px;margin:0 auto;line-height:1.7;
}}

/* ════════════════════════════════════════════
   SECTION LABELS
════════════════════════════════════════════ */
.sec-label{{
  font-family:'Inter',sans-serif;
  font-size:.62rem;font-weight:700;letter-spacing:.18em;
  text-transform:uppercase;color:{A2};
  display:flex;align-items:center;gap:.75rem;
  margin:2.2rem 0 1rem;
}}
.sec-label::after{{
  content:'';flex:1;height:1px;
  background:linear-gradient(90deg,{BORDER},transparent);
}}

/* ════════════════════════════════════════════
   KPI CARDS
════════════════════════════════════════════ */
.kpi-card{{
  background:{SURF};
  border:1px solid {BORDER};
  border-radius:16px;
  padding:1.5rem 1.6rem;
  position:relative;overflow:hidden;
  transition:box-shadow .25s,border-color .25s,transform .25s;
  animation:fadeUp .5s cubic-bezier(.22,1,.36,1) both;
  box-shadow:{CARD_SHD};
}}
.kpi-card:hover{{
  transform:translateY(-3px);
  box-shadow:{CARD_HOV};
  border-color:{A1}55;
}}
.kpi-accent{{
  position:absolute;top:-28px;right:-28px;
  width:90px;height:90px;border-radius:50%;
  pointer-events:none;opacity:{".35" if D else ".18"};
}}
.kpi-label{{
  font-size:.62rem;font-weight:700;letter-spacing:.15em;
  text-transform:uppercase;color:{SUB};
  margin-bottom:.5rem;
}}
.kpi-value{{
  font-family:'Poppins',sans-serif;
  font-size:2rem;font-weight:800;
  color:{TEXT};letter-spacing:-.035em;line-height:1;
  margin-bottom:.4rem;
}}
.kpi-value.grad{{
  background:{GRAD};background-size:200% 200%;
  animation:gradmv 5s ease infinite;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.kpi-meta{{font-size:.72rem;color:{SUBTLE};}}

/* ════════════════════════════════════════════
   CHART WRAPPER
════════════════════════════════════════════ */
[data-testid="stPlotlyChart"]{{
  background:{SURF}!important;
  border:1px solid {BORDER}!important;
  border-radius:16px!important;
  padding:.15rem!important;
  box-shadow:{CARD_SHD}!important;
  transition:box-shadow .25s,border-color .25s!important;
}}
[data-testid="stPlotlyChart"]:hover{{
  box-shadow:{CARD_HOV}!important;
  border-color:{A1}55!important;
}}

/* ════════════════════════════════════════════
   FORM COLUMN CARDS
════════════════════════════════════════════ */
[data-testid="stHorizontalBlock"]>div{{
  background:{SURF}!important;
  border:1px solid {BORDER}!important;
  border-radius:16px!important;
  padding:1.4rem 1.3rem!important;
  box-shadow:{CARD_SHD}!important;
  transition:box-shadow .25s,border-color .25s,transform .25s!important;
}}
[data-testid="stHorizontalBlock"]>div:hover{{
  box-shadow:{CARD_HOV}!important;
  border-color:{A1}55!important;
  transform:translateY(-2px)!important;
}}

/* ── Widget labels ── */
[data-testid="stWidgetLabel"] p{{
  font-family:'Inter',sans-serif!important;
  font-size:.62rem!important;font-weight:700!important;
  letter-spacing:.12em!important;text-transform:uppercase!important;
  color:{SUB}!important;
}}

/* ── Selectbox ── */
[data-testid="stSelectbox"]>div>div{{
  background:{SURF2}!important;border:1px solid {BORDER}!important;
  border-radius:8px!important;color:{TEXT}!important;
  font-family:'Inter',sans-serif!important;font-size:.85rem!important;
}}
[data-testid="stSelectbox"] svg{{fill:{SUB}!important;}}
[data-testid="stSelectbox"] ul{{
  background:{SURF}!important;border:1px solid {BORDER}!important;
  border-radius:10px!important;box-shadow:{CARD_SHD}!important;
}}
[data-testid="stSelectbox"] li{{
  color:{TEXT}!important;font-family:'Inter',sans-serif!important;font-size:.84rem!important;
}}

/* ── Slider ── */
[data-testid="stSlider"]>div>div>div{{
  background:{SURF2}!important;border-radius:99px!important;
}}
[data-testid="stSlider"] [role="slider"]{{
  background:{A1}!important;border:none!important;
  box-shadow:0 0 10px rgba(109,40,217,.55)!important;
}}
[data-testid="stSlider"] p{{
  color:{A1}!important;font-weight:600!important;font-family:'Inter',sans-serif!important;
}}

/* ════════════════════════════════════════════
   FUNCTIONAL NAV BUTTONS — strip styling
════════════════════════════════════════════ */
div[data-testid="column"].nb .stButton>button,
div[data-testid="column"].tb .stButton>button{{
  width:auto!important;padding:.4rem 1.1rem!important;
  font-family:'Inter',sans-serif!important;
  font-size:.8rem!important;font-weight:600!important;
  letter-spacing:.01em!important;text-transform:none!important;
  border-radius:8px!important;margin-top:0!important;
  animation:none!important;position:static!important;
  transition:color .18s,background .18s,border-color .18s,box-shadow .18s!important;
}}
/* default nav button */
div[data-testid="column"].nb .stButton>button{{
  background:transparent!important;
  border:1px solid {BORDER}!important;
  color:{"#ffffff" if D else "#000000"}!important;
  box-shadow:none!important;
}}
div[data-testid="column"].nb .stButton>button:hover{{
  color:{"#ffffff" if D else "#000000"}!important;
  background:{"rgba(255,255,255,.1)" if D else "rgba(0,0,0,.07)"}!important;
  border-color:{"rgba(255,255,255,.28)" if D else "rgba(0,0,0,.22)"}!important;
  box-shadow:none!important;transform:none!important;
}}
/* active nav button — dark pill, bold, border highlight */
div[data-testid="column"].nb-active .stButton>button{{
  background:{"rgba(255,255,255,.16)" if D else "rgba(0,0,0,.11)"}!important;
  border:1px solid {"rgba(255,255,255,.32)" if D else "rgba(0,0,0,.24)"}!important;
  color:{"#ffffff" if D else "#000000"}!important;
  font-weight:700!important;
  box-shadow:{"0 0 0 1px rgba(255,255,255,.1)" if D else "0 0 0 1px rgba(0,0,0,.08)"}!important;
}}
div[data-testid="column"].nb-active .stButton>button:hover{{
  transform:none!important;box-shadow:none!important;
  color:{"#ffffff" if D else "#000000"}!important;
}}
div[data-testid="column"].nb .stButton>button::after,
div[data-testid="column"].nb-active .stButton>button::after,
div[data-testid="column"].tb .stButton>button::after{{display:none!important;}}

/* Theme toggle button */
div[data-testid="column"].tb .stButton>button{{
  background:transparent!important;
  border:1px solid {BORDER}!important;
  border-radius:999px!important;color:{TEXT}!important;
  box-shadow:none!important;padding:.34rem .9rem!important;
  font-size:.72rem!important;
}}
div[data-testid="column"].tb .stButton>button:hover{{
  border-color:{A1}!important;
  box-shadow:0 0 10px rgba(109,40,217,.3)!important;
  color:{TEXT}!important;transform:none!important;
}}

/* ════════════════════════════════════════════
   PRIMARY ACTION BUTTON
════════════════════════════════════════════ */
div[data-testid="column"].predict-btn .stButton>button,
.predict-row .stButton>button{{
  width:100%!important;padding:1rem 2.5rem!important;
  background:#000000!important;
  animation:none!important;
  border:none!important;border-radius:12px!important;color:#ffffff!important;
  font-family:'Inter',sans-serif!important;font-size:.88rem!important;
  font-weight:700!important;letter-spacing:.08em!important;text-transform:uppercase!important;
  box-shadow:0 2px 12px rgba(0,0,0,.4)!important;
  margin-top:.9rem!important;position:relative!important;overflow:hidden!important;
  transition:transform .2s ease,box-shadow .2s ease,background .2s ease!important;
}}
div[data-testid="column"].predict-btn .stButton>button::after,
.predict-row .stButton>button::after{{display:none!important;}}
div[data-testid="column"].predict-btn .stButton>button:hover,
.predict-row .stButton>button:hover{{
  background:#111111!important;
  transform:translateY(-2px) scale(1.015)!important;
  box-shadow:0 0 0 1px {A1}66,0 6px 28px rgba(109,40,217,.45),0 2px 12px rgba(0,0,0,.4)!important;
}}

/* ════════════════════════════════════════════
   RESULT CARDS
════════════════════════════════════════════ */
.result-card{{
  border-radius:18px;padding:2.2rem 2.2rem;text-align:center;
  animation:fadeUp .5s cubic-bezier(.22,1,.36,1) both;
}}
.result-card.risk-high{{
  background:{"#160808" if D else "#fef2f2"};
  border:1px solid {"#7f1d1d" if D else "#fecaca"};
  box-shadow:0 0 30px rgba(220,38,38,{"0.2" if D else "0.1"});
}}
.result-card.risk-medium{{
  background:{"#160e04" if D else "#fffbeb"};
  border:1px solid {"#78350f" if D else "#fde68a"};
  box-shadow:0 0 30px rgba(217,119,6,{"0.2" if D else "0.1"});
}}
.result-card.risk-low{{
  background:{"#061410" if D else "#f0fdf4"};
  border:1px solid {"#14532d" if D else "#bbf7d0"};
  box-shadow:0 0 30px rgba(22,163,74,{"0.18" if D else "0.08"});
}}
.result-risk-label{{
  font-family:'Poppins',sans-serif;
  font-size:1.6rem;font-weight:800;
  letter-spacing:-.02em;margin-bottom:.45rem;
}}
.result-risk-label.high{{color:{"#f87171" if D else "#dc2626"};}}
.result-risk-label.medium{{color:{"#fbbf24" if D else "#d97706"};}}
.result-risk-label.low{{color:{"#4ade80" if D else "#16a34a"};}}
.result-desc{{font-size:.85rem;color:{SUB};line-height:1.65;max-width:400px;margin:.35rem auto 0;}}

/* Probability strip */
.prob-strip{{
  background:{SURF};border:1px solid {BORDER};border-radius:14px;
  padding:1.2rem 1.8rem;display:flex;align-items:center;
  justify-content:space-between;margin-top:.9rem;
  animation:fadeUp .6s .08s both;box-shadow:{CARD_SHD};
}}
.prob-strip-label{{font-size:.6rem;font-weight:700;letter-spacing:.17em;text-transform:uppercase;color:{SUB};}}
.prob-strip-meta{{font-size:.78rem;color:{SUBTLE};margin-top:.15rem;}}
.prob-strip-value{{
  font-family:'Poppins',sans-serif;font-size:2.6rem;font-weight:800;
  background:{GRAD};background-size:200% 200%;animation:gradmv 4s ease infinite;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  letter-spacing:-.04em;
}}

/* ── progress bar ── */
[data-testid="stProgress"]>div>div{{
  background:{SURF2}!important;border-radius:99px!important;height:8px!important;
}}
[data-testid="stProgress"]>div>div>div{{
  background:linear-gradient(90deg,{A1},{A2})!important;
  border-radius:99px!important;
  box-shadow:0 0 8px rgba(109,40,217,.4)!important;
}}

/* ── metric cards ── */
[data-testid="metric-container"]{{
  background:{SURF}!important;border:1px solid {BORDER}!important;
  border-radius:12px!important;padding:.85rem 1.1rem!important;
  box-shadow:{CARD_SHD}!important;
}}
[data-testid="metric-container"] label{{
  color:{SUB}!important;font-family:'Inter',sans-serif!important;
  font-size:.6rem!important;text-transform:uppercase!important;letter-spacing:.12em!important;
}}
[data-testid="stMetricValue"]{{
  color:{TEXT}!important;font-family:'Poppins',sans-serif!important;
  font-size:1.55rem!important;font-weight:800!important;
}}

hr{{border-color:{BORDER}!important;margin:1.2rem 0!important;}}
::-webkit-scrollbar{{width:4px;}}
::-webkit-scrollbar-track{{background:{BG};}}
::-webkit-scrollbar-thumb{{background:{A1}66;border-radius:99px;}}
::-webkit-scrollbar-thumb:hover{{background:{A1};}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────
cur = st.session_state.page

SIDE_BG    = SURF
SIDE_GRAD  = GRAD

with st.sidebar:
    # Brand header
    st.markdown(f"""
    <div style='padding:.6rem 0 1.6rem; border-bottom:1px solid {BORDER}; margin-bottom:1.2rem;'>
      <div style='display:flex;align-items:center;gap:.7rem;'>
        <div style='width:34px;height:34px;border-radius:9px;
             background:{GRAD};
             display:flex;align-items:center;justify-content:center;
             box-shadow:0 0 16px rgba(109,40,217,{'.5' if D else '.25'});flex-shrink:0;'>
          <div style='width:14px;height:14px;border-radius:3px;
               border:2.5px solid rgba(255,255,255,.9);position:relative;'>
            <div style='position:absolute;top:50%;left:50%;
                 transform:translate(-50%,-50%);
                 width:4px;height:4px;border-radius:50%;background:white;'></div>
          </div>
        </div>
        <div>
          <div style='font-family:Poppins,sans-serif;font-size:.87rem;font-weight:800;
               color:{TEXT};letter-spacing:-.01em;line-height:1.15;'>Customer Churn</div>
          <div style='font-size:.58rem;font-weight:500;color:{SUB};
               letter-spacing:.07em;text-transform:uppercase;'>Intelligence System</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Status badge
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:.45rem;
         margin-bottom:1.4rem;padding:.4rem .75rem;
         background:{'rgba(74,222,128,.1)' if D else 'rgba(22,163,74,.08)'};
         border:1px solid {'rgba(74,222,128,.25)' if D else 'rgba(22,163,74,.2)'};
         border-radius:8px;'>
      <div style='width:7px;height:7px;border-radius:50%;
           background:{'#4ade80' if D else '#16a34a'};
           box-shadow:0 0 8px {'rgba(74,222,128,.8)' if D else 'rgba(22,163,74,.5)'};
           animation:blink 2.4s ease-in-out infinite;flex-shrink:0;'></div>
      <span style='font-size:.65rem;font-weight:700;
            color:{'#4ade80' if D else '#16a34a'};
            letter-spacing:.05em;text-transform:uppercase;'>Model Active</span>
    </div>
    """, unsafe_allow_html=True)

    # Section label
    st.markdown(f"""
    <div style='font-size:.58rem;font-weight:700;letter-spacing:.15em;
         text-transform:uppercase;color:{SUB};margin-bottom:.6rem;
         padding-left:.25rem;'>Navigation</div>
    """, unsafe_allow_html=True)

    # Nav buttons
    pages = [("Dashboard", "Dashboard"), ("Predict", "Predict Churn"), ("Insights", "Model Insights")]
    for key, label in pages:
        is_active = cur == key
        st.markdown(f'<div class="{"sb-active" if is_active else "sb-inactive"}">',
                    unsafe_allow_html=True)
        if st.button(label, key=f"nav_{key}"):
            st.session_state.page = key
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Divider
    st.markdown(f"<hr style='border-color:{BORDER}!important;margin:1.4rem 0 1rem!important;'>",
                unsafe_allow_html=True)

    # Theme toggle
    st.markdown('<div class="sb-theme">', unsafe_allow_html=True)
    toggle_label = "Switch to Light Mode" if D else "Switch to Dark Mode"
    if st.button(toggle_label, key="theme_toggle"):
        st.session_state.dark = not D
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div style='position:fixed;bottom:1.5rem;left:0;width:260px;
         text-align:center;font-size:.58rem;color:{SUB};line-height:1.6;'>
      Powered by XGBoost &nbsp;|&nbsp; 7,043 records
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────────
def pl(fig, title="", h=360):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Poppins",size=13,color=TEXT), x=0.015, pad=dict(l=8)),
        paper_bgcolor=PPAPER, plot_bgcolor=PPAPER,
        font=dict(family="Inter",color=TEXT,size=11),
        xaxis=dict(showgrid=True,gridcolor=GRID,color=SUB,zeroline=False,
                   linecolor=BORDER,tickfont=dict(size=10)),
        yaxis=dict(showgrid=True,gridcolor=GRID,color=SUB,zeroline=False,
                   linecolor=BORDER,tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT,size=10),
                    bordercolor=BORDER,borderwidth=1),
        margin=dict(l=8,r=8,t=46,b=10),
        hoverlabel=dict(bgcolor=SURF2,font_color=TEXT,bordercolor=BORDER,
                        font_family="Inter"),
        height=h,
    )
    return fig

NEON = [A1,"#ef4444",A2,"#f59e0b","#10b981",A3,"#6366f1"]

# ══════════════════════════════════════════════════════════════════
#  DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════
if cur == "Dashboard":

    st.markdown(f"""
    <div class="hero-section">
      <div class="hero-eyebrow">Analytics Platform</div>
      <div class="hero-heading">Customer Churn<br><span>Intelligence System</span></div>
      <div class="hero-sub">AI-powered customer retention and churn analytics platform.</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI CARDS
    st.markdown(f'<div class="sec-label">Key Performance Indicators</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4, gap="medium")

    kpi_rows = [
        (k1, A1,       "Total Customers",      f"{total_cust:,}", "Live dataset", False),
        (k2, "#ef4444","Churn Rate",            f"{churn_r:.1f}%",f"{churn_n:,} customers", True),
        (k3, "#22c55e","Retention Rate",        f"{retain_r:.1f}%",f"{total_cust-churn_n:,} retained", True),
        (k4, A2,       "Avg Monthly Charge",    f"${avg_mth:.0f}", f"Avg tenure {avg_ten:.0f} mo", False),
    ]
    for col, color, label, val, meta, use_grad in kpi_rows:
        with col:
            val_cls = "kpi-value grad" if use_grad else "kpi-value"
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-accent" style="background:radial-gradient(circle,{color}44 0%,transparent 70%)"></div>
              <div class="kpi-label">{label}</div>
              <div class="{val_cls}">{val}</div>
              <div class="kpi-meta">{meta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 1: Donut + Contract
    st.markdown(f'<div class="sec-label">Churn Distribution and Contract Analysis</div>', unsafe_allow_html=True)
    r1a, r1b = st.columns([1, 1.6], gap="medium")

    with r1a:
        cv = df["Churn"].value_counts()
        fig_d = go.Figure(go.Pie(
            labels=["Retained","Churned"],
            values=[cv.get("No",0), cv.get("Yes",0)],
            hole=0.64,
            marker=dict(colors=[A1,"#ef4444"],
                        line=dict(color=SURF, width=2)),
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>",
        ))
        fig_d.add_annotation(
            text=f"<b>{churn_r:.1f}%</b><br><span style='font-size:10px;'>Churn</span>",
            x=0.5,y=0.5,showarrow=False,
            font=dict(size=15,color=TEXT,family="Poppins"),
        )
        pl(fig_d,"Churn Distribution",330)
        fig_d.update_layout(
            showlegend=True,
            legend=dict(orientation="h",y=-0.06,x=0.5,xanchor="center"),
        )
        st.plotly_chart(fig_d, use_container_width=True)

    with r1b:
        cc = df.groupby(["Contract","Churn"]).size().reset_index(name="n")
        fig_b = px.bar(cc,x="Contract",y="n",color="Churn",
                       color_discrete_map={"Yes":"#ef4444","No":A1},
                       barmode="group",text="n")
        fig_b.update_traces(texttemplate="%{text:,}",textposition="outside",
                            marker_line_width=0,textfont_size=10)
        pl(fig_b,"Contract Type vs Churn",330)
        st.plotly_chart(fig_b, use_container_width=True)

    # ROW 2: Tenure + Payment
    st.markdown(f'<div class="sec-label">Tenure Trend and Payment Method Analysis</div>', unsafe_allow_html=True)
    r2a,r2b = st.columns(2, gap="medium")

    with r2a:
        df["TenureBin"] = pd.cut(df["tenure"],bins=range(0,73,6),right=False)
        tb2 = df.groupby("TenureBin",observed=True)["ChurnBinary"].mean().reset_index()
        tb2["Bin"]      = tb2["TenureBin"].astype(str)
        tb2["ChurnPct"] = (tb2["ChurnBinary"]*100).round(1)
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(
            x=tb2["Bin"],y=tb2["ChurnPct"],mode="lines+markers",
            line=dict(color=A2,width=2.5),
            marker=dict(size=7,color=A1,line=dict(color=A2,width=1.5)),
            fill="tozeroy",
            fillcolor=f"rgba({'8,145,178' if D else '2,132,199'},{'.12' if D else '.07'})",
            hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>",
        ))
        pl(fig_l,"Tenure (months) vs Churn Rate %",330)
        fig_l.update_xaxes(tickangle=-38)
        st.plotly_chart(fig_l, use_container_width=True)

    with r2b:
        pc = df.groupby(["PaymentMethod","Churn"]).size().reset_index(name="n")
        fig_p = px.bar(pc,x="PaymentMethod",y="n",color="Churn",
                       color_discrete_map={"Yes":"#ef4444","No":A1},
                       barmode="stack")
        fig_p.update_traces(marker_line_width=0)
        pl(fig_p,"Payment Method vs Churn",330)
        fig_p.update_xaxes(tickangle=-28)
        st.plotly_chart(fig_p, use_container_width=True)

    # ROW 3: Heatmap
    st.markdown(f'<div class="sec-label">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    corr = df[["tenure","MonthlyCharges","TotalCharges","SeniorCitizen","ChurnBinary"]].corr()
    fig_h = go.Figure(go.Heatmap(
        z=corr.values,x=corr.columns.tolist(),y=corr.columns.tolist(),
        colorscale=[[0,"#ef4444"],[0.5,SURF2],[1,A1]],
        zmin=-1,zmax=1,
        text=np.round(corr.values,2),texttemplate="%{text}",
        textfont=dict(size=11,family="Inter"),
        hovertemplate="<b>%{x} vs %{y}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    pl(fig_h,"Feature Correlation Heatmap",320)
    st.plotly_chart(fig_h, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  PREDICT CHURN PAGE
# ══════════════════════════════════════════════════════════════════
elif cur == "Predict":

    st.markdown(f"""
    <div class="hero-section">
      <div class="hero-eyebrow">Real-Time Inference</div>
      <div class="hero-heading">Customer Churn<br><span>Intelligence System</span></div>
      <div class="hero-sub">Complete the customer profile to generate an instant churn risk score.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sec-label">Customer Profile Input</div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3, gap="medium")
    with c1:
        gender     = st.selectbox("Gender",          ["Male","Female"])
        senior     = st.selectbox("Senior Citizen",  ["No","Yes"])
        partner    = st.selectbox("Partner",         ["No","Yes"])
        dependents = st.selectbox("Dependents",      ["No","Yes"])
        tenure     = st.slider("Tenure (Months)",    0,72,12)
    with c2:
        phone    = st.selectbox("Phone Service",     ["No","Yes"])
        multiple = st.selectbox("Multiple Lines",    ["No","Yes"])
        internet = st.selectbox("Internet Service",  ["DSL","Fiber optic","No"])
        online_s = st.selectbox("Online Security",   ["No","Yes"])
        online_b = st.selectbox("Online Backup",     ["No","Yes"])
    with c3:
        device   = st.selectbox("Device Protection", ["No","Yes"])
        tech     = st.selectbox("Tech Support",      ["No","Yes"])
        tv       = st.selectbox("Streaming TV",      ["No","Yes"])
        movies   = st.selectbox("Streaming Movies",  ["No","Yes"])
        contract = st.selectbox("Contract",          ["Month-to-month","One year","Two year"])

    c4,c5 = st.columns(2, gap="medium")
    with c4:
        paperless = st.selectbox("Paperless Billing", ["No","Yes"])
        payment   = st.selectbox("Payment Method",[
            "Electronic check","Mailed check",
            "Bank transfer (automatic)","Credit card (automatic)"])
    with c5:
        monthly = st.slider("Monthly Charges ($)", 18.0,120.0,70.0,0.5)
        total_c = st.slider("Total Charges ($)",   18.0,9000.0,1500.0,50.0)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="predict-row">', unsafe_allow_html=True)
    run = st.button("Analyse Churn Risk")
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        enc = {
            "gender":           1 if gender=="Male"    else 0,
            "SeniorCitizen":    1 if senior=="Yes"     else 0,
            "Partner":          1 if partner=="Yes"    else 0,
            "Dependents":       1 if dependents=="Yes" else 0,
            "tenure":           tenure,
            "PhoneService":     1 if phone=="Yes"      else 0,
            "MultipleLines":    1 if multiple=="Yes"   else 0,
            "InternetService":  {"DSL":0,"Fiber optic":1,"No":2}[internet],
            "OnlineSecurity":   1 if online_s=="Yes"   else 0,
            "OnlineBackup":     1 if online_b=="Yes"   else 0,
            "DeviceProtection": 1 if device=="Yes"     else 0,
            "TechSupport":      1 if tech=="Yes"       else 0,
            "StreamingTV":      1 if tv=="Yes"         else 0,
            "StreamingMovies":  1 if movies=="Yes"     else 0,
            "Contract":         {"Month-to-month":0,"One year":1,"Two year":2}[contract],
            "PaperlessBilling": 1 if paperless=="Yes"  else 0,
            "PaymentMethod":    {
                "Electronic check":0,"Mailed check":1,
                "Bank transfer (automatic)":2,"Credit card (automatic)":3}[payment],
            "MonthlyCharges":   monthly,
            "TotalCharges":     total_c,
        }
        prob = float(model.predict_proba(pd.DataFrame([enc]))[0][1])

        # 3-level classification
        if prob >= threshold:
            risk_key  = "high"
            risk_txt  = "High Risk — Customer Likely to Churn"
            risk_desc = "This customer exhibits strong churn indicators. Immediate intervention is recommended: targeted retention offer, proactive outreach, or loyalty incentive."
        elif prob >= threshold * 0.65:
            risk_key  = "medium"
            risk_txt  = "Medium Risk — Monitor Closely"
            risk_desc = "This customer shows moderate churn signals. Consider scheduling a proactive check-in and reviewing their service satisfaction."
        else:
            risk_key  = "low"
            risk_txt  = "Low Risk — Customer Likely to Stay"
            risk_desc = "This customer shows strong retention indicators. Continue delivering consistent service quality to maintain loyalty."

        st.markdown(f"<hr style='border-color:{BORDER}!important;margin:1.5rem 0!important;'>", unsafe_allow_html=True)
        st.markdown(f'<div class="sec-label">Prediction Result</div>', unsafe_allow_html=True)

        res_col, gauge_col = st.columns([1.4, 1], gap="medium")

        with res_col:
            st.markdown(f"""
            <div class="result-card risk-{risk_key}">
              <div class="result-risk-label {risk_key}">{risk_txt}</div>
              <div class="result-desc">{risk_desc}</div>
            </div>
            <div class="prob-strip">
              <div>
                <div class="prob-strip-label">Churn Probability Score</div>
                <div class="prob-strip-meta">Model threshold: {threshold:.2f} &nbsp;|&nbsp; XGBoost, recall-optimised</div>
              </div>
              <div class="prob-strip-value">{prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(prob, 1.0))

        with gauge_col:
            gauge_color = (
                "#ef4444" if risk_key=="high"
                else "#f59e0b" if risk_key=="medium"
                else "#22c55e"
            )
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",value=prob*100,
                number={"suffix":"%","font":{"size":28,"family":"Poppins","color":TEXT}},
                gauge=dict(
                    axis=dict(range=[0,100],tickcolor=SUB,tickfont=dict(color=SUB,size=10)),
                    bar=dict(color=gauge_color,thickness=0.22),
                    bgcolor=SURF2,borderwidth=0,
                    steps=[
                        dict(range=[0, threshold*0.65*100],
                             color=f"rgba(34,197,94,{'.14' if D else '.08'})"),
                        dict(range=[threshold*0.65*100, threshold*100],
                             color=f"rgba(245,158,11,{'.14' if D else '.08'})"),
                        dict(range=[threshold*100, 100],
                             color=f"rgba(239,68,68,{'.14' if D else '.08'})"),
                    ],
                    threshold=dict(line=dict(color=A2,width=2),
                                   thickness=0.75,value=threshold*100),
                ),
            ))
            fig_g.update_layout(
                paper_bgcolor=PPAPER,
                font=dict(color=TEXT,family="Inter"),
                margin=dict(l=16,r=16,t=24,b=16),height=240,
            )
            st.plotly_chart(fig_g, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  MODEL INSIGHTS PAGE
# ══════════════════════════════════════════════════════════════════
elif cur == "Insights":

    st.markdown(f"""
    <div class="hero-section">
      <div class="hero-eyebrow">Model Intelligence</div>
      <div class="hero-heading">Customer Churn<br><span>Intelligence System</span></div>
      <div class="hero-sub">Feature importance, performance metrics, and key churn drivers from your trained model.</div>
    </div>
    """, unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4, gap="medium")
    with m1: st.metric("Algorithm","XGBoost")
    with m2: st.metric("Accuracy","~80 %")
    with m3: st.metric("Recall (Churn)","~81 %")
    with m4: st.metric("Threshold",f"{threshold:.2f}")

    # Feature importance
    st.markdown(f'<div class="sec-label">Feature Importance</div>', unsafe_allow_html=True)
    feature_names = [
        "gender","SeniorCitizen","Partner","Dependents","tenure",
        "PhoneService","MultipleLines","InternetService",
        "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
        "StreamingTV","StreamingMovies","Contract","PaperlessBilling",
        "PaymentMethod","MonthlyCharges","TotalCharges",
    ]
    try:
        fi = pd.DataFrame({"Feature":feature_names,"Score":model.feature_importances_})
        fi.sort_values("Score",ascending=True,inplace=True)
        fi["pct"] = (fi["Score"]/fi["Score"].max()*100).round(1)

        fig_fi = go.Figure(go.Bar(
            x=fi["Score"],y=fi["Feature"],orientation="h",
            text=fi["pct"].astype(str)+"%",
            textposition="outside",
            textfont=dict(size=9,color=SUB),
            marker=dict(
                color=fi["Score"],
                colorscale=[[0,A1],[0.45,A2],[1,A3]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.5f}<extra></extra>",
        ))
        pl(fig_fi,"Feature Importance — XGBoost",520)
        fig_fi.update_layout(yaxis=dict(showgrid=False,tickfont=dict(size=11)))
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception:
        st.info("Feature importance is not available for this model object.")

    # Key drivers
    st.markdown(f'<div class="sec-label">Key Churn Driver Analysis</div>', unsafe_allow_html=True)
    da1,da2 = st.columns(2, gap="medium")

    with da1:
        ic = df.groupby("InternetService")["ChurnBinary"].mean().reset_index()
        ic["Churn %"] = (ic["ChurnBinary"]*100).round(1)
        fig_ic = px.bar(ic,x="InternetService",y="Churn %",color="Churn %",
                        color_continuous_scale=[[0,A1],[1,"#ef4444"]],
                        text="Churn %")
        fig_ic.update_traces(texttemplate="%{text:.1f}%",textposition="outside",marker_line_width=0)
        pl(fig_ic,"Churn Rate by Internet Service",300)
        fig_ic.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_ic, use_container_width=True)

    with da2:
        sc = df.groupby("SeniorCitizen")["ChurnBinary"].mean().reset_index()
        sc["Segment"] = sc["SeniorCitizen"].map({0:"Non-Senior",1:"Senior"})
        sc["Churn %"] = (sc["ChurnBinary"]*100).round(1)
        fig_sc = px.bar(sc,x="Segment",y="Churn %",color="Churn %",
                        color_continuous_scale=[[0,A2],[1,"#ef4444"]],
                        text="Churn %")
        fig_sc.update_traces(texttemplate="%{text:.1f}%",textposition="outside",marker_line_width=0)
        pl(fig_sc,"Churn Rate by Segment",300)
        fig_sc.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_sc, use_container_width=True)

    # Monthly charges distribution
    st.markdown(f'<div class="sec-label">Monthly Charges Distribution by Churn Status</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(df,x="MonthlyCharges",color="Churn",
                             color_discrete_map={"Yes":"#ef4444","No":A1},
                             nbins=42,barmode="overlay",opacity=0.78)
    fig_hist.update_traces(marker_line_width=0)
    pl(fig_hist,"Monthly Charges Distribution — Churned vs Retained",310)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Why XGBoost
    st.markdown(f'<div class="sec-label">Model Rationale</div>', unsafe_allow_html=True)
    w1,w2,w3 = st.columns(3, gap="medium")
    rationale = [
        ("High Recall Optimisation",
         f"The model is tuned with a custom threshold of {threshold:.2f} to maximise recall on the minority churn class — identifying more at-risk customers before they leave."),
        ("Gradient Boosting Architecture",
         "XGBoost builds an ensemble of decision trees sequentially, each correcting residuals from the previous — delivering strong accuracy on structured tabular data."),
        ("Business Interpretability",
         "Built-in feature importance scores map directly to business attributes, enabling targeted retention campaigns based on the most influential churn drivers."),
    ]
    for col,(title,desc) in zip([w1,w2,w3],rationale):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="padding:1.6rem;">
              <div style='font-family:Poppins,sans-serif;font-size:.9rem;font-weight:700;
                   color:{TEXT};margin-bottom:.55rem;letter-spacing:-.01em;'>{title}</div>
              <div style='font-size:.79rem;color:{SUB};line-height:1.68;'>{desc}</div>
            </div>""", unsafe_allow_html=True)