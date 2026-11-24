"""
  AI Multi-Disease Risk Assessment System  |  app.py  |  v2.0
  10 Diseases | Full Error Handling | Diet Recommendations
"""

import streamlit as st
import numpy as np
import pandas as pd
import time, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Safe imports with fallback 
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from utils.risk_assessment import (
    classify_risk, apply_cross_disease_amplifier,
    compute_health_score, simulate_risk_progression,
    get_health_tips, get_diet_recommendations, get_age_group
)
from utils.preprocessing import (
    predict_disease, get_feature_importance,
    load_metrics, models_exist, partial_models_exist
)

#  PAGE CONFIG
st.set_page_config(
    page_title="HealthAI — 10-Disease Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  DISEASE REGISTRY — single source of truth
DISEASES = {
    "diabetes"    : {"label": "Diabetes",        "icon": "🩸", "color": "#e74c3c"},
    "heart"       : {"label": "Heart Disease",   "icon": "❤️", "color": "#e91e63"},
    "parkinsons"  : {"label": "Parkinson's",     "icon": "🎤", "color": "#9b59b6"},
    "ckd"         : {"label": "Kidney Disease",  "icon": "🫘", "color": "#3498db"},
    "liver"       : {"label": "Liver Disease",   "icon": "🌿", "color": "#27ae60"},
    "stroke"      : {"label": "Stroke Risk",     "icon": "🧠", "color": "#e67e22"},
    "breast_cancer":{"label": "Breast Cancer",   "icon": "💗", "color": "#ff4081"},
    "thyroid"     : {"label": "Thyroid Disease", "icon": "⚖️", "color": "#00bcd4"},
    "pcos"        : {"label": "PCOS",            "icon": "🌸", "color": "#ab47bc"},
    "anemia"      : {"label": "Anemia",          "icon": "🩺", "color": "#f44336"},
}

#  CSS INJECTION
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding: 1.2rem 2rem; max-width: 1300px; }

    /* ── Cards ─────────────────────────────────────────────── */
    .hcard {
        background: #1e2130;
        border: 1px solid #2d3246;
        border-radius: 16px;
        padding: 1.4rem;
        margin: 0.6rem 0;
        transition: transform 0.2s;
    }
    .hcard:hover { transform: translateY(-2px); }
    .hcard-low    { border-left: 5px solid #2ecc71; }
    .hcard-medium { border-left: 5px solid #f39c12; }
    .hcard-high   { border-left: 5px solid #e74c3c; }

    /* ── Diet Card ──────────────────────────────────────────── */
    .diet-card {
        background: linear-gradient(135deg, #1a2a1a, #1e2a1e);
        border: 1px solid #2d4a2d;
        border-radius: 16px;
        padding: 1.4rem;
        margin: 0.6rem 0;
    }
    .diet-food-item {
        background: #1a2a1a;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        margin: 0.25rem 0;
        font-size: 0.88rem;
        border-left: 3px solid #2ecc71;
    }
    .diet-avoid-item {
        background: #2a1a1a;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        margin: 0.25rem 0;
        font-size: 0.88rem;
        border-left: 3px solid #e74c3c;
    }
    .meal-box {
        background: #252840;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.88rem;
    }
    .meal-label {
        font-weight: 700;
        color: #6c63ff;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .supplement-pill {
        display: inline-block;
        background: #1a2040;
        color: #7c8ff5;
        border: 1px solid #3040a0;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.78rem;
        margin: 0.15rem;
    }

    /* ── Progress bar ───────────────────────────────────────── */
    .pbar-wrap { background:#2d3246; border-radius:999px; height:10px; overflow:hidden; margin:6px 0; }
    .pbar-fill  { height:100%; border-radius:999px; }

    /* ── Privacy / info badges ──────────────────────────────── */
    .badge-privacy {
        background:#1a2540; color:#5dade2;
        border:1px solid #2c3e6a; border-radius:8px;
        padding:0.3rem 0.8rem; font-size:0.76rem;
        display:inline-flex; align-items:center; gap:5px;
    }
    .badge-disclaimer {
        background:#2c1a00; border:1px solid #7d4e00;
        border-radius:10px; padding:0.8rem 1.2rem;
        font-size:0.82rem; color:#ffcc80; margin-top:1.5rem;
    }

    /* ── Section heading ────────────────────────────────────── */
    .sec-title {
        font-size:1.1rem; font-weight:700;
        color:#e8eaf0; margin:1.2rem 0 0.6rem 0;
        display:flex; align-items:center; gap:8px;
    }

    /* ── Tip item ───────────────────────────────────────────── */
    .tip-item {
        background:#1e2130; border-radius:9px;
        padding:0.55rem 1rem; margin:0.3rem 0;
        font-size:0.88rem; border-left:3px solid #6c63ff;
    }

    /* ── Error box ──────────────────────────────────────────── */
    .err-box {
        background:#2a1a1a; border:1px solid #7d2020;
        border-radius:10px; padding:0.7rem 1rem;
        font-size:0.85rem; color:#e88080; margin:0.5rem 0;
    }

    /* ── Buttons ────────────────────────────────────────────── */
    .stButton > button {
        background:linear-gradient(135deg,#6c63ff,#9c59ff);
        color:white; border:none; border-radius:12px;
        padding:0.6rem 1.8rem; font-weight:600;
        width:100%; transition:opacity 0.2s;
    }
    .stButton > button:hover { opacity:0.88; }

    /* ── Sidebar ────────────────────────────────────────────── */
    [data-testid="stSidebar"] { background:#151826 !important; border-right:1px solid #2d3246; }

    /* ── Slider thumb ───────────────────────────────────────── */
    .stSlider > div > div > div { background:#6c63ff !important; }

    footer { visibility:hidden; }
    </style>
    """, unsafe_allow_html=True)


#  SAFE CHART RENDERERS (matplotlib fallback when plotly unavailable)

def render_gauge(prob: float, title: str, color: str):
    """Render a semi-circular gauge using matplotlib."""
    if not HAS_MPL:
        st.metric(title, f"{prob*100:.1f}%")
        return
    fig, ax = plt.subplots(figsize=(4, 2.4), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')

    # Background arc (gray)
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), lw=18, color='#2d3246', solid_capstyle='round')

    # Coloured fill arc
    filled = max(0.01, min(1.0, prob))
    theta_fill = np.linspace(np.pi, np.pi - filled * np.pi, 300)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), lw=18, color=color, solid_capstyle='round')

    # Zone markers
    for pct, col in [(0.33, '#2ecc71'), (0.66, '#f39c12')]:
        angle = np.pi - pct * np.pi
        ax.plot([0.72 * np.cos(angle), 0.88 * np.cos(angle)],
                [0.72 * np.sin(angle), 0.88 * np.sin(angle)], color='#1e2130', lw=3)

    # Labels
    ax.text(0, 0.15, f"{prob*100:.1f}%", ha='center', va='center',
            fontsize=18, fontweight='bold', color=color)
    ax.text(0, -0.15, title, ha='center', va='center',
            fontsize=9, color='#9ba3b2')
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.35, 1.2)
    ax.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_progression(progression: list, title: str, color: str):
    """Line chart for 5-year risk projection."""
    if not HAS_MPL:
        data = pd.DataFrame(progression, columns=["Year", "Risk"])
        st.line_chart(data.set_index("Year"))
        return
    years = [p[0] for p in progression]
    probs = [p[1] * 100 for p in progression]

    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')
    ax.fill_between(years, probs, alpha=0.15, color=color)
    ax.plot(years, probs, color=color, lw=2.5, marker='o', markersize=5)
    ax.axhline(33, color='#2ecc71', lw=1, ls='--', alpha=0.6)
    ax.axhline(66, color='#f39c12', lw=1, ls='--', alpha=0.6)
    ax.set_xlabel("Years", color='#9ba3b2', fontsize=8)
    ax.set_ylabel("Risk %",  color='#9ba3b2', fontsize=8)
    ax.set_title(f"{title} — 5-Year Trend", color='#e8eaf0', fontsize=9)
    ax.set_ylim(0, 100)
    ax.tick_params(colors='#9ba3b2', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3246')
    ax.grid(color='#2d3246', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_feature_chart(importances: dict, title: str):
    """Horizontal bar chart for feature importances."""
    if not importances:
        return
    if not HAS_MPL:
        st.bar_chart(pd.Series(importances))
        return
    features = list(importances.keys())[:8]
    values   = [importances[f] for f in features]
    # Reverse for top-at-top display
    features, values = features[::-1], values[::-1]

    fig, ax = plt.subplots(figsize=(5, max(2.5, len(features)*0.38)))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')
    colors_bar = [f'#{int(72+(v/max(values))*80):02x}63ff' if v==v else '#6c63ff' for v in values]
    bars = ax.barh(features, values, color='#6c63ff', alpha=0.8, height=0.6)
    # Color gradient
    for bar, val in zip(bars, values):
        intensity = val / max(values) if max(values) > 0 else 0
        bar.set_color((0.25 + 0.35*intensity, 0.2 + 0.1*intensity, 0.8 + 0.2*intensity))
    ax.set_xlabel("Importance (%)", color='#9ba3b2', fontsize=8)
    ax.set_title(f"{title} — Feature Importance", color='#e8eaf0', fontsize=9)
    ax.tick_params(colors='#9ba3b2', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3246')
    ax.grid(axis='x', color='#2d3246', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_health_donut(score: float, grade: str, color: str):
    """Donut chart for overall health score."""
    if not HAS_MPL:
        st.metric("Overall Health Score", f"{score:.0f}/100", grade)
        return
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')
    sizes = [score, 100 - score]
    colors_d = [color, '#2d3246']
    wedges, _ = ax.pie(sizes, colors=colors_d, startangle=90,
                        wedgeprops=dict(width=0.42, edgecolor='#1e2130', linewidth=2))
    ax.text(0, 0.08, f"{score:.0f}", ha='center', va='center',
            fontsize=26, fontweight='bold', color=color)
    ax.text(0, -0.22, grade, ha='center', va='center',
            fontsize=10, color='#9ba3b2')
    ax.text(0, -0.42, "Health Score", ha='center', va='center',
            fontsize=8, color='#9ba3b2')
    plt.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  RISK CARD RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_risk_card(disease_key: str, risk_res: dict, raw_prob: float, amplified: bool):
    info  = DISEASES[disease_key]
    level = risk_res["level"]
    color = risk_res["color"]
    prob  = risk_res["probability_pct"]
    conf  = risk_res["confidence"]

    amp_badge = ('<span style="font-size:0.72rem;background:#3a1a00;color:#f39c12;'
                 'border-radius:6px;padding:2px 7px;margin-left:8px;">⚡ Amplified</span>'
                 if amplified else "")

    st.markdown(f"""
    <div class="hcard hcard-{level.lower()}">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <span style="font-size:1.6rem;">{info['icon']}</span>
          <span style="font-size:1.05rem;font-weight:700;margin-left:8px;">
            {info['label']}</span>{amp_badge}
        </div>
        <span style="background:{color}22;color:{color};padding:4px 14px;
               border-radius:20px;font-weight:700;font-size:0.9rem;">{level} Risk</span>
      </div>
      <div style="margin-top:0.9rem;">
        <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#9ba3b2;">
          <span>Risk Probability</span><span><b style="color:{color}">{prob:.1f}%</b></span>
        </div>
        <div class="pbar-wrap">
          <div class="pbar-fill" style="width:{prob}%;background:{color};"></div>
        </div>
        <div style="display:flex;gap:1.2rem;margin-top:0.5rem;font-size:0.8rem;color:#9ba3b2;">
          <span>🎯 Confidence: <b style="color:#e8eaf0">{conf:.0f}%</b></span>
          <span>👤 Age Group: <b style="color:#e8eaf0">{risk_res.get('age_group','adult').title()}</b></span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DIET RECOMMENDATION RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_diet_card(disease_key: str, risk_level: str):
    """Render full diet recommendation card."""
    try:
        diet = get_diet_recommendations(disease_key, risk_level)
    except Exception:
        st.info("Diet recommendations unavailable for this disease.")
        return

    st.markdown(f"""
    <div class="diet-card">
      <div style="font-size:1.1rem;font-weight:700;color:#4caf50;margin-bottom:0.8rem;">
        🥗 {diet.get('headline','Diet Recommendations')}
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ Foods to Eat**")
        for item in diet.get("foods_eat", []):
            st.markdown(f'<div class="diet-food-item">{item}</div>', unsafe_allow_html=True)

        st.markdown("<br>**💊 Supplements** *(consult doctor first)*", unsafe_allow_html=True)
        supps = diet.get("supplements", [])
        supp_html = "".join(f'<span class="supplement-pill">{s}</span>' for s in supps)
        st.markdown(supp_html, unsafe_allow_html=True)

    with col2:
        st.markdown("**❌ Foods to Avoid**")
        for item in diet.get("foods_avoid", []):
            st.markdown(f'<div class="diet-avoid-item">{item}</div>', unsafe_allow_html=True)

        st.markdown("<br>**💧 Hydration**", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#1a2030;border-radius:9px;padding:0.6rem 0.9rem;
             font-size:0.86rem;border-left:3px solid #3498db;">
          {diet.get('hydration','Stay well hydrated.')}
        </div>
        """, unsafe_allow_html=True)

    # Meal plan
    st.markdown("**🍽️ Sample Meal Plan**")
    meal_plan = diet.get("meal_plan", {})
    mcols = st.columns(4)
    for i, (meal, desc) in enumerate(meal_plan.items()):
        with mcols[i % 4]:
            st.markdown(f"""
            <div class="meal-box">
              <div class="meal-label">{meal}</div>
              <div style="margin-top:0.3rem;font-size:0.82rem;color:#c8cedb;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Nutrition targets
    nutrition = diet.get("nutrition", [])
    if nutrition:
        st.markdown("**📊 Nutrition Targets**")
        ncols = st.columns(len(nutrition))
        for i, nt in enumerate(nutrition):
            ncols[i].markdown(f"""
            <div style="background:#252840;border-radius:8px;padding:0.5rem 0.7rem;
                 text-align:center;font-size:0.8rem;color:#a0a8c0;">{nt}</div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.72rem;color:#666;margin-top:0.8rem;font-style:italic;">
    ⚕️ These are general dietary guidelines only, not medical prescriptions.
    Always consult a registered dietitian or physician before making major dietary changes.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  INPUT PANELS — All 10 diseases
# ─────────────────────────────────────────────────────────────────────────────

def panel_basic_info() -> dict:
    st.markdown('<div class="sec-title">👤 Basic Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age    = st.slider("Age (years)", 15, 90, 35)
        gender = st.radio("Biological Sex", ["Female", "Male"], horizontal=True)
        height = st.slider("Height (cm)", 140, 220, 165)
        weight = st.slider("Weight (kg)", 30, 180, 65)
    with c2:
        smoker   = st.selectbox("Smoking Status", ["Never Smoked", "Ex-Smoker", "Current Smoker"])
        alcohol  = st.selectbox("Alcohol Consumption", ["None", "Occasional", "Moderate", "Heavy"])
        exercise = st.selectbox("Exercise Level", ["Sedentary", "Light (1-2x/wk)", "Moderate (3-4x/wk)", "Active (5+/wk)"])
        married  = st.radio("Marital Status", ["Single", "Married"], horizontal=True)

    bmi = weight / ((height / 100) ** 2)
    bmi_cat = ("Underweight" if bmi < 18.5 else "Normal" if bmi < 25
               else "Overweight" if bmi < 30 else "Obese")
    st.info(f"📊 BMI: **{bmi:.1f}** ({bmi_cat}) &nbsp;|&nbsp; "
            f"🚬 Smoking: **{smoker}** &nbsp;|&nbsp; "
            f"🏋️ Exercise: **{exercise}**")
    return {
        "age": age, "gender": 1 if gender == "Male" else 0,
        "height": height, "weight": weight, "bmi": round(bmi, 2),
        "smoker_val": {"Never Smoked": 0, "Ex-Smoker": 1, "Current Smoker": 2}[smoker],
        "married_val": 1 if married == "Married" else 0,
    }


def panel_diabetes(b: dict) -> list:
    st.markdown('<div class="sec-title">🩸 Diabetes — Blood Sugar & Metabolic</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        preg = st.slider("Pregnancies (0 for males)", 0, 17, 0) if b["gender"] == 0 else 0
        gluc = st.slider("Fasting Glucose (mg/dL) — Normal: 70–99", 50, 250, 100)
        bp   = st.slider("Diastolic Blood Pressure (mmHg) — Normal: 60–80", 40, 130, 72)
        skin = st.slider("Skin Thickness / Triceps (mm) — Normal: 10–40", 0, 70, 20)
    with c2:
        ins  = st.slider("Insulin Level (μU/mL) — Normal fasting: 2–20", 0, 500, 80)
        st.metric("BMI (from Basic Info)", f"{b['bmi']:.1f}")
        dpf  = st.slider("Diabetes Pedigree Function (family history score)", 0.08, 2.50, 0.47, step=0.01)
    return [preg, gluc, bp, skin, ins, b["bmi"], dpf, b["age"]]


def panel_heart(b: dict) -> list:
    st.markdown('<div class="sec-title">❤️ Heart Disease — Cardiovascular</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        cp_opt  = st.selectbox("Chest Pain Type",
                               ["Typical Angina (0)", "Atypical Angina (1)",
                                "Non-Anginal Pain (2)", "Asymptomatic (3)"])
        cp      = int(cp_opt[-2])
        trestbps= st.slider("Resting Blood Pressure (mmHg) — Normal: 90–120", 90, 200, 120)
        chol    = st.slider("Cholesterol (mg/dL) — Normal: <200 | High: >240", 120, 600, 200)
        fbs     = st.checkbox("Fasting Blood Sugar > 120 mg/dL")
        restecg_opt = st.selectbox("Resting ECG Result",
                                   ["Normal (0)", "ST-T Wave Abnormality (1)",
                                    "Left Ventricular Hypertrophy (2)"])
        restecg = int(restecg_opt[-2])
    with c2:
        thalach = st.slider("Max Heart Rate Achieved — Target: 220-age", 70, 210, 150)
        exang   = st.checkbox("Exercise-Induced Chest Pain (Angina)")
        oldpeak = st.slider("ST Depression (Oldpeak) — Normal: 0–1", 0.0, 6.0, 1.0, step=0.1)
        slope_opt = st.selectbox("ST Segment Slope",
                                 ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
        slope   = int(slope_opt[-2])
        ca      = st.slider("Major Vessels Coloured by Fluoroscopy (0–4)", 0, 4, 0)
        thal_opt= st.selectbox("Thalassemia",
                               ["Unknown (0)", "Normal (1)",
                                "Fixed Defect (2)", "Reversible Defect (3)"])
        thal    = int(thal_opt[-2])
    return [b["age"], b["gender"], cp, trestbps, chol, int(fbs), restecg,
            thalach, int(exang), oldpeak, slope, ca, thal]


def panel_parkinsons() -> list:
    st.markdown('<div class="sec-title">🎤 Parkinson\'s — Voice Biomarkers</div>', unsafe_allow_html=True)
    with st.expander("📖 What are these voice features?", expanded=False):
        st.markdown("""
        These features are extracted from sustained vowel recordings (saying "ahhh").
        | Feature | Meaning | Healthy Range |
        |---------|---------|---------------|
        | **MDVP:Fo(Hz)** | Average vocal pitch frequency | 85–255 Hz |
        | **Jitter (%)** | Cycle-to-cycle frequency variation | < 1% |
        | **Shimmer** | Amplitude variation | < 3% |
        | **HNR** | Harmonics-to-Noise Ratio | > 20 dB |
        | **NHR** | Noise-to-Harmonics Ratio | < 0.05 |
        | **RPDE / DFA / PPE** | Nonlinear voice dynamics | Varies |
        """)
    c1, c2 = st.columns(2)
    with c1:
        fo  = st.slider("Avg Vocal Frequency — MDVP:Fo(Hz)", 80.0, 260.0, 154.0, step=0.5)
        fhi = st.slider("Max Vocal Frequency — MDVP:Fhi(Hz)", 100.0, 600.0, 197.0, step=0.5)
        flo = st.slider("Min Vocal Frequency — MDVP:Flo(Hz)", 65.0, 240.0, 116.0, step=0.5)
        jp  = st.slider("Jitter % — Freq variation", 0.001, 0.080, 0.006, step=0.001, format="%.3f")
        ja  = st.slider("Jitter (Abs)", 0.000001, 0.0003, 0.000038, step=0.000001, format="%.6f")
        rap = st.slider("MDVP:RAP", 0.0, 0.05, 0.003, step=0.001, format="%.3f")
        ppq = st.slider("MDVP:PPQ", 0.0, 0.05, 0.003, step=0.001, format="%.3f")
        ddp = st.slider("Jitter:DDP", 0.0, 0.16, 0.01, step=0.001, format="%.3f")
        sh  = st.slider("Shimmer — Amplitude variation", 0.009, 0.200, 0.030, step=0.001, format="%.3f")
        shd = st.slider("Shimmer(dB)", 0.085, 2.0, 0.28, step=0.01)
        a3  = st.slider("Shimmer:APQ3", 0.0, 0.1, 0.016, step=0.001, format="%.3f")
    with c2:
        a5  = st.slider("Shimmer:APQ5", 0.0, 0.14, 0.018, step=0.001, format="%.3f")
        apq = st.slider("MDVP:APQ", 0.0, 0.14, 0.024, step=0.001, format="%.3f")
        dda = st.slider("Shimmer:DDA", 0.0, 0.3, 0.047, step=0.001, format="%.3f")
        nhr = st.slider("NHR — Noise ratio", 0.0006, 0.30, 0.025, step=0.001, format="%.3f")
        hnr = st.slider("HNR — Harmonics ratio (dB)", 8.0, 34.0, 21.9, step=0.1)
        rpde= st.slider("RPDE", 0.26, 0.69, 0.50, step=0.01)
        dfa = st.slider("DFA", 0.57, 0.83, 0.72, step=0.01)
        sp1 = st.slider("spread1", -8.0, -2.0, -5.68, step=0.01)
        sp2 = st.slider("spread2", 0.05, 0.45, 0.226, step=0.001, format="%.3f")
        d2  = st.slider("D2", 1.4, 3.7, 2.38, step=0.01)
        ppe = st.slider("PPE — Pitch Period Entropy", 0.04, 0.53, 0.207, step=0.001, format="%.3f")
    return [fo,fhi,flo,jp,ja,rap,ppq,ddp,sh,shd,a3,a5,apq,dda,nhr,hnr,rpde,dfa,sp1,sp2,d2,ppe]


def panel_ckd(b: dict) -> list:
    st.markdown('<div class="sec-title">🫘 Kidney Disease — Renal Biomarkers</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        bp   = st.slider("Blood Pressure (mmHg) — Normal: <120/80", 50, 180, 80)
        sg   = st.selectbox("Urine Specific Gravity — Normal: 1.010–1.025",
                            [1.005, 1.010, 1.015, 1.020, 1.025])
        al   = st.slider("Albumin in Urine (0=absent, 4=heavy)", 0, 4, 0)
        su   = st.slider("Sugar in Urine (0=absent, 4=heavy)", 0, 4, 0)
        rbc_opt = st.selectbox("Red Blood Cells in Urine", ["Normal (0)", "Abnormal (1)"])
        rbc  = int(rbc_opt[-2])
        bu   = st.slider("Blood Urea (mg/dL) — Normal: 7–20", 1, 400, 40)
        sc   = st.slider("Serum Creatinine (mg/dL) — Normal: 0.6–1.2", 0.4, 76.0, 1.2, step=0.1)
        sod  = st.slider("Sodium (mEq/L) — Normal: 135–145", 111, 163, 137)
    with c2:
        pot  = st.slider("Potassium (mEq/L) — Normal: 3.5–5.0", 2.5, 47.0, 4.5, step=0.1)
        hemo = st.slider("Haemoglobin (g/dL) — Normal M:13.5–17.5, F:12–15.5", 3.0, 18.0, 13.5, step=0.1)
        pcv  = st.slider("Packed Cell Volume (%) — Normal: 37–52", 9, 54, 40)
        wbc  = st.slider("WBC Count (cells/mm³) — Normal: 4,500–11,000", 3800, 26400, 8000, step=100)
        rbc2 = st.slider("RBC Count (M/μL) — Normal: 4.2–5.9", 2.1, 8.0, 4.5, step=0.1)
        htn  = st.checkbox("Hypertension diagnosed")
        dm   = st.checkbox("Diabetes Mellitus diagnosed")
    return [b["age"], bp, float(sg), al, su, rbc, bu, sc, sod, pot, hemo, pcv, wbc, rbc2, int(htn), int(dm)]


def panel_liver(b: dict) -> list:
    st.markdown('<div class="sec-title">🌿 Liver Disease — Hepatic Biomarkers</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        tb   = st.slider("Total Bilirubin (mg/dL) — Normal: 0.2–1.2", 0.4, 75.0, 0.9, step=0.1)
        db   = st.slider("Direct Bilirubin (mg/dL) — Normal: 0–0.3", 0.1, 20.0, 0.3, step=0.1)
        alk  = st.slider("Alkaline Phosphotase / ALP (IU/L) — Normal: 44–147", 60, 2200, 200)
        sgt  = st.slider("SGPT / ALT (IU/L) — Normal: 7–56", 10, 2000, 35)
    with c2:
        sgo  = st.slider("SGOT / AST (IU/L) — Normal: 10–40", 10, 5000, 35)
        tp   = st.slider("Total Proteins (g/dL) — Normal: 6.0–8.3", 2.7, 9.6, 6.8, step=0.1)
        alb  = st.slider("Albumin (g/dL) — Normal: 3.5–5.0", 0.9, 5.5, 3.3, step=0.1)
        ag   = st.slider("Albumin/Globulin Ratio — Normal: 1.1–2.5", 0.3, 2.8, 1.0, step=0.1)
    return [b["age"], b["gender"], tb, db, alk, sgt, sgo, tp, alb, ag]


def panel_stroke(b: dict) -> list:
    st.markdown('<div class="sec-title">🧠 Stroke Risk — Neurological & Vascular</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        htn  = st.checkbox("Hypertension diagnosed")
        hrtd = st.checkbox("Heart Disease diagnosed")
        gluc = st.slider("Avg Glucose Level (mg/dL) — Normal: 70–100", 55, 300, 100)
        work_opt = st.selectbox("Work Type",
                               ["Private sector (0)", "Self-employed (1)",
                                "Government job (2)", "Children/Student (3)", "Never worked (4)"])
        work = int(work_opt[-2])
    with c2:
        resid_opt = st.selectbox("Residence Type", ["Rural (0)", "Urban (1)"])
        resid = int(resid_opt[-2])
        smok_opt  = st.selectbox("Smoking Status",
                                 ["Never smoked (0)", "Formerly smoked (1)", "Smokes (2)"])
        smok = int(smok_opt[-2])
    return [b["gender"], b["age"], int(htn), int(hrtd), b["married_val"],
            work, resid, gluc, b["bmi"], smok]


def panel_breast_cancer() -> list:
    st.markdown('<div class="sec-title">💗 Breast Cancer — Cell Nucleus Measurements</div>',
                unsafe_allow_html=True)
    st.caption("These measurements come from Fine Needle Aspiration (FNA) biopsy imaging.")
    with st.expander("📖 What do these measurements mean?", expanded=False):
        st.markdown("""
        These values describe properties of cell nuclei from breast tissue samples.
        Larger, more irregular nuclei typically indicate malignancy.
        | Feature | Meaning | Benign Range | Malignant Range |
        |---------|---------|--------------|-----------------|
        | radius_mean | Average distance center to perimeter | ~12 | ~17+ |
        | texture_mean | Std deviation of gray-scale values | ~18 | ~21+ |
        | smoothness_mean | Local variation in radius lengths | ~0.09 | ~0.10+ |
        | compactness | (perimeter²/area - 1.0) | ~0.08 | ~0.14+ |
        | concavity | Severity of concave portions | ~0.046 | ~0.16+ |
        """)
    c1, c2 = st.columns(2)
    with c1:
        radius   = st.slider("Radius Mean (μm) — Benign: ~12, Malignant: ~17+", 7.0, 28.0, 14.0, step=0.1)
        texture  = st.slider("Texture Mean — Benign: ~18, Malignant: ~21+", 9.0, 40.0, 19.0, step=0.1)
        perim    = st.slider("Perimeter Mean (μm) — Benign: ~78, Malignant: ~115+", 44.0, 190.0, 92.0, step=0.5)
        area     = st.slider("Area Mean (μm²) — Benign: ~462, Malignant: ~978+", 140.0, 2500.0, 655.0, step=5.0)
        smooth   = st.slider("Smoothness Mean — Normal: 0.09", 0.05, 0.16, 0.096, step=0.001, format="%.3f")
    with c2:
        compact  = st.slider("Compactness Mean — Benign: ~0.08", 0.01, 0.35, 0.10, step=0.001, format="%.3f")
        concav   = st.slider("Concavity Mean — Benign: ~0.05, Malignant: ~0.16+", 0.0, 0.43, 0.09, step=0.001, format="%.3f")
        concpts  = st.slider("Concave Points Mean — Benign: ~0.025", 0.0, 0.20, 0.049, step=0.001, format="%.3f")
        symm     = st.slider("Symmetry Mean — Normal: 0.17–0.19", 0.10, 0.30, 0.18, step=0.001, format="%.3f")
        frac     = st.slider("Fractal Dimension Mean — Normal: 0.06", 0.04, 0.097, 0.062, step=0.001, format="%.3f")
    return [radius, texture, perim, area, smooth, compact, concav, concpts, symm, frac]


def panel_thyroid(b: dict) -> list:
    st.markdown('<div class="sec-title">⚖️ Thyroid Disease — Hormonal Panel</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        tsh = st.slider("TSH — Thyroid Stimulating Hormone (mIU/L) — Normal: 0.4–4.0", 0.005, 100.0, 2.0, step=0.005, format="%.3f",
                        help="HIGH TSH → hypothyroid | LOW TSH → hyperthyroid")
        t3  = st.slider("T3 — Triiodothyronine (nmol/L) — Normal: 1.2–2.8", 0.2, 5.5, 1.8, step=0.1)
        tt4 = st.slider("TT4 — Total Thyroxine (μg/dL) — Normal: 5.0–12.0", 15.0, 300.0, 95.0, step=0.5)
    with c2:
        t4u = st.slider("T4U — T4 Uptake — Normal: 0.7–1.3", 0.5, 2.2, 0.95, step=0.01)
        fti = st.slider("FTI — Free Thyroxine Index — Normal: 60–120", 8.0, 600.0, 100.0, step=0.5)
        on_meds = st.checkbox("Currently on Thyroxine medication")
    return [b["age"], b["gender"], tsh, t3, tt4, t4u, fti, int(on_meds)]


def panel_pcos(b: dict) -> list:
    st.markdown('<div class="sec-title">🌸 PCOS — Hormonal & Reproductive</div>',
                unsafe_allow_html=True)
    st.caption("PCOS affects ~1 in 10 women of reproductive age. All inputs are female-specific.")
    c1, c2 = st.columns(2)
    with c1:
        cycle_opt = st.radio("Menstrual Cycle Regularity",
                             ["Regular (every 21–35 days)", "Irregular (>35 days or absent)"],
                             horizontal=False)
        cycle = 1 if "Irregular" in cycle_opt else 0
        amh   = st.slider("AMH — Anti-Müllerian Hormone (ng/mL) — Normal: 1–3.5 | PCOS: >3.5", 0.2, 20.0, 2.5, step=0.1)
        fsh   = st.slider("FSH (mIU/mL) — Normal: 3–10", 0.5, 15.0, 5.5, step=0.1)
        lh    = st.slider("LH (mIU/mL) — Normal: 2–7 | PCOS: often >10", 0.2, 25.0, 6.0, step=0.1)
        waist = st.slider("Waist Circumference (cm) — Risk: >80cm (women)", 55.0, 130.0, 80.0, step=0.5)
        bglu  = st.slider("Blood Glucose (mg/dL) — Normal fasting: 70–99", 60.0, 200.0, 90.0, step=0.5)
    with c2:
        hir   = st.checkbox("Hirsutism (excess facial/body hair)")
        skin  = st.checkbox("Skin Darkening (acanthosis nigricans)")
        hair_l= st.checkbox("Hair Thinning / Hair Loss")
        foll  = st.slider("Follicle Count (both ovaries combined) — PCOS: >20", 5, 40, 10)
        st.metric("BMI (from Basic Info)", f"{b['bmi']:.1f}")
        weight= b["weight"]
    fsh_lh = lh / (fsh + 0.01)
    if fsh_lh > 2:
        st.warning(f"⚠️ LH/FSH Ratio: {fsh_lh:.1f} (>2 is a PCOS indicator)")
    return [b["age"], b["bmi"], cycle, amh, fsh, lh, fsh_lh, waist, weight,
            int(hir), int(skin), int(hair_l), foll, bglu]


def panel_anemia(b: dict) -> list:
    st.markdown('<div class="sec-title">🩺 Anemia — Blood Cell Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        hemog = st.slider("Hemoglobin (g/dL) — Normal M:13.5–17.5 | F:12–15.5",
                          4.0, 18.0, 13.5 if b["gender"] == 1 else 12.5, step=0.1)
        mch   = st.slider("MCH — Mean Corpuscular Hemoglobin (pg) — Normal: 27–33", 14.0, 40.0, 27.5, step=0.1)
        mchc  = st.slider("MCHC — MCH Concentration (g/dL) — Normal: 32–36", 25.0, 40.0, 33.5, step=0.1)
        mcv   = st.slider("MCV — Mean Corpuscular Volume (fL) — Normal: 80–100", 55.0, 115.0, 85.0, step=0.5)
    with c2:
        rbc_c = st.slider("RBC Count (M/μL) — Normal M:4.7–6.1 | F:4.2–5.4",
                          2.0, 7.0, 4.7 if b["gender"] == 1 else 4.2, step=0.1)
        wbc_c = st.slider("WBC Count (cells/μL) — Normal: 4,500–11,000", 2000.0, 15000.0, 7000.0, step=100.0)
        plat  = st.slider("Platelets (×10³/μL) — Normal: 150–400", 50.0, 650.0, 260.0, step=5.0)
    return [b["gender"], hemog, mch, mchc, mcv, rbc_c, wbc_c, plat]


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## 🏥 HealthAI")
        st.markdown("*10-Disease Risk Assessment*")
        st.markdown("---")
        st.markdown("### 🔬 Select Diseases")

        selections = {}
        for key, info in DISEASES.items():
            selections[key] = st.checkbox(
                f"{info['icon']} {info['label']}", value=(key in ["diabetes", "heart", "stroke"])
            )

        st.markdown("---")
        selected = [k for k, v in selections.items() if v]
        st.markdown(f"**{len(selected)}** disease(s) selected")

        st.markdown("---")
        st.markdown("### ⚙️ Display Settings")
        show_xai   = st.toggle("🧠 Feature Importance (XAI)", value=True)
        show_trend = st.toggle("📈 5-Year Risk Trend",         value=True)
        show_diet  = st.toggle("🥗 Diet Recommendations",      value=True)

        st.markdown("---")
        st.markdown('<div class="badge-privacy">🔒 No data stored</div>',
                    unsafe_allow_html=True)
        st.caption("v2.0 | AI Health System")

    return {"selected": selected, "show_xai": show_xai,
            "show_trend": show_trend, "show_diet": show_diet}


# ─────────────────────────────────────────────────────────────────────────────
#  FULL RESULTS RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_results(results: dict, basic: dict, settings: dict):
    """Render all prediction results — with full error isolation per disease."""
    st.markdown("---")
    st.markdown("## 📊 Your Health Risk Assessment Results")
    st.markdown('<div class="badge-privacy">🔒 No data stored &nbsp;|&nbsp; ⚕️ Not medical advice</div>',
                unsafe_allow_html=True)
    st.markdown("")

    # ── Overall Health Score ──────────────────────────────────────────────────
    try:
        probs_dict = {k: v["raw_prob"] for k, v in results.items() if "raw_prob" in v}
        health = compute_health_score(probs_dict)
        c_score, c_summary = st.columns([1, 2])
        with c_score:
            render_health_donut(health["score"], health["grade"], health["color"])
        with c_summary:
            st.markdown(f"""
            <div class="hcard" style="margin-top:1rem;">
              <div style="font-size:1.15rem;font-weight:700;">🏆 Overall Health Summary</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.8rem;">
                <div><span style="color:#9ba3b2;font-size:0.8rem;">Health Score</span>
                     <div style="font-size:2rem;font-weight:800;color:{health['color']}">{health['score']:.0f}/100</div></div>
                <div><span style="color:#9ba3b2;font-size:0.8rem;">Grade</span>
                     <div style="font-size:2rem;font-weight:800;color:{health['color']}">{health['grade']}</div></div>
                <div><span style="color:#9ba3b2;font-size:0.8rem;">Age</span>
                     <div style="font-size:1.3rem;font-weight:600;">{basic['age']} yrs</div></div>
                <div><span style="color:#9ba3b2;font-size:0.8rem;">BMI</span>
                     <div style="font-size:1.3rem;font-weight:600;">{basic['bmi']:.1f}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.warning("Overall health score could not be computed.")

    st.markdown("---")

    # ── Per-disease results ───────────────────────────────────────────────────
    for disease_key, res in results.items():
        info = DISEASES.get(disease_key, {"label": disease_key, "icon": "🔬"})

        # If prediction itself failed, show friendly error box
        if not res.get("success", True):
            st.markdown(f"""
            <div class="err-box">
              {info['icon']} <b>{info['label']}</b> — Could not compute risk.
              Please ensure models are trained and try again.
            </div>
            """, unsafe_allow_html=True)
            continue

        risk_res = res.get("risk", {})
        raw_prob = res.get("raw_prob", 0.5)
        amplified= res.get("amplified", False)
        level    = risk_res.get("level", "Medium")

        try:
            with st.expander(
                f"{info['icon']} {info['label']}  —  "
                f"{risk_res.get('label','Medium Risk')}  "
                f"({raw_prob*100:.1f}%)",
                expanded=True
            ):
                tabs = ["📊 Risk Card", "📈 Visuals", "🧠 XAI", "💡 Tips"]
                if settings["show_diet"]:
                    tabs.append("🥗 Diet Plan")

                tab_objs = st.tabs(tabs)

                # TAB 1 — Risk Card
                with tab_objs[0]:
                    render_risk_card(disease_key, risk_res, raw_prob, amplified)

                # TAB 2 — Charts
                with tab_objs[1]:
                    try:
                        vc1, vc2 = st.columns(2)
                        with vc1:
                            render_gauge(raw_prob, info["label"],
                                        risk_res.get("color", "#6c63ff"))
                        with vc2:
                            if settings["show_trend"]:
                                prog = simulate_risk_progression(raw_prob)
                                render_progression(prog, info["label"],
                                                   risk_res.get("color", "#6c63ff"))
                    except Exception:
                        st.info("Charts could not be rendered.")

                # TAB 3 — XAI
                with tab_objs[2]:
                    try:
                        if settings["show_xai"]:
                            importances = get_feature_importance(disease_key)
                            if importances:
                                render_feature_chart(importances, info["label"])
                                st.caption("Higher % = stronger influence on the prediction.")
                            else:
                                st.info("Feature importance not available for this model.")
                        else:
                            st.info("Enable 'Feature Importance' in sidebar to view.")
                    except Exception:
                        st.info("XAI analysis could not be displayed.")

                # TAB 4 — Tips
                with tab_objs[3]:
                    try:
                        tips = get_health_tips(disease_key, level)
                        for tip in tips:
                            st.markdown(f'<div class="tip-item">✅ {tip}</div>',
                                        unsafe_allow_html=True)
                    except Exception:
                        st.info("Health tips unavailable.")

                # TAB 5 — Diet Plan
                if settings["show_diet"] and len(tab_objs) > 4:
                    with tab_objs[4]:
                        try:
                            render_diet_card(disease_key, level)
                        except Exception:
                            st.info("Diet recommendations could not be loaded.")

        except Exception:
            # Last-resort catch — never crash the page
            st.markdown(f"""
            <div class="err-box">
              {info['icon']} <b>{info['label']}</b> result section encountered an issue.
              Please try reloading the page.
            </div>
            """, unsafe_allow_html=True)

    # ── Model Metrics ─────────────────────────────────────────────────────────
    try:
        with st.expander("📋 ML Model Accuracy Metrics", expanded=False):
            all_m = load_metrics()
            if all_m:
                rows = []
                for key in results:
                    m = all_m.get(key, {})
                    if m:
                        rows.append({
                            "Disease"  : DISEASES.get(key, {}).get("label", key),
                            "Accuracy" : f"{m.get('accuracy',0):.4f}",
                            "Precision": f"{m.get('precision',0):.4f}",
                            "Recall"   : f"{m.get('recall',0):.4f}",
                            "F1-Score" : f"{m.get('f1_score',0):.4f}",
                            "CV Acc"   : f"{m.get('cv_mean',0):.4f} ± {m.get('cv_std',0):.4f}",
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Metrics file not found. Run train_models.py to generate.")
    except Exception:
        pass  # silently skip metrics on error

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="badge-disclaimer">
    ⚕️ <b>Medical Disclaimer</b>: This application is for educational and informational purposes
    only. All assessments use ML models trained on synthetic data and are <b>not</b> a substitute
    for professional medical advice, diagnosis, or treatment. Diet recommendations are general
    guidelines — always consult a qualified healthcare professional or registered dietitian.
    No user data is stored or transmitted.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL TRAINING (first-run or on-demand)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_models():
    """Train all models if not present. Catches and displays any training errors."""
    if models_exist():
        return

    existing = partial_models_exist()
    msg = (f"Only {len(existing)}/10 models found. Retraining all models..."
           if existing else "First-time setup: Training 10 ML models...")

    placeholder = st.empty()
    with placeholder.container():
        st.info(f"⏳ {msg} This takes about 60–90 seconds.")
        prog = st.progress(0)
        status = st.empty()

    try:
        from train_models import train_all_models
        # Monkey-patch print to update UI
        import builtins
        _orig_print = builtins.print
        counter = [0]
        def _ui_print(*args, **kwargs):
            _orig_print(*args, **kwargs)
            counter[0] += 1
            try:
                pct = min(99, counter[0] * 7)
                prog.progress(pct)
                if args:
                    status.caption(str(args[0])[:80])
            except Exception:
                pass
        builtins.print = _ui_print
        train_all_models()
        builtins.print = _orig_print
        prog.progress(100)
        status.empty()
    except Exception as e:
        st.error(f"Model training encountered an issue: {str(e)[:200]}")
        st.info("Please run `python train_models.py` manually from your terminal.")
        st.stop()
    finally:
        placeholder.empty()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    inject_css()

    # Sidebar settings
    settings = render_sidebar()

    # Header
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.3rem 0;">
      <h1 style="font-size:2.3rem;font-weight:800;
        background:linear-gradient(135deg,#6c63ff,#e91e8c);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;">
        🏥 AI Health Risk Analyzer
      </h1>
      <p style="color:#9ba3b2;font-size:1rem;margin-top:0.3rem;">
        10-Disease Risk Assessment  ·  Diet Recommendations  ·  Explainable AI
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Check disease selection
    if not settings["selected"]:
        st.warning("⚠️ Please select at least one disease from the sidebar to begin.")
        st.stop()

    # Ensure models exist
    ensure_models()

    # ── STEP 1: Basic Info ────────────────────────────────────────────────────
    st.markdown("### Step 1 — Basic Information")
    with st.form("form_basic"):
        basic = panel_basic_info()
        ok1 = st.form_submit_button("✅ Save & Continue to Clinical Inputs →")

    if ok1:
        st.session_state["basic"]    = basic
        st.session_state["step"]     = 2
        st.session_state.pop("results", None)
    if "basic" not in st.session_state:
        st.stop()

    basic = st.session_state["basic"]

    # Confirmed basic info banner
    st.markdown(f"""
    <div class="hcard" style="border-left:4px solid #6c63ff;padding:0.8rem 1.2rem;">
      <b>✅ Basic Info:</b>
      Age {basic['age']} &nbsp;|&nbsp; BMI {basic['bmi']:.1f} &nbsp;|&nbsp;
      {'Male' if basic['gender']==1 else 'Female'} &nbsp;|&nbsp;
      {["Never Smoked","Ex-Smoker","Current Smoker"][basic['smoker_val']]}
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 2: Clinical Inputs ───────────────────────────────────────────────
    st.markdown("### Step 2 — Clinical Data for Selected Diseases")
    disease_inputs = {}

    # Only show PCOS for females
    selected_diseases = settings["selected"]
    if "pcos" in selected_diseases and basic["gender"] == 1:
        st.info("ℹ️ PCOS is a female hormonal condition — skipped as Biological Sex = Male.")
        selected_diseases = [d for d in selected_diseases if d != "pcos"]

    with st.form("form_clinical"):
        panel_map = {
            "diabetes"    : lambda: panel_diabetes(basic),
            "heart"       : lambda: panel_heart(basic),
            "parkinsons"  : lambda: panel_parkinsons(),
            "ckd"         : lambda: panel_ckd(basic),
            "liver"       : lambda: panel_liver(basic),
            "stroke"      : lambda: panel_stroke(basic),
            "breast_cancer": lambda: panel_breast_cancer(),
            "thyroid"     : lambda: panel_thyroid(basic),
            "pcos"        : lambda: panel_pcos(basic),
            "anemia"      : lambda: panel_anemia(basic),
        }

        for key in selected_diseases:
            if key in panel_map:
                try:
                    disease_inputs[key] = panel_map[key]()
                    st.markdown("---")
                except Exception:
                    st.warning(f"⚠️ Could not render input panel for {DISEASES[key]['label']}.")

        col_b, col_a = st.columns([1, 2])
        with col_b:
            back = st.form_submit_button("← Back to Basic Info")
        with col_a:
            analyze = st.form_submit_button("🔍 Analyze My Health Risk →", type="primary")

    if back:
        for k in ["basic", "step", "results"]:
            st.session_state.pop(k, None)
        st.rerun()

    if not analyze:
        st.stop()

    # ── STEP 3: Prediction ────────────────────────────────────────────────────
    progress_ph = st.empty()
    with progress_ph.container():
        st.markdown("""
        <div style="text-align:center;padding:1.5rem;">
          <div style="font-size:2rem;">🔬</div>
          <div style="font-size:1.1rem;font-weight:600;color:#6c63ff;">
            Analyzing your health biomarkers…</div>
          <div style="color:#9ba3b2;font-size:0.9rem;">
            AI models processing your inputs</div>
        </div>
        """, unsafe_allow_html=True)
        prog2 = st.progress(0)
        n = max(len(disease_inputs), 1)
        for i in range(n * 12):
            time.sleep(0.01)
            prog2.progress(min(99, int((i / (n * 12)) * 100)))
    progress_ph.empty()

    # ── Run predictions (fully isolated per disease) ──────────────────────────
    raw_probs = {}
    results   = {}

    for key, inputs in disease_inputs.items():
        try:
            pred = predict_disease(key, inputs)
            if pred["success"]:
                raw_probs[key] = pred["probability"]
                results[key]   = {"success": True, "raw_prob": pred["probability"]}
            else:
                results[key] = {
                    "success": False,
                    "error"  : pred.get("error", "Unknown error"),
                    "raw_prob": 0.5,
                }
        except Exception as e:
            results[key] = {"success": False, "error": str(e), "raw_prob": 0.5}

    # ── Cross-disease amplification (only on successful predictions) ──────────
    try:
        original_probs  = dict(raw_probs)
        amplified_probs = apply_cross_disease_amplifier(raw_probs)
    except Exception:
        amplified_probs = raw_probs
        original_probs  = raw_probs

    # ── Enrich results with risk classification ───────────────────────────────
    for key in results:
        if not results[key]["success"]:
            continue
        try:
            prob = amplified_probs.get(key, results[key]["raw_prob"])
            results[key]["raw_prob"]  = prob
            results[key]["risk"]      = classify_risk(prob, basic["age"])
            results[key]["amplified"] = (
                abs(prob - original_probs.get(key, prob)) > 0.001
            )
        except Exception:
            results[key]["success"] = False
            results[key]["error"]   = "Risk classification failed"

    # Store and render
    st.session_state["results"] = results
    render_results(results, basic, settings)

    # ── Reset button ──────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔄 Start New Assessment"):
        for k in ["basic", "step", "results"]:
            st.session_state.pop(k, None)
        st.rerun()


# Old main() kept for internal use
# Entry point is now run_app() defined below

# =============================================================================
#  AUTH UI — Login / Register / Profile / History
# =============================================================================

from auth import (init_db, register_user, login_user, save_assessment,
                  get_user_history, update_profile, change_password, get_total_users)

AUTH_CSS = """
<style>
.auth-wrap {
    max-width: 480px; margin: 2rem auto;
    background: #1e2130; border: 1px solid #2d3246;
    border-radius: 20px; padding: 2.5rem 2.8rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
}
.auth-logo {
    text-align:center; margin-bottom:1.5rem;
}
.auth-logo h1 {
    font-size:2rem; font-weight:800; margin:0;
    background:linear-gradient(135deg,#6c63ff,#e91e8c);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.auth-logo p { color:#9ba3b2; font-size:0.9rem; margin:0.3rem 0 0 0; }
.tab-btn {
    display:inline-block; width:48%; text-align:center;
    padding:0.55rem; border-radius:10px; cursor:pointer;
    font-weight:600; font-size:0.95rem;
    transition: all 0.2s;
}
.tab-active   { background:#6c63ff; color:white; }
.tab-inactive { background:#252840; color:#9ba3b2; }
.field-hint { font-size:0.75rem; color:#666; margin-top:2px; }
.strength-bar { height:4px; border-radius:2px; margin:4px 0; transition:all 0.3s; }
.user-badge {
    display:inline-flex; align-items:center; gap:8px;
    background:#252840; border:1px solid #2d3246;
    border-radius:20px; padding:4px 14px; font-size:0.85rem;
}
.history-row {
    background:#252840; border-radius:10px;
    padding:0.7rem 1rem; margin:0.4rem 0;
    display:flex; justify-content:space-between;
    align-items:center; font-size:0.85rem;
}
.score-pill {
    border-radius:12px; padding:2px 10px;
    font-weight:700; font-size:0.82rem;
}
</style>
"""


def password_strength(pwd: str) -> tuple:
    """Return (score 0-4, label, color)."""
    score = 0
    if len(pwd) >= 8:   score += 1
    if any(c.isupper() for c in pwd): score += 1
    if any(c.islower() for c in pwd): score += 1
    if any(c.isdigit() for c in pwd): score += 1
    labels = ["", "Weak", "Fair", "Good", "Strong"]
    colors = ["", "#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    return score, labels[score] if score else "", colors[score] if score else "#2d3246"


def render_auth_page():
    """Full-page login / register UI. Returns True when user is authenticated."""
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    init_db()

    # ── Session default ───────────────────────────────────────────────────────
    if "auth_tab" not in st.session_state:
        st.session_state["auth_tab"] = "login"

    # ── Logo ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 0 0;">
      <div style="font-size:3rem;">🏥</div>
      <h1 style="font-size:2.2rem; font-weight:800; margin:0.2rem 0;
        background:linear-gradient(135deg,#6c63ff,#e91e8c);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        HealthAI
      </h1>
      <p style="color:#9ba3b2; font-size:0.95rem; margin:0 0 0.5rem 0;">
        AI-Powered Multi-Disease Risk Assessment
      </p>
      <div style="display:flex; justify-content:center; gap:1.5rem;
                  font-size:0.8rem; color:#666; margin-bottom:1.5rem;">
        <span>🔬 10 Diseases</span>
        <span>🥗 Diet Plans</span>
        <span>🧠 Explainable AI</span>
        <span>🔒 Secure</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tab switcher ──────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        if st.button("🔑  Login", use_container_width=True,
                     type="primary" if st.session_state["auth_tab"] == "login" else "secondary"):
            st.session_state["auth_tab"] = "login"
            st.rerun()
    with col_r:
        if st.button("✍️  Register", use_container_width=True,
                     type="primary" if st.session_state["auth_tab"] == "register" else "secondary"):
            st.session_state["auth_tab"] = "register"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── LOGIN FORM ────────────────────────────────────────────────────────────
    if st.session_state["auth_tab"] == "login":
        st.markdown("#### 👋 Welcome Back")
        with st.form("login_form", clear_on_submit=False):
            ident = st.text_input("Username or Email",
                                   placeholder="Enter your username or email")
            pwd   = st.text_input("Password", type="password",
                                   placeholder="Enter your password")
            col_sub, col_hint = st.columns([2, 1])
            with col_sub:
                submitted = st.form_submit_button("Login →", use_container_width=True,
                                                   type="primary")
            with col_hint:
                st.markdown('<div class="field-hint" style="padding-top:0.7rem;">No account yet?<br>Click Register ↑</div>',
                            unsafe_allow_html=True)

        if submitted:
            if not ident or not pwd:
                st.error("Please fill in both fields.")
            else:
                with st.spinner("Verifying credentials…"):
                    result = login_user(ident, pwd)
                if result["success"]:
                    st.session_state["user"]          = result["user"]
                    st.session_state["authenticated"] = True
                    st.success(result["message"])
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")

    # ── REGISTER FORM ─────────────────────────────────────────────────────────
    else:
        st.markdown("#### 🌟 Create Your Free Account")
        with st.form("register_form", clear_on_submit=False):
            full_name = st.text_input("Full Name *",
                                       placeholder="e.g. Priya Sharma")
            col1, col2 = st.columns(2)
            with col1:
                username  = st.text_input("Username *",
                                           placeholder="e.g. priya_sharma")
            with col2:
                email     = st.text_input("Email *",
                                           placeholder="priya@email.com")
            col3, col4 = st.columns(2)
            with col3:
                gender = st.selectbox("Gender", ["Female", "Male", "Prefer not to say"])
            with col4:
                age    = st.number_input("Age", min_value=10, max_value=100, value=25)

            pwd1 = st.text_input("Password *", type="password",
                                  placeholder="Min 8 chars, 1 upper, 1 number")
            pwd2 = st.text_input("Confirm Password *", type="password",
                                  placeholder="Re-enter password")

            # Live password strength indicator
            if pwd1:
                score, label, color = password_strength(pwd1)
                bar_w = f"{score*25}%"
                st.markdown(f"""
                <div style="font-size:0.75rem;color:#9ba3b2;margin-top:-8px;">
                  Strength: <b style="color:{color}">{label}</b>
                  <div style="background:#2d3246;border-radius:2px;height:4px;margin-top:3px;">
                    <div style="width:{bar_w};background:{color};height:4px;border-radius:2px;transition:width 0.3s;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            agree = st.checkbox("I agree this app is for educational purposes only and not a medical tool.")
            submitted = st.form_submit_button("Create Account →",
                                               use_container_width=True, type="primary")

        if submitted:
            # Validation checks with friendly messages
            if not agree:
                st.warning("⚠️ Please accept the terms to continue.")
            elif not full_name or not username or not email or not pwd1 or not pwd2:
                st.error("❌ Please fill in all required fields (marked with *).")
            elif pwd1 != pwd2:
                st.error("❌ Passwords do not match. Please re-enter.")
            else:
                with st.spinner("Creating your account…"):
                    result = register_user(
                        username=username, email=email,
                        full_name=full_name, password=pwd1,
                        gender=gender, age=age
                    )
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                    st.info("👆 Now switch to the **Login** tab and sign in!")
                    st.session_state["auth_tab"] = "login"
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")

    # ── Footer stats ──────────────────────────────────────────────────────────
    try:
        total = get_total_users()
        st.markdown(f"""
        <div style="text-align:center; margin-top:2rem; color:#555; font-size:0.78rem;">
          🔒 No health data stored &nbsp;·&nbsp;
          👥 {total} registered user{'s' if total != 1 else ''} &nbsp;·&nbsp;
          ⚕️ For educational use only
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  PROFILE & HISTORY PAGE
# ─────────────────────────────────────────────────────────────────────────────

def render_profile_page():
    """User profile + assessment history."""
    user = st.session_state.get("user", {})

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
      <div style="width:60px; height:60px; border-radius:50%;
           background:linear-gradient(135deg,#6c63ff,#e91e8c);
           display:flex; align-items:center; justify-content:center;
           font-size:1.6rem; font-weight:800; color:white;">
        {user.get('full_name','U')[0].upper()}
      </div>
      <div>
        <div style="font-size:1.3rem; font-weight:700;">{user.get('full_name','User')}</div>
        <div style="color:#9ba3b2; font-size:0.85rem;">
          @{user.get('username','')} &nbsp;·&nbsp; {user.get('email','')}
        </div>
        <div style="color:#666; font-size:0.78rem;">
          Member since {user.get('created_at','')[:10]}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_prof, tab_hist, tab_pwd = st.tabs(["👤 Edit Profile", "📋 Assessment History", "🔒 Change Password"])

    # ── Edit Profile ─────────────────────────────────────────────────────────
    with tab_prof:
        with st.form("profile_form"):
            new_name   = st.text_input("Full Name", value=user.get("full_name", ""))
            new_gender = st.selectbox("Gender",
                                      ["Female", "Male", "Prefer not to say"],
                                      index=["Female","Male","Prefer not to say"]
                                      .index(user.get("gender","Female")
                                             if user.get("gender") in
                                             ["Female","Male","Prefer not to say"]
                                             else "Prefer not to say"))
            new_age    = st.number_input("Age", min_value=10, max_value=100,
                                          value=int(user.get("age", 25)))
            save_p = st.form_submit_button("💾 Save Changes", type="primary")
        if save_p:
            r = update_profile(user["id"], new_name, new_gender, new_age)
            if r["success"]:
                st.session_state["user"]["full_name"] = new_name
                st.session_state["user"]["gender"]    = new_gender
                st.session_state["user"]["age"]       = new_age
                st.success(r["message"])
            else:
                st.error(r["message"])

    # ── Assessment History ────────────────────────────────────────────────────
    with tab_hist:
        history = get_user_history(user["id"])
        if not history:
            st.info("📋 No assessments yet. Go to the **Analyzer** tab to check your health risk!")
        else:
            st.caption(f"Showing your last {len(history)} assessments")
            for row in history:
                score = row["health_score"] or 0
                color = ("#2ecc71" if score >= 65 else
                         "#f39c12" if score >= 35 else "#e74c3c")
                st.markdown(f"""
                <div class="history-row">
                  <div>
                    <div style="font-weight:600;">📅 {row['assessed_at'][:16]}</div>
                    <div style="color:#9ba3b2; font-size:0.8rem; margin-top:2px;">
                      {row['diseases']}
                    </div>
                  </div>
                  <div style="text-align:right;">
                    <span class="score-pill"
                          style="background:{color}22; color:{color};">
                      {score:.0f}/100
                    </span>
                    <div style="font-size:0.75rem; color:#666; margin-top:3px;">
                      {row['risk_summary'][:40] if row.get('risk_summary') else ''}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Change Password ───────────────────────────────────────────────────────
    with tab_pwd:
        with st.form("pwd_form"):
            cur_pwd  = st.text_input("Current Password", type="password")
            new_pwd1 = st.text_input("New Password", type="password",
                                      placeholder="Min 8 chars, 1 upper, 1 number")
            new_pwd2 = st.text_input("Confirm New Password", type="password")
            save_pwd = st.form_submit_button("🔒 Update Password", type="primary")
        if save_pwd:
            if not cur_pwd or not new_pwd1 or not new_pwd2:
                st.error("Please fill all password fields.")
            elif new_pwd1 != new_pwd2:
                st.error("New passwords do not match.")
            else:
                r = change_password(user["id"], cur_pwd, new_pwd1)
                if r["success"]:
                    st.success(r["message"])
                else:
                    st.error(r["message"])


# ─────────────────────────────────────────────────────────────────────────────
#  NAVBAR (shown when logged in)
# ─────────────────────────────────────────────────────────────────────────────

def render_navbar() -> str:
    """Top navbar with user info and page switcher. Returns active page."""
    user = st.session_state.get("user", {})
    first = user.get("full_name", "User").split()[0]

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
         padding:0.6rem 0; border-bottom:1px solid #2d3246; margin-bottom:1rem;">
      <div style="font-size:1.2rem; font-weight:800;
           background:linear-gradient(135deg,#6c63ff,#e91e8c);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        🏥 HealthAI
      </div>
      <div class="user-badge">
        <span style="background:#6c63ff; color:white; border-radius:50%;
               width:26px; height:26px; display:inline-flex;
               align-items:center; justify-content:center;
               font-size:0.8rem; font-weight:700;">
          {first[0].upper()}
        </span>
        <span style="color:#e8eaf0; font-size:0.85rem;">Hi, <b>{first}</b></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Page navigation
    pages = {"🔬 Analyzer": "analyzer", "👤 My Profile": "profile"}
    cols  = st.columns([2, 1, 1, 1])
    active_page = st.session_state.get("active_page", "analyzer")

    with cols[1]:
        if st.button("🔬 Analyzer", use_container_width=True,
                     type="primary" if active_page == "analyzer" else "secondary"):
            st.session_state["active_page"] = "analyzer"
            st.rerun()
    with cols[2]:
        if st.button("👤 My Profile", use_container_width=True,
                     type="primary" if active_page == "profile" else "secondary"):
            st.session_state["active_page"] = "profile"
            st.rerun()
    with cols[3]:
        if st.button("🚪 Logout", use_container_width=True):
            # Clear all session state
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    return st.session_state.get("active_page", "analyzer")


# ─────────────────────────────────────────────────────────────────────────────
#  PATCHED main() — with auth gate
# ─────────────────────────────────────────────────────────────────────────────

def run_app():
    """Entry point: shows auth page or main app depending on session."""
    inject_css()
    init_db()   # ensure DB exists on every run

    # ── Not logged in → show auth page ───────────────────────────────────────
    if not st.session_state.get("authenticated", False):
        render_auth_page()
        return

    # ── Logged in → show navbar + active page ─────────────────────────────────
    active_page = render_navbar()

    # Sidebar only on analyzer page
    if active_page == "analyzer":
        settings = render_sidebar()

        if not settings["selected"]:
            st.warning("⚠️ Please select at least one disease from the sidebar to begin.")
            return

        ensure_models()
        _run_analyzer(settings)

    elif active_page == "profile":
        render_profile_page()


def _run_analyzer(settings: dict):
    """The core health analyzer flow (extracted from old main())."""
    user = st.session_state.get("user", {})

    # Header
    st.markdown("""
    <div style="text-align:center; padding:0.8rem 0 0.3rem 0;">
      <h1 style="font-size:2rem; font-weight:800;
        background:linear-gradient(135deg,#6c63ff,#e91e8c);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
        AI Health Risk Analyzer
      </h1>
      <p style="color:#9ba3b2; font-size:0.95rem; margin-top:0.3rem;">
        10-Disease Risk Assessment · Diet Recommendations · Explainable AI
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 1: Basic Info ────────────────────────────────────────────────────
    st.markdown("### Step 1 — Basic Information")

    # Pre-fill from user profile if available
    default_age    = int(user.get("age", 30)) if user.get("age", 0) > 0 else 30
    default_gender = user.get("gender", "Female")

    with st.form("form_basic"):
        basic = panel_basic_info()
        ok1   = st.form_submit_button("✅ Save & Continue to Clinical Inputs →")

    if ok1:
        st.session_state["basic"]  = basic
        st.session_state["step"]   = 2
        st.session_state.pop("results", None)

    if "basic" not in st.session_state:
        return

    basic = st.session_state["basic"]
    st.markdown(f"""
    <div class="hcard" style="border-left:4px solid #6c63ff; padding:0.8rem 1.2rem;">
      <b>✅ Basic Info:</b>
      Age {basic['age']} &nbsp;|&nbsp; BMI {basic['bmi']:.1f} &nbsp;|&nbsp;
      {'Male' if basic['gender']==1 else 'Female'} &nbsp;|&nbsp;
      {["Never Smoked","Ex-Smoker","Current Smoker"][basic['smoker_val']]}
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 2: Clinical Inputs ───────────────────────────────────────────────
    st.markdown("### Step 2 — Clinical Data for Selected Diseases")
    disease_inputs = {}

    selected_diseases = settings["selected"]
    if "pcos" in selected_diseases and basic["gender"] == 1:
        st.info("ℹ️ PCOS is a female hormonal condition — skipped for Male.")
        selected_diseases = [d for d in selected_diseases if d != "pcos"]

    with st.form("form_clinical"):
        panel_map = {
            "diabetes"    : lambda: panel_diabetes(basic),
            "heart"       : lambda: panel_heart(basic),
            "parkinsons"  : lambda: panel_parkinsons(),
            "ckd"         : lambda: panel_ckd(basic),
            "liver"       : lambda: panel_liver(basic),
            "stroke"      : lambda: panel_stroke(basic),
            "breast_cancer": lambda: panel_breast_cancer(),
            "thyroid"     : lambda: panel_thyroid(basic),
            "pcos"        : lambda: panel_pcos(basic),
            "anemia"      : lambda: panel_anemia(basic),
        }
        for key in selected_diseases:
            if key in panel_map:
                try:
                    disease_inputs[key] = panel_map[key]()
                    st.markdown("---")
                except Exception:
                    st.warning(f"⚠️ Could not render {DISEASES[key]['label']} inputs.")

        col_b, col_a = st.columns([1, 2])
        with col_b:
            back    = st.form_submit_button("← Back to Basic Info")
        with col_a:
            analyze = st.form_submit_button("🔍 Analyze My Health Risk →", type="primary")

    if back:
        for k in ["basic", "step", "results"]:
            st.session_state.pop(k, None)
        st.rerun()

    if not analyze:
        return

    # ── Loading animation ─────────────────────────────────────────────────────
    ph = st.empty()
    with ph.container():
        st.markdown("""
        <div style="text-align:center; padding:1.5rem;">
          <div style="font-size:2rem;">🔬</div>
          <div style="font-size:1.1rem; font-weight:600; color:#6c63ff;">
            Analyzing your health biomarkers…</div>
        </div>
        """, unsafe_allow_html=True)
        prog = st.progress(0)
        for i in range(max(len(disease_inputs), 1) * 12):
            time.sleep(0.01)
            prog.progress(min(99, int(i / (max(len(disease_inputs),1) * 12) * 100)))
    ph.empty()

    # ── Predict ───────────────────────────────────────────────────────────────
    raw_probs, results = {}, {}
    for key, inputs in disease_inputs.items():
        try:
            pred = predict_disease(key, inputs)
            if pred["success"]:
                raw_probs[key]  = pred["probability"]
                results[key]    = {"success": True, "raw_prob": pred["probability"]}
            else:
                results[key]    = {"success": False, "error": pred.get("error",""), "raw_prob": 0.5}
        except Exception as e:
            results[key]        = {"success": False, "error": str(e), "raw_prob": 0.5}

    # Cross-disease amplification
    try:
        original_probs  = dict(raw_probs)
        amplified_probs = apply_cross_disease_amplifier(raw_probs)
    except Exception:
        amplified_probs = raw_probs
        original_probs  = raw_probs

    # Enrich with risk levels
    for key in results:
        if not results[key]["success"]:
            continue
        try:
            prob = amplified_probs.get(key, results[key]["raw_prob"])
            results[key]["raw_prob"]  = prob
            results[key]["risk"]      = classify_risk(prob, basic["age"])
            results[key]["amplified"] = abs(prob - original_probs.get(key, prob)) > 0.001
        except Exception:
            results[key]["success"] = False

    # ── Save assessment to history ────────────────────────────────────────────
    try:
        if user.get("id") and raw_probs:
            hs = compute_health_score(raw_probs)
            risk_labels = []
            for k, v in results.items():
                if v.get("success") and "risk" in v:
                    risk_labels.append(f"{DISEASES[k]['label']}: {v['risk']['level']}")
            save_assessment(
                user_id=user["id"],
                diseases=list(raw_probs.keys()),
                health_score=hs["score"],
                risk_summary=" | ".join(risk_labels[:3])
            )
    except Exception:
        pass  # silently skip — non-critical

    st.session_state["results"] = results
    render_results(results, basic, settings)

    st.markdown("---")
    if st.button("🔄 Start New Assessment"):
        for k in ["basic", "step", "results"]:
            st.session_state.pop(k, None)
        st.rerun()


# Override entry point
if __name__ == "__main__":
    run_app()
