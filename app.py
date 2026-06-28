import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECGSense | Clinical Decision Support",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1a1f36;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0057B8 0%, #003580 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #e8f0fe !important; }
[data-testid="stSidebar"] .stRadio label {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 500;
    font-size: 0.88rem;
    letter-spacing: 0.01em;
    transition: background 0.18s;
    margin-bottom: 2px;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.14);
}

/* ── Main container ── */
.main .block-container {
    padding: 2.4rem 3rem 4rem;
    max-width: 1160px;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0057B8 0%, #0078f0 55%, #00afd8 100%);
    border-radius: 16px;
    padding: 3rem 3.2rem;
    color: #fff !important;
    margin-bottom: 2.4rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "🫀";
    position: absolute;
    right: 2.8rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.14;
    pointer-events: none;
}
.hero .pill {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    color: #fff !important;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.6rem;
    line-height: 1.2;
}
.hero p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 1rem;
    margin: 0;
    max-width: 580px;
    line-height: 1.65;
}

/* ── Cards ── */
.card {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 1px 12px rgba(0,87,184,0.07);
    padding: 1.5rem 1.7rem;
    margin-bottom: 1rem;
    border: 1px solid #e8ecf5;
    transition: box-shadow 0.22s, transform 0.18s;
}
.card:hover {
    box-shadow: 0 4px 24px rgba(0,87,184,0.12);
    transform: translateY(-2px);
}
.card-blue  { border-left: 4px solid #0057B8; }
.card-green { border-left: 4px solid #0a9a60; }
.card-red   { border-left: 4px solid #D62828; }

/* ── Feature cards (Home) ── */
.feat-card {
    background: #fff;
    border-radius: 14px;
    box-shadow: 0 2px 16px rgba(0,87,184,0.07);
    border: 1px solid #e8ecf5;
    padding: 2rem 1.8rem 1.8rem;
    height: 100%;
    transition: box-shadow 0.22s, transform 0.18s;
}
.feat-card:hover {
    box-shadow: 0 6px 28px rgba(0,87,184,0.13);
    transform: translateY(-3px);
}
.feat-icon {
    font-size: 2.2rem;
    margin-bottom: 1rem;
}
.feat-title {
    font-size: 1rem;
    font-weight: 700;
    color: #0057B8;
    margin-bottom: 0.5rem;
}
.feat-desc {
    font-size: 0.85rem;
    color: #4a5568;
    line-height: 1.65;
    margin: 0;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8492b4;
    margin: 2rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e8ecf5;
}

/* ── Stat chips ── */
.stat-chip {
    background: #f4f7ff;
    border: 1px solid #dce5f7;
    border-radius: 10px;
    padding: 1rem 1rem;
    text-align: center;
}
.stat-chip .val {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0057B8;
    line-height: 1.15;
}
.stat-chip .lbl {
    font-size: 0.7rem;
    color: #6b7a9e;
    margin-top: 3px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Prediction result box ── */
.pred-box {
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    border: 2px solid;
    margin-bottom: 1rem;
}
.pred-normal  { background: #f0faf5; border-color: #0a9a60; }
.pred-warning { background: #fff8f0; border-color: #e67e22; }
.pred-danger  { background: #fff4f4; border-color: #D62828; }

/* ── Instructions step list ── */
.step-list {
    counter-reset: step;
    list-style: none;
    padding: 0;
    margin: 0;
}
.step-list li {
    counter-increment: step;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 0.55rem 0;
    font-size: 0.88rem;
    color: #1a1f36;
    border-bottom: 1px solid #f0f2f8;
    line-height: 1.55;
}
.step-list li:last-child { border-bottom: none; }
.step-list li::before {
    content: counter(step);
    flex-shrink: 0;
    width: 22px;
    height: 22px;
    background: #0057B8;
    color: #fff;
    border-radius: 50%;
    font-size: 0.72rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 1px;
}

/* ── Disclaimer ── */
.disclaimer {
    background: #fffbea;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    font-size: 0.82rem;
    color: #7c5a0a !important;
    margin-top: 1.4rem;
    line-height: 1.6;
}

/* ── Footer ── */
.ecg-footer {
    border-top: 1px solid #e8ecf5;
    margin-top: 3.5rem;
    padding-top: 1.4rem;
    text-align: center;
    font-size: 0.79rem;
    color: #8492b4 !important;
}
.ecg-footer strong { color: #0057B8; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: #0057B8 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.6rem !important;
    transition: background 0.18s !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover { background: #0040a0 !important; }
div[data-testid="stMetric"] { display: none; }
h1,h2,h3,h4 { font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1.4rem 0 2rem;'>
        <div style='font-size:2.4rem; line-height:1;'>🫀</div>
        <div style='font-size:1.15rem; font-weight:700; margin-top:0.5rem; letter-spacing:0.02em;'>ECGSense</div>
        <div style='font-size:0.68rem; opacity:0.6; margin-top:3px; text-transform:uppercase;
                    letter-spacing:0.1em;'>Clinical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "",
        [
            "🏠  Home",
            "🧪  Demo Prediction",
            "📁  Upload ECG",
            "📈  ECG Signal Explorer",
            "📖  Instructions",
        ],
        label_visibility="collapsed",
    )

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND  (unchanged — model load, data load, dictionaries)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    model    = load_model("ecg_classifier.keras")
    test_df  = pd.read_csv("mitbih_test.csv", header=None)
    X_test   = test_df.iloc[:, :-1].values
    y_test   = test_df.iloc[:, -1].values.astype(int)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return model, X_test, y_test, X_test_cnn

model, X_test, y_test, X_test_cnn = load_resources()

CLASS_LABELS = {
    0: "Normal Beat (N)",
    1: "Supraventricular Beat (S)",
    2: "Ventricular Beat (V)",
    3: "Fusion Beat (F)",
    4: "Unknown Beat (Q)",
}
CLASS_SHORT = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
CLASS_RISK  = {0: "Low", 1: "Moderate", 2: "High", 3: "Moderate", 4: "Moderate"}
CLASS_COLOR = {0: "pred-normal", 1: "pred-warning", 2: "pred-danger", 3: "pred-warning", 4: "pred-warning"}
CLASS_EMOJI = {0: "✅", 1: "⚠️", 2: "🚨", 3: "⚠️", 4: "❓"}
CLASS_INTERP = {
    0: "The heartbeat pattern appears within normal sinus rhythm parameters. No immediate intervention indicated.",
    1: "Premature supraventricular complex detected. May indicate atrial ectopy. Clinical correlation advised.",
    2: "Ventricular ectopic beat identified. This may indicate myocardial irritability. Prompt clinical review recommended.",
    3: "Fusion beat detected — a combination of normal and ventricular conduction. Warrants further evaluation.",
    4: "Beat morphology is ambiguous. Manual review by a qualified cardiologist is strongly advised.",
}
CLASS_REC = {
    0: "Continue routine monitoring. No urgent intervention required.",
    1: "Consider Holter monitoring. Evaluate for structural heart disease or electrolyte imbalance.",
    2: "Cardiology referral recommended. Evaluate for structural or ischemic pathology.",
    3: "Consult cardiology for further investigation and possible electrophysiology study.",
    4: "Refer signal to a cardiologist for manual interpretation and clinical correlation.",
}

def footer():
    st.markdown("""
    <div class='ecg-footer'>
        Developed by <strong>Anusha Rani and Keerthi Sowmya</strong> &nbsp;·&nbsp;
        Biomedical Engineering (Minor: AI/ML) &nbsp;·&nbsp;
        University College of Engineering (A), Osmania University, Hyderabad &nbsp;·&nbsp; 2026
    </div>
    """, unsafe_allow_html=True)

def _plot_signal(signal, title, figsize=(11, 2.8), show_grid=True):
    """Shared ECG waveform plotting — preserves all existing logic."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="#f8faff")
    ax.set_facecolor("#f8faff")
    ax.plot(signal, color="#0057B8", linewidth=1.3)
    ax.set_xlabel("Sample", fontsize=8, color="#6b7a9e")
    ax.set_ylabel("Amplitude (mV)", fontsize=8, color="#6b7a9e")
    ax.set_title(title, fontsize=9, color="#1a1f36", fontweight="600")
    if show_grid:
        ax.grid(True, linestyle="--", alpha=0.35, color="#c8d9f5")
    ax.tick_params(labelsize=7, colors="#6b7a9e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#d6e0f4")
    plt.tight_layout(pad=0.5)
    return fig

def _prob_chart(probs, pred_class):
    """Shared horizontal probability bar chart — preserves all existing logic."""
    fig, ax = plt.subplots(figsize=(5, 2.2), facecolor="#f8faff")
    ax.set_facecolor("#f8faff")
    bar_colors = ["#0057B8" if i == pred_class else "#d6e0f4" for i in range(5)]
    ax.barh(list(CLASS_SHORT.values()), probs * 100, color=bar_colors, height=0.52, edgecolor="none")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", fontsize=8, color="#6b7a9e")
    ax.tick_params(labelsize=8, colors="#6b7a9e")
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#e0e8f4")
    for sp in ax.spines.values():
        sp.set_edgecolor("#d6e0f4")
    plt.tight_layout(pad=0.5)
    return fig

def _render_prediction(pred_class, confidence, true_label=None):
    """Render prediction result card + optional ground-truth chip."""
    css_cls    = CLASS_COLOR[pred_class]
    emoji      = CLASS_EMOJI[pred_class]
    risk       = CLASS_RISK[pred_class]
    risk_color = {"Low": "#0a9a60", "Moderate": "#e67e22", "High": "#D62828"}[risk]

    st.markdown(f"""
    <div class='pred-box {css_cls}'>
        <div style='font-size:1.8rem; margin-bottom:0.4rem; line-height:1;'>{emoji}</div>
        <div style='font-size:1.3rem; font-weight:700; color:#1a1f36; margin-bottom:0.3rem;'>
            {CLASS_LABELS[pred_class]}
        </div>
        <div style='font-size:0.8rem; color:#6b7a9e;'>Predicted Class</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(confidence))
    st.markdown(
        f"<div style='font-size:0.83rem; color:#4a5568; margin-bottom:0.6rem;'>"
        f"<strong>{confidence:.1f}%</strong> confidence</div>",
        unsafe_allow_html=True,
    )

    chip_a, chip_b = st.columns(2)
    with chip_a:
        st.markdown(f"""
        <div class='stat-chip'>
            <div class='val' style='color:{risk_color}; font-size:1.1rem;'>{risk}</div>
            <div class='lbl'>Risk Level</div>
        </div>""", unsafe_allow_html=True)
    with chip_b:
        if true_label is not None:
            match = "✅ Match" if pred_class == true_label else "❌ Mismatch"
            st.markdown(f"""
            <div class='stat-chip'>
                <div class='val' style='font-size:0.95rem;'>{match}</div>
                <div class='lbl'>vs Ground Truth</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='stat-chip'>
                <div class='val' style='font-size:1.1rem; color:#0057B8;'>{CLASS_SHORT[pred_class]}</div>
                <div class='lbl'>Class Code</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='card card-blue' style='margin-top:0.9rem;'>
        <div style='font-size:0.72rem; font-weight:700; color:#0057B8; text-transform:uppercase;
                    letter-spacing:0.06em; margin-bottom:0.4rem;'>Clinical Interpretation</div>
        <p style='font-size:0.87rem; color:#1a1f36; margin:0; line-height:1.65;'>{CLASS_INTERP[pred_class]}</p>
    </div>
    <div class='card card-green'>
        <div style='font-size:0.72rem; font-weight:700; color:#0a7a4a; text-transform:uppercase;
                    letter-spacing:0.06em; margin-bottom:0.4rem;'>Clinical Recommendation</div>
        <p style='font-size:0.87rem; color:#1a1f36; margin:0; line-height:1.65;'>{CLASS_REC[pred_class]}</p>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════
if nav == "🏠  Home":
    st.markdown("""
    <div class='hero'>
        <div class='pill'>🏥 Clinical Decision Support</div>
        <h1>AI-Assisted Arrhythmia Detection</h1>
        <p>ECGSense combines a trained deep learning model with ECG visualisation and
           structured clinical guidance to support rapid heartbeat assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Three feature cards ──────────────────────────────────────────────
    c1, c2, c3 = st.columns(3, gap="medium")
    features = [
        ("🧠", "AI Classification",
         "Automatically classifies ECG heartbeat signals into five heartbeat categories "
         "using a trained deep learning model."),
        ("📤", "Upload ECG",
         "Upload your own ECG CSV file for real-time heartbeat analysis and prediction."),
        ("📈", "ECG Visualization",
         "Visualize ECG waveforms and inspect heartbeat morphology with an interactive "
         "signal viewer."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], features):
        with col:
            st.markdown(f"""
            <div class='feat-card'>
                <div class='feat-icon'>{icon}</div>
                <div class='feat-title'>{title}</div>
                <p class='feat-desc'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Clinical Decision Support blurb ─────────────────────────────────
    st.markdown("<div style='margin-top:2.4rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card card-blue' style='padding:1.6rem 2rem;'>
        <div style='font-size:0.72rem; font-weight:700; color:#0057B8; text-transform:uppercase;
                    letter-spacing:0.08em; margin-bottom:0.6rem;'>Clinical Decision Support</div>
        <p style='font-size:0.92rem; color:#1a1f36; margin:0; line-height:1.75;'>
            ECGSense assists clinicians by combining deep learning predictions with ECG visualisation
            and confidence scores to support rapid heartbeat assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    footer()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DEMO PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "🧪  Demo Prediction":
    st.markdown("""
    <div class='hero'>
        <div class='pill'>🧪 MIT-BIH Test Set</div>
        <h1>Demo Prediction</h1>
        <p>Select a sample, load it, and run the trained CNN model to obtain a heartbeat
           classification with confidence and clinical guidance.</p>
    </div>
    """, unsafe_allow_html=True)

    if "pred_history" not in st.session_state:
        st.session_state.pred_history = []
    if "demo_loaded_idx" not in st.session_state:
        st.session_state.demo_loaded_idx = None

    # ── Controls row ──────────────────────────────────────────────────────
    ctrl_l, ctrl_r = st.columns([2, 1], gap="medium")
    with ctrl_l:
        sample_options = ["Random Sample"] + [f"Sample {i}" for i in range(len(X_test))]
        selected_option = st.selectbox("Select Sample", sample_options, label_visibility="visible")
    with ctrl_r:
        st.markdown("<div style='margin-top:1.78rem;'></div>", unsafe_allow_html=True)
        load_btn = st.button("🔄  Load Sample", use_container_width=True)

    if load_btn:
        if selected_option == "Random Sample":
            st.session_state.demo_loaded_idx = int(np.random.randint(0, len(X_test)))
        else:
            st.session_state.demo_loaded_idx = int(selected_option.split(" ")[1])

    sample_idx = st.session_state.demo_loaded_idx

    if sample_idx is None:
        st.markdown("""
        <div class='card' style='text-align:center; padding:3rem 2rem; color:#8492b4; margin-top:1rem;'>
            <div style='font-size:2.8rem; margin-bottom:0.8rem;'>🔄</div>
            <strong>No Sample Loaded</strong>
            <p style='font-size:0.85rem; margin:0.5rem 0 0;'>Select a sample from the dropdown and click <em>Load Sample</em>.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        true_label = y_test[sample_idx]
        signal = X_test[sample_idx]

        # ── Two-column layout ─────────────────────────────────────────────
        left, right = st.columns([1, 1.55], gap="large")

        with left:
            st.markdown("<div class='section-label'>Sample Info</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='card card-blue' style='padding:1rem 1.3rem; margin-bottom:0.8rem;'>
                <div style='font-size:0.7rem; text-transform:uppercase; font-weight:700;
                            letter-spacing:0.07em; color:#0057B8; margin-bottom:0.3rem;'>Ground Truth Class</div>
                <div style='font-size:0.95rem; font-weight:600; color:#1a1f36;'>
                    {CLASS_LABELS[true_label]}
                </div>
                <div style='font-size:0.75rem; color:#8492b4; margin-top:4px;'>Sample #{sample_idx}</div>
            </div>
            """, unsafe_allow_html=True)

            predict_btn = st.button("🔍  Run Prediction", use_container_width=True)

            st.markdown("<div class='section-label'>ECG Waveform</div>", unsafe_allow_html=True)
            fig = _plot_signal(signal, f"Sample #{sample_idx}", figsize=(5, 2.3))
            st.pyplot(fig)
            plt.close()

        with right:
            st.markdown("<div class='section-label'>Prediction Result</div>", unsafe_allow_html=True)

            if predict_btn:
                with st.spinner("Running inference…"):
                    inp   = X_test_cnn[sample_idx:sample_idx + 1]
                    probs = model.predict(inp, verbose=0)[0]
                    pred_class = int(np.argmax(probs))
                    confidence = float(probs[pred_class]) * 100

                st.session_state.pred_history.append({
                    "Sample":     sample_idx,
                    "Predicted":  CLASS_SHORT[pred_class],
                    "True":       CLASS_SHORT[true_label],
                    "Confidence": f"{confidence:.1f}%",
                    "Risk":       CLASS_RISK[pred_class],
                })

                _render_prediction(pred_class, confidence, true_label=true_label)

                st.markdown("<div class='section-label'>Class Probabilities</div>", unsafe_allow_html=True)
                fig2 = _prob_chart(probs, pred_class)
                st.pyplot(fig2)
                plt.close()

            else:
                st.markdown("""
                <div class='card' style='text-align:center; padding:2.5rem; color:#8492b4;'>
                    <div style='font-size:2.5rem; margin-bottom:0.7rem;'>🔍</div>
                    <strong>Awaiting Prediction</strong>
                    <p style='font-size:0.84rem; margin:0.4rem 0 0;'>Click <em>Run Prediction</em> to analyse the loaded sample.</p>
                </div>
                """, unsafe_allow_html=True)

        # ── Prediction history ────────────────────────────────────────────
        if st.session_state.pred_history:
            st.markdown("<div class='section-label'>Prediction History</div>", unsafe_allow_html=True)
            hist_df = pd.DataFrame(st.session_state.pred_history)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            if st.button("🗑️  Clear History"):
                st.session_state.pred_history = []
                st.rerun()

    footer()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD ECG
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "📁  Upload ECG":
    st.markdown("""
    <div class='hero'>
        <div class='pill'>📁 Custom Signal</div>
        <h1>Upload ECG</h1>
        <p>Upload a CSV file containing 187 ECG signal values to obtain a real-time
           heartbeat classification and clinical interpretation.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── 1. Upload ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Upload CSV</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload ECG CSV or TXT",
        type=["csv", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Parse — support CSV/TXT, one-per-row or comma-separated
        try:
            raw_df = pd.read_csv(uploaded_file, header=None)
            signal = raw_df.values.flatten().astype(np.float32)
        except Exception as e:
            st.markdown(f"""
            <div class='card card-red' style='padding:1rem 1.4rem;'>
                <strong style='color:#D62828;'>❌ File Parse Error</strong>
                <p style='font-size:0.87rem; color:#1a1f36; margin:0.4rem 0 0;'>
                    Could not read the file: <code>{e}</code><br>
                    Ensure the file is a valid CSV or TXT with numeric values only.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Validate
        if len(signal) != 187:
            st.markdown(f"""
            <div class='card card-red' style='padding:1rem 1.4rem;'>
                <strong style='color:#D62828;'>❌ Invalid Signal Length</strong>
                <p style='font-size:0.87rem; color:#1a1f36; margin:0.4rem 0 0;'>
                    Expected <strong>187</strong> values — received <strong>{len(signal)}</strong>.
                    Please check the file and try again.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        st.success("✅ ECG file loaded successfully. Ready for analysis.")

        # ── 2. ECG Preview ────────────────────────────────────────────────
        st.markdown("<div class='section-label'>ECG Preview</div>", unsafe_allow_html=True)
        fig = _plot_signal(signal, "Uploaded ECG Signal")
        st.pyplot(fig)
        plt.close()

        # ── 3. Predict Button ─────────────────────────────────────────────
        st.markdown("<div class='section-label'>Analysis</div>", unsafe_allow_html=True)
        analyze_btn = st.button("🩺  Analyze ECG")

        if analyze_btn:
            with st.spinner("Analysing uploaded ECG signal…"):
                inp        = signal.astype(np.float32).reshape(1, 187, 1)
                probs      = model.predict(inp, verbose=0)[0]
                pred_class = int(np.argmax(probs))
                confidence = float(probs[pred_class]) * 100

            res_l, res_r = st.columns([1, 1], gap="large")

            # ── 4 & 5. Prediction Result + Clinical Interpretation ─────────
            with res_l:
                st.markdown("<div class='section-label'>Prediction Result</div>", unsafe_allow_html=True)
                _render_prediction(pred_class, confidence)

            # ── 6. Recommendation (probability chart) ─────────────────────
            with res_r:
                st.markdown("<div class='section-label'>Class Probabilities</div>", unsafe_allow_html=True)
                fig2 = _prob_chart(probs, pred_class)
                st.pyplot(fig2)
                plt.close()

        else:
            st.markdown("""
            <div class='card' style='text-align:center; padding:2.2rem; color:#8492b4;'>
                <div style='font-size:2.5rem; margin-bottom:0.7rem;'>🩺</div>
                <strong>Ready to Analyse</strong>
                <p style='font-size:0.84rem; margin:0.4rem 0 0;'>Click <em>Analyze ECG</em> to run the model.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='disclaimer'>
            ⚠️ <strong>Research Use Only.</strong> ECGSense is not a certified medical device.
            Outputs must not substitute professional clinical diagnosis.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class='card' style='text-align:center; padding:3rem 2rem; color:#8492b4;'>
            <div style='font-size:3rem; margin-bottom:0.8rem;'>📁</div>
            <strong>No File Uploaded</strong>
            <p style='font-size:0.85rem; margin:0.5rem 0 0;'>
                Upload a CSV or TXT file containing exactly <strong>187</strong> numeric ECG values.
            </p>
        </div>
        """, unsafe_allow_html=True)

    footer()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ECG SIGNAL EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "📈  ECG Signal Explorer":
    st.markdown("""
    <div class='hero'>
        <div class='pill'>📈 Waveform Inspection</div>
        <h1>ECG Signal Explorer</h1>
        <p>Select and load any sample from the MIT-BIH test set to inspect its
           waveform and signal statistics.</p>
    </div>
    """, unsafe_allow_html=True)

    if "explorer_loaded_idx" not in st.session_state:
        st.session_state.explorer_loaded_idx = None

    # ── Controls ──────────────────────────────────────────────────────────
    ctl_l, ctl_m, ctl_r, ctl_btn = st.columns([2.2, 0.9, 0.9, 1], gap="medium")
    with ctl_l:
        exp_options = ["Random Sample"] + [f"Sample {i}" for i in range(len(X_test))]
        exp_selection = st.selectbox("Select ECG Sample", exp_options, label_visibility="visible")
    with ctl_m:
        show_grid  = st.checkbox("Show Grid",       value=True)
    with ctl_r:
        show_stats = st.checkbox("Show Statistics", value=True)
    with ctl_btn:
        st.markdown("<div style='margin-top:1.78rem;'></div>", unsafe_allow_html=True)
        exp_load_btn = st.button("🔄  Load Sample", use_container_width=True)

    if exp_load_btn:
        if exp_selection == "Random Sample":
            st.session_state.explorer_loaded_idx = int(np.random.randint(0, len(X_test)))
        else:
            st.session_state.explorer_loaded_idx = int(exp_selection.split(" ")[1])

    vis_idx = st.session_state.explorer_loaded_idx

    if vis_idx is None:
        st.markdown("""
        <div class='card' style='text-align:center; padding:3rem 2rem; color:#8492b4; margin-top:1rem;'>
            <div style='font-size:2.8rem; margin-bottom:0.8rem;'>📈</div>
            <strong>No Sample Loaded</strong>
            <p style='font-size:0.85rem; margin:0.5rem 0 0;'>Select a sample and click <em>Load Sample</em> to display the waveform.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        signal   = X_test[vis_idx]
        true_cls = y_test[vis_idx]
        t        = np.arange(len(signal)) / 360  # MIT-BIH sampling rate = 360 Hz

        # ── Waveform (preserving all original plot logic) ─────────────────
        st.markdown("<div class='section-label'>Waveform</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(11, 3.2), facecolor="#f8faff")
        ax.set_facecolor("#f8faff")
        ax.plot(t, signal, color="#0057B8", linewidth=1.3)
        ax.axhline(0, color="#D62828", linewidth=0.6, linestyle="--", alpha=0.45)
        ax.set_xlabel("Time (s)", fontsize=9, color="#6b7a9e")
        ax.set_ylabel("Amplitude (mV)", fontsize=9, color="#6b7a9e")
        ax.set_title(
            f"Sample #{vis_idx}  ·  Class: {CLASS_LABELS[true_cls]}",
            fontsize=10, fontweight="600", color="#1a1f36",
        )
        if show_grid:
            ax.grid(True, linestyle="--", alpha=0.3, color="#c8d9f5")
        ax.tick_params(labelsize=8, colors="#6b7a9e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#d6e0f4")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close()

        # ── Statistics (4 metrics as specified) ──────────────────────────
        if show_stats:
            st.markdown("<div class='section-label'>Signal Statistics</div>", unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            stats = [
                (s1, "Maximum Amplitude", f"{signal.max():.4f}"),
                (s2, "Minimum Amplitude", f"{signal.min():.4f}"),
                (s3, "Mean Amplitude",    f"{signal.mean():.4f}"),
                (s4, "Signal Length",     "187"),
            ]
            for col, lbl, val in stats:
                with col:
                    st.markdown(f"""
                    <div class='stat-chip'>
                        <div class='val'>{val}</div>
                        <div class='lbl'>{lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)

    footer()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: INSTRUCTIONS
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "📖  Instructions":
    st.markdown("""
    <div class='hero'>
        <div class='pill'>📖 User Guide</div>
        <h1>Instructions</h1>
        <p>How to use ECGSense, interpret outputs, and understand supported formats.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Overview ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card card-blue'>
        <p style='font-size:0.9rem; color:#1a1f36; line-height:1.75; margin:0;'>
            ECGSense is an AI-assisted clinical decision support tool that classifies ECG heartbeat
            signals into five categories using a <strong>One-Dimensional Convolutional Neural Network
            (1D-CNN)</strong> trained on the <strong>MIT-BIH Arrhythmia Database</strong>.
            It provides per-beat confidence scores, risk stratification, and structured clinical
            recommendations to support — not replace — clinician judgment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── How-to sections — two columns ────────────────────────────────────
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("<div class='section-label'>Demo Prediction</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            <ol class='step-list'>
                <li>Open <strong>Demo Prediction</strong> from the sidebar.</li>
                <li>Select <em>Random Sample</em> or choose a specific sample number.</li>
                <li>Click <strong>Load Sample</strong> to load the ECG.</li>
                <li>Click <strong>Run Prediction</strong> to run the model.</li>
                <li>Review the predicted heartbeat class and confidence score.</li>
                <li>Review the ECG waveform alongside the result.</li>
                <li>Read the clinical interpretation and recommendation.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-label'>ECG Signal Explorer</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            <p style='font-size:0.88rem; color:#1a1f36; line-height:1.65; margin:0 0 0.6rem;'>
                Use the <strong>ECG Signal Explorer</strong> to inspect raw waveforms from the MIT-BIH
                test set without running a prediction.
            </p>
            <ol class='step-list'>
                <li>Select a sample from the dropdown or choose <em>Random Sample</em>.</li>
                <li>Toggle <strong>Show Grid</strong> for a gridded waveform view.</li>
                <li>Toggle <strong>Show Statistics</strong> to display signal metrics.</li>
                <li>Click <strong>Load Sample</strong> to render the waveform.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-label'>Upload ECG</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            <ol class='step-list'>
                <li>Open <strong>Upload ECG</strong> from the sidebar.</li>
                <li>Upload a CSV or TXT file containing 187 ECG signal values.</li>
                <li>The signal is automatically loaded and previewed.</li>
                <li>Click <strong>Analyze ECG</strong> to run the model.</li>
                <li>Review the predicted heartbeat class and confidence score.</li>
                <li>Read the clinical interpretation and recommendation.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-label'>Output Interpretation</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            <div style='font-size:0.87rem; color:#1a1f36; line-height:1.7;'>
                <p style='margin:0 0 0.5rem;'>
                    <strong>Predicted Class</strong> — the heartbeat category assigned by the model
                    (N, S, V, F, or Q).
                </p>
                <p style='margin:0 0 0.5rem;'>
                    <strong>Confidence Score</strong> — the model's probability for the predicted class
                    expressed as a percentage. Higher values indicate greater certainty.
                </p>
                <p style='margin:0 0 0.5rem;'>
                    <strong>Waveform</strong> — the raw 187-point ECG signal plotted against sample index.
                    Morphological features correspond to cardiac electrical activity.
                </p>
                <p style='margin:0;'>
                    <strong>Clinical Recommendation</strong> — a structured suggested action based on the
                    predicted class and associated risk level (Low / Moderate / High).
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Supported File Format ─────────────────────────────────────────────
    st.markdown("<div class='section-label'>Supported File Format</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card card-blue' style='padding:1.2rem 1.6rem;'>
        <div style='font-size:0.87rem; color:#1a1f36; line-height:1.8;'>
            <strong>Format:</strong> CSV (.csv) or plain text (.txt)<br>
            <strong>Values:</strong> Exactly 187 numeric ECG signal samples per file<br>
            <strong>Structure:</strong> One heartbeat per file — values on a single row (comma-separated)
            or one value per line<br>
            <strong>Missing values:</strong> Not permitted — all 187 values must be present and numeric
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Disclaimer</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='disclaimer'>
        ⚠️ ECGSense is intended for <strong>research and educational purposes only</strong> and must not
        replace professional medical diagnosis. It is not a certified medical device. All outputs should
        be reviewed by a qualified healthcare professional before any clinical decision is made.
    </div>
    """, unsafe_allow_html=True)

    footer()