import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ECGSense | AI Arrhythmia Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ───────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1a1f36;
}

/* ── Sidebar ───────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0057B8 0%, #003d82 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #e8f0fe !important; }
[data-testid="stSidebar"] .stRadio label {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 500;
    font-size: 0.9rem;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.12);
}

/* ── Main area ──────────────────────────── */
.main .block-container {
    padding: 2rem 2.5rem 3rem;
    max-width: 1200px;
}

/* ── Card component ─────────────────────── */
.card {
    background: #fff;
    border-radius: 14px;
    box-shadow: 0 2px 16px rgba(0,87,184,0.08);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    border: 1px solid #e6eaf4;
    transition: box-shadow 0.25s, transform 0.2s;
}
.card:hover {
    box-shadow: 0 6px 28px rgba(0,87,184,0.13);
    transform: translateY(-2px);
}
.card-blue {
    border-left: 4px solid #0057B8;
}
.card-red {
    border-left: 4px solid #D62828;
}
.card-green {
    border-left: 4px solid #0a9a60;
}
.card-orange {
    border-left: 4px solid #e67e22;
}

/* ── Hero banner ────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0057B8 0%, #0082f4 60%, #00b4d8 100%);
    border-radius: 18px;
    padding: 2.8rem 3rem;
    color: #fff !important;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🫀";
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 7rem;
    opacity: 0.18;
    pointer-events: none;
}
.hero h1 { color: #fff !important; margin-bottom: 0.3rem; font-size: 2.2rem; font-weight: 700; }
.hero p  { color: rgba(255,255,255,0.88) !important; font-size: 1.05rem; margin: 0; }
.hero .tag {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.4);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin-bottom: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.04em;
}

/* ── Section header ─────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.8rem 0 1rem;
    padding-bottom: 0.7rem;
    border-bottom: 2px solid #e6eaf4;
}
.section-header h2 {
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0;
    color: #0057B8;
}

/* ── Metric chips ───────────────────────── */
.metric-chip {
    background: #f0f5ff;
    border: 1px solid #c8d9f5;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-chip .value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #0057B8;
    line-height: 1.1;
}
.metric-chip .label {
    font-size: 0.75rem;
    color: #5a6483;
    margin-top: 4px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Heartbeat class cards ──────────────── */
.hb-card {
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    color: #fff;
}
.hb-normal   { background: linear-gradient(135deg, #0a9a60, #06c47a); }
.hb-supra    { background: linear-gradient(135deg, #0057B8, #0082f4); }
.hb-ventri   { background: linear-gradient(135deg, #D62828, #f25c5c); }
.hb-fusion   { background: linear-gradient(135deg, #7b2d8b, #b44fd8); }
.hb-unknown  { background: linear-gradient(135deg, #4a5568, #718096); }
.hb-card h3  { color: #fff !important; margin-bottom: 0.5rem; font-size: 1.05rem; }
.hb-card p   { color: rgba(255,255,255,0.9) !important; font-size: 0.85rem; margin: 0; }
.hb-card .badge {
    display: inline-block;
    background: rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 0.6rem;
}

/* ── Prediction result ──────────────────── */
.pred-result {
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-top: 1rem;
    border: 2px solid;
}
.pred-normal  { background: #f0faf6; border-color: #0a9a60; }
.pred-warning { background: #fff8f0; border-color: #e67e22; }
.pred-danger  { background: #fff5f5; border-color: #D62828; }

/* ── Architecture boxes ─────────────────── */
.arch-layer {
    background: #f4f7ff;
    border: 1.5px solid #c8d9f5;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.88rem;
    font-weight: 500;
}
.arch-layer .layer-icon {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: #0057B8;
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}

/* ── Footer ─────────────────────────────── */
.footer {
    background: #f0f5ff;
    border-top: 2px solid #c8d9f5;
    border-radius: 12px;
    padding: 1.4rem 2rem;
    text-align: center;
    margin-top: 3rem;
    color: #5a6483 !important;
    font-size: 0.82rem;
}
.footer strong { color: #0057B8; }

/* ── Disclaimer ─────────────────────────── */
.disclaimer {
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    font-size: 0.82rem;
    color: #78510a !important;
    margin-top: 1.2rem;
}

/* ── Streamlit overrides ─────────────────── */
.stButton > button {
    background: #0057B8 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.5rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #0040a0 !important;
}
div[data-testid="stMetric"] { display: none; }

h1,h2,h3,h4 { font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.8rem; margin-bottom:0.3rem;'>🫀</div>
        <div style='font-size:1.3rem; font-weight:700; letter-spacing:0.02em;'>ECGSense</div>
        <div style='font-size:0.72rem; opacity:0.7; margin-top:2px; text-transform:uppercase; letter-spacing:0.08em;'>Clinical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "",
        [
            "🏠  Home",
            "🫀  Prediction",
            "📈  ECG Visualization",
            "📊  Dataset Information",
            "❤️  Heartbeat Classes",
            "🧠  CNN Architecture",
            "📖  About ECG",
            "ℹ️  About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.2); margin:1.2rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; opacity:0.65; text-align:center; line-height:1.7;'>
        MIT-BIH Dataset · 5 Classes<br>
        CNN · 98.34% Accuracy<br>
        Biomedical Engineering Project
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# BACKEND — load model & data (cached)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    model = load_model("ecg_classifier.keras")
    test_df = pd.read_csv("mitbih_test.csv", header=None)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values.astype(int)
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
CLASS_SHORT = {0:"N", 1:"S", 2:"V", 3:"F", 4:"Q"}
CLASS_RISK  = {0:"Low", 1:"Moderate", 2:"High", 3:"Moderate", 4:"Moderate"}
CLASS_COLOR = {0:"pred-normal", 1:"pred-warning", 2:"pred-danger", 3:"pred-warning", 4:"pred-warning"}
CLASS_EMOJI = {0:"✅", 1:"⚠️", 2:"🚨", 3:"⚠️", 4:"❓"}
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
    <div class='footer'>
        Developed by <strong>Anusha Rani and Keerthi Sowmya</strong> · Biomedical Engineering (Minor: AI/ML)<br>
        University College of Engineering (A), Osmania University, Hyderabad · 2026
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════
if nav == "🏠  Home":
    st.markdown("""
    <div class='hero'>
        <div class='tag'>🏥 Clinical Decision Support System</div>
        <h1>AI-Assisted Arrhythmia Detection</h1>
        <p>Deep Learning–powered ECG classification using the MIT-BIH dataset.<br>
           Multi-class heartbeat detection with clinical-grade interpretation and confidence scoring.</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    cols = st.columns(4)
    metrics = [
        ("98.34%", "Model Accuracy"),
        ("21,892", "Test Samples"),
        ("5", "Heartbeat Classes"),
        ("187", "Signal Length"),
    ]
    for col, (val, lbl) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class='metric-chip'>
                <div class='value'>{val}</div>
                <div class='label'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'><h2>🎯 Project Objectives</h2></div>", unsafe_allow_html=True)

    obj_cols = st.columns(2)
    objectives = [
        ("🔍", "Automated Detection", "Classify ECG heartbeat signals into 5 clinically meaningful categories using a 1D CNN."),
        ("⚡", "Real-Time Analysis", "Deliver sub-second predictions with confidence scoring for point-of-care support."),
        ("🩺", "Clinical Guidance", "Provide structured clinical interpretation and referral recommendations per beat class."),
        ("📊", "Transparent AI", "Visualise model confidence and signal morphology to support clinician decision-making."),
    ]
    for i, (icon, title, desc) in enumerate(objectives):
        with obj_cols[i % 2]:
            st.markdown(f"""
            <div class='card card-blue'>
                <div style='font-size:1.6rem; margin-bottom:0.5rem;'>{icon}</div>
                <strong style='color:#0057B8;'>{title}</strong>
                <p style='margin:0.4rem 0 0; font-size:0.87rem; color:#4a5568;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'><h2>✨ Key Features</h2></div>", unsafe_allow_html=True)
    feat_cols = st.columns(3)
    features = [
        ("🧠", "1D CNN Model", "Trained on 87,554 labelled MIT-BIH samples. Handles class imbalance with SMOTE oversampling."),
        ("📈", "Interactive Visualisation", "Inspect raw ECG waveforms with amplitude/time axes, grid overlay, and zoom capability."),
        ("🏥", "CDSS Interface", "Hospital-grade UI with risk stratification, clinical notes, and structured recommendation output."),
    ]
    for col, (icon, title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(f"""
            <div class='card'>
                <div style='font-size:1.8rem; margin-bottom:0.6rem;'>{icon}</div>
                <strong>{title}</strong>
                <p style='margin:0.4rem 0 0; font-size:0.84rem; color:#4a5568;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer'>
        ⚠️ <strong>Medical Disclaimer:</strong> ECGSense is a research and educational prototype.
        It is NOT a certified medical device and must NOT be used as a substitute for professional clinical
        diagnosis. All outputs should be reviewed by a qualified cardiologist before any clinical decision is made.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🫀  Start ECG Prediction →"):
        st.session_state["_nav_override"] = "🫀  Prediction"
        st.rerun()

    footer()

# ════════════════════════════════════════════
# PAGE: PREDICTION
# ════════════════════════════════════════════
elif nav == "🫀  Prediction":
    st.markdown("""
    <div class='hero'>
        <div class='tag'>🫀 Real-Time Classification</div>
        <h1>ECG Heartbeat Prediction</h1>
        <p>Select a test sample, inspect its waveform, and obtain an AI-assisted classification with clinical context.</p>
    </div>
    """, unsafe_allow_html=True)

    if "pred_history" not in st.session_state:
        st.session_state.pred_history = []

    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.markdown("<div class='section-header'><h2>⚙️ Select Sample</h2></div>", unsafe_allow_html=True)
        sample_idx = st.slider("Sample Index", 0, len(X_test) - 1, 0)
        true_label = y_test[sample_idx]
        st.markdown(f"""
        <div class='card card-blue' style='margin-top:0.8rem;'>
            <strong>Ground Truth Class</strong><br>
            <span style='color:#0057B8; font-size:1rem; font-weight:600;'>{CLASS_LABELS[true_label]}</span>
        </div>
        """, unsafe_allow_html=True)

        predict_btn = st.button("🔍  Run Prediction", use_container_width=True)

        st.markdown("<div class='section-header'><h2>📉 ECG Waveform</h2></div>", unsafe_allow_html=True)
        signal = X_test[sample_idx]
        fig, ax = plt.subplots(figsize=(5, 2.4), facecolor="#f8faff")
        ax.set_facecolor("#f8faff")
        ax.plot(signal, color="#0057B8", linewidth=1.2)
        ax.set_xlabel("Sample", fontsize=8, color="#4a5568")
        ax.set_ylabel("Amplitude (mV)", fontsize=8, color="#4a5568")
        ax.set_title(f"Sample #{sample_idx}", fontsize=9, color="#1a1f36", fontweight="600")
        ax.grid(True, linestyle="--", alpha=0.4, color="#c8d9f5")
        ax.tick_params(labelsize=7, colors="#4a5568")
        for spine in ax.spines.values():
            spine.set_edgecolor("#c8d9f5")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close()

    with right:
        st.markdown("<div class='section-header'><h2>🩺 Prediction Result</h2></div>", unsafe_allow_html=True)

        if predict_btn:
            with st.spinner("Analysing ECG signal…"):
                inp = X_test_cnn[sample_idx:sample_idx+1]
                probs = model.predict(inp, verbose=0)[0]
                pred_class = int(np.argmax(probs))
                confidence = float(probs[pred_class]) * 100

            # persist to history
            st.session_state.pred_history.append({
                "Sample": sample_idx,
                "Predicted": CLASS_SHORT[pred_class],
                "True": CLASS_SHORT[true_label],
                "Confidence": f"{confidence:.1f}%",
                "Risk": CLASS_RISK[pred_class],
            })

            css_cls = CLASS_COLOR[pred_class]
            emoji   = CLASS_EMOJI[pred_class]
            risk    = CLASS_RISK[pred_class]
            risk_color = {"Low":"#0a9a60","Moderate":"#e67e22","High":"#D62828"}[risk]

            st.markdown(f"""
            <div class='pred-result {css_cls}'>
                <div style='font-size:2rem; margin-bottom:0.4rem;'>{emoji}</div>
                <div style='font-size:1.4rem; font-weight:700; color:#1a1f36;'>{CLASS_LABELS[pred_class]}</div>
                <div style='margin:0.6rem 0 0.3rem; font-size:0.88rem; color:#4a5568;'>
                    Confidence Score
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))
            st.markdown(f"**{confidence:.1f}%** confidence")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class='card' style='text-align:center; padding:0.9rem;'>
                    <div style='font-size:0.72rem; color:#5a6483; text-transform:uppercase; letter-spacing:0.06em;'>Risk Level</div>
                    <div style='font-size:1.3rem; font-weight:700; color:{risk_color};'>{risk}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                correct = "✅ Correct" if pred_class == true_label else "❌ Mismatch"
                st.markdown(f"""
                <div class='card' style='text-align:center; padding:0.9rem;'>
                    <div style='font-size:0.72rem; color:#5a6483; text-transform:uppercase; letter-spacing:0.06em;'>vs Ground Truth</div>
                    <div style='font-size:1rem; font-weight:600; color:#1a1f36;'>{correct}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='card card-blue'>
                <strong style='color:#0057B8;'>🩺 Clinical Interpretation</strong>
                <p style='margin:0.5rem 0 0; font-size:0.87rem; color:#1a1f36;'>{CLASS_INTERP[pred_class]}</p>
            </div>
            <div class='card card-green'>
                <strong style='color:#0a7a4a;'>📋 Clinical Recommendation</strong>
                <p style='margin:0.5rem 0 0; font-size:0.87rem; color:#1a1f36;'>{CLASS_REC[pred_class]}</p>
            </div>
            """, unsafe_allow_html=True)

            # Per-class probability bar chart
            st.markdown("<div class='section-header'><h2>📊 Class Probabilities</h2></div>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(5, 2.2), facecolor="#f8faff")
            ax2.set_facecolor("#f8faff")
            bar_colors = ["#0057B8" if i == pred_class else "#c8d9f5" for i in range(5)]
            ax2.barh(list(CLASS_SHORT.values()), probs * 100, color=bar_colors, height=0.55, edgecolor="none")
            ax2.set_xlim(0, 100)
            ax2.set_xlabel("Probability (%)", fontsize=8, color="#4a5568")
            ax2.tick_params(labelsize=8, colors="#4a5568")
            ax2.grid(axis="x", linestyle="--", alpha=0.4, color="#e0e8f4")
            for spine in ax2.spines.values():
                spine.set_edgecolor("#c8d9f5")
            plt.tight_layout(pad=0.5)
            st.pyplot(fig2)
            plt.close()

        else:
            st.markdown("""
            <div class='card' style='text-align:center; padding:2.5rem; color:#5a6483;'>
                <div style='font-size:3rem; margin-bottom:0.8rem;'>🔍</div>
                <strong>Awaiting Analysis</strong>
                <p style='font-size:0.85rem; margin-top:0.4rem;'>Select a sample and click <em>Run Prediction</em> to begin.</p>
            </div>
            """, unsafe_allow_html=True)

    # Prediction history
    if st.session_state.pred_history:
        st.markdown("<div class='section-header'><h2>📋 Prediction History</h2></div>", unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.pred_history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("🗑️  Clear History"):
            st.session_state.pred_history = []
            st.rerun()

    footer()

# ════════════════════════════════════════════
# PAGE: ECG VISUALIZATION
# ════════════════════════════════════════════
elif nav == "📈  ECG Visualization":
    st.markdown("""
    <div class='hero'>
        <div class='tag'>📈 Waveform Explorer</div>
        <h1>ECG Signal Visualization</h1>
        <p>Inspect individual ECG waveforms from the MIT-BIH test set with amplitude, time grid, and signal statistics.</p>
    </div>
    """, unsafe_allow_html=True)

    controls = st.columns([2, 1, 1])
    with controls[0]:
        vis_idx = st.slider("Select Sample", 0, len(X_test) - 1, 42)
    with controls[1]:
        show_grid = st.checkbox("Show Grid", value=True)
    with controls[2]:
        show_stats = st.checkbox("Show Statistics", value=True)

    signal = X_test[vis_idx]
    true_cls = y_test[vis_idx]
    t = np.arange(len(signal)) / 360  # MIT-BIH sampling rate = 360 Hz

    fig, ax = plt.subplots(figsize=(11, 3.5), facecolor="#f8faff")
    ax.set_facecolor("#f8faff")
    ax.plot(t, signal, color="#0057B8", linewidth=1.3, label="ECG Signal")
    ax.axhline(0, color="#D62828", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)", fontsize=10, color="#4a5568")
    ax.set_ylabel("Amplitude (mV)", fontsize=10, color="#4a5568")
    ax.set_title(f"ECG Waveform — Sample #{vis_idx} · Class: {CLASS_LABELS[true_cls]}", fontsize=11, fontweight="600", color="#1a1f36")
    if show_grid:
        ax.grid(True, linestyle="--", alpha=0.35, color="#c8d9f5")
    ax.tick_params(labelsize=9, colors="#4a5568")
    for spine in ax.spines.values():
        spine.set_edgecolor("#c8d9f5")
    plt.tight_layout(pad=0.6)
    st.pyplot(fig)
    plt.close()

    if show_stats:
        st.markdown("<div class='section-header'><h2>📐 Signal Statistics</h2></div>", unsafe_allow_html=True)
        stat_cols = st.columns(5)
        stats = [
            ("Max", f"{signal.max():.4f}"),
            ("Min", f"{signal.min():.4f}"),
            ("Mean", f"{signal.mean():.4f}"),
            ("Std Dev", f"{signal.std():.4f}"),
            ("Range", f"{(signal.max()-signal.min()):.4f}"),
        ]
        for col, (lbl, val) in zip(stat_cols, stats):
            with col:
                st.markdown(f"""
                <div class='metric-chip'>
                    <div class='value' style='font-size:1.2rem;'>{val}</div>
                    <div class='label'>{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

    # Overlay comparison — 5 samples, one per class
    st.markdown("<div class='section-header'><h2>🔀 Class Comparison — One Sample per Class</h2></div>", unsafe_allow_html=True)
    colors = ["#0a9a60","#0057B8","#D62828","#7b2d8b","#718096"]
    fig2, axes = plt.subplots(1, 5, figsize=(13, 2.8), facecolor="#f8faff", sharey=True)
    for c in range(5):
        idx = np.where(y_test == c)[0][0]
        axes[c].plot(X_test[idx], color=colors[c], linewidth=1.1)
        axes[c].set_title(CLASS_SHORT[c], fontsize=10, fontweight="700", color=colors[c])
        axes[c].set_facecolor("#f8faff")
        axes[c].grid(True, linestyle="--", alpha=0.3, color="#c8d9f5")
        axes[c].tick_params(labelsize=7, colors="#4a5568")
        for spine in axes[c].spines.values():
            spine.set_edgecolor("#c8d9f5")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig2)
    plt.close()

    footer()

# ════════════════════════════════════════════
# PAGE: DATASET INFORMATION
# ════════════════════════════════════════════
elif nav == "📊  Dataset Information":
    st.markdown("""
    <div class='hero'>
        <div class='tag'>📊 MIT-BIH Arrhythmia Database</div>
        <h1>Dataset Information</h1>
        <p>Overview of the training and testing data used to build and evaluate the ECGSense CNN model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'><h2>🗂️ Dataset Overview</h2></div>", unsafe_allow_html=True)
    d_cols = st.columns(4)
    dataset_metrics = [
        ("109,446", "Total Samples"),
        ("87,554", "Training Samples"),
        ("21,892", "Testing Samples"),
        ("360 Hz", "Sampling Rate"),
    ]
    for col, (v, l) in zip(d_cols, dataset_metrics):
        with col:
            st.markdown(f"""
            <div class='metric-chip'>
                <div class='value'>{v}</div>
                <div class='label'>{l}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Class distribution from test set
    st.markdown("<div class='section-header'><h2>📊 Heartbeat Class Distribution (Test Set)</h2></div>", unsafe_allow_html=True)
    class_counts = {CLASS_LABELS[i]: int(np.sum(y_test == i)) for i in range(5)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#f8faff")
    bar_colors = ["#0a9a60","#0057B8","#D62828","#7b2d8b","#718096"]

    # Bar chart
    ax1 = axes[0]
    ax1.set_facecolor("#f8faff")
    bars = ax1.bar(list(CLASS_SHORT.values()), list(class_counts.values()), color=bar_colors, edgecolor="none", width=0.55)
    ax1.set_xlabel("Heartbeat Class", fontsize=10, color="#4a5568")
    ax1.set_ylabel("Sample Count", fontsize=10, color="#4a5568")
    ax1.set_title("Distribution by Class", fontsize=11, fontweight="600", color="#1a1f36")
    ax1.grid(axis="y", linestyle="--", alpha=0.4, color="#c8d9f5")
    ax1.tick_params(labelsize=9, colors="#4a5568")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#c8d9f5")
    for bar, cnt in zip(bars, class_counts.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40, str(cnt),
                 ha="center", va="bottom", fontsize=8, color="#1a1f36", fontweight="600")

    # Pie chart
    ax2 = axes[1]
    ax2.set_facecolor("#f8faff")
    wedges, texts, autotexts = ax2.pie(
        list(class_counts.values()),
        labels=list(CLASS_SHORT.values()),
        colors=bar_colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.78,
        wedgeprops={"edgecolor": "#f8faff", "linewidth": 2},
    )
    for t in texts:
        t.set_fontsize(10); t.set_fontweight("600")
    for at in autotexts:
        at.set_fontsize(8); at.set_color("#fff")
    ax2.set_title("Proportion by Class", fontsize=11, fontweight="600", color="#1a1f36")

    plt.tight_layout(pad=1)
    st.pyplot(fig)
    plt.close()

    st.markdown("<div class='section-header'><h2>📋 Class Breakdown Table</h2></div>", unsafe_allow_html=True)
    class_table = pd.DataFrame({
        "Class Code": list(CLASS_SHORT.values()),
        "Class Name": list(CLASS_LABELS.values()),
        "Test Samples": list(class_counts.values()),
        "% Share": [f"{v/sum(class_counts.values())*100:.1f}%" for v in class_counts.values()],
    })
    st.dataframe(class_table, use_container_width=True, hide_index=True)

    footer()

# ════════════════════════════════════════════
# PAGE: HEARTBEAT CLASSES
# ════════════════════════════════════════════
elif nav == "❤️  Heartbeat Classes":
    st.markdown("""
    <div class='hero'>
        <div class='tag'>❤️ AAMI EC57 Classification</div>
        <h1>Heartbeat Classes</h1>
        <p>The five AAMI-standard heartbeat categories detected by ECGSense, with clinical descriptions.</p>
    </div>
    """, unsafe_allow_html=True)

    classes = [
        {
            "code": "N", "name": "Normal Beat", "css": "hb-normal",
            "desc": "Standard sinus beat arising from the SA node, conducting normally through the AV node and bundle branches.",
            "symptoms": "None — physiologically normal.",
            "significance": "Baseline reference for arrhythmia comparison.",
            "ecg": "Regular P–QRS–T morphology. Normal intervals. Narrow QRS (<0.12 s).",
        },
        {
            "code": "S", "name": "Supraventricular Beat", "css": "hb-supra",
            "desc": "Premature beat originating above the ventricles (atria or AV junction), with aberrant conduction possible.",
            "symptoms": "Palpitations, occasional dizziness, irregular pulse.",
            "significance": "Can indicate atrial fibrillation risk, electrolyte disturbance, or structural heart disease.",
            "ecg": "Early P wave (possibly inverted or absent). Narrow QRS if no aberrancy. Compensatory pause may follow.",
        },
        {
            "code": "V", "name": "Ventricular Beat", "css": "hb-ventri",
            "desc": "Premature ectopic beat originating within the ventricular myocardium, bypassing normal conduction pathways.",
            "symptoms": "Palpitations, chest discomfort, near-syncope in severe cases.",
            "significance": "High clinical priority. Frequent PVCs may indicate ischaemia or cardiomyopathy; risk of VT/VF.",
            "ecg": "Wide, bizarre QRS (>0.12 s). No preceding P wave. T wave opposite QRS polarity. Full compensatory pause.",
        },
        {
            "code": "F", "name": "Fusion Beat", "css": "hb-fusion",
            "desc": "Simultaneous activation of ventricles by both a sinus impulse and an ectopic ventricular impulse, producing a hybrid morphology.",
            "symptoms": "Usually asymptomatic; found incidentally.",
            "significance": "Confirms presence of ventricular ectopy. Useful diagnostic marker in Wolff-Parkinson-White and accelerated idioventricular rhythm.",
            "ecg": "QRS morphology intermediate between normal and PVC. Not preceded by a fully normal P wave.",
        },
        {
            "code": "Q", "name": "Unknown / Paced Beat", "css": "hb-unknown",
            "desc": "Beats that do not fit standard morphological criteria, including paced beats or heavily artefacted signals.",
            "symptoms": "Depends on underlying condition; paced beats may be entirely asymptomatic.",
            "significance": "Requires manual cardiologist review for definitive classification.",
            "ecg": "Variable morphology. Paced beats exhibit a pacemaker spike before the QRS.",
        },
    ]

    for cls in classes:
        with st.expander(f"{cls['code']} — {cls['name']}", expanded=False):
            st.markdown(f"""
            <div class='hb-card {cls["css"]}'>
                <div class='badge'>{cls["code"]}</div>
                <h3>{cls["name"]}</h3>
                <p>{cls["desc"]}</p>
            </div>
            """, unsafe_allow_html=True)
            detail_cols = st.columns(3)
            with detail_cols[0]:
                st.markdown(f"""
                <div class='card' style='height:100%;'>
                    <strong>🩺 Symptoms</strong>
                    <p style='font-size:0.85rem; color:#4a5568; margin-top:0.4rem;'>{cls["symptoms"]}</p>
                </div>
                """, unsafe_allow_html=True)
            with detail_cols[1]:
                st.markdown(f"""
                <div class='card card-red' style='height:100%;'>
                    <strong>⚕️ Medical Significance</strong>
                    <p style='font-size:0.85rem; color:#4a5568; margin-top:0.4rem;'>{cls["significance"]}</p>
                </div>
                """, unsafe_allow_html=True)
            with detail_cols[2]:
                st.markdown(f"""
                <div class='card card-blue' style='height:100%;'>
                    <strong>📡 ECG Characteristics</strong>
                    <p style='font-size:0.85rem; color:#4a5568; margin-top:0.4rem;'>{cls["ecg"]}</p>
                </div>
                """, unsafe_allow_html=True)

    footer()

# ════════════════════════════════════════════
# PAGE: CNN ARCHITECTURE
# ════════════════════════════════════════════
elif nav == "🧠  CNN Architecture":
    st.markdown("""
    <div class='hero'>
        <div class='tag'>🧠 Deep Learning Model</div>
        <h1>CNN Architecture</h1>
        <p>1D Convolutional Neural Network designed for time-series ECG classification across five AAMI beat categories.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'><h2>🏗️ Layer-by-Layer Breakdown</h2></div>", unsafe_allow_html=True)

    layers = [
        ("IN", "#4a5568", "Input Layer", "Shape: (187, 1) — single-lead ECG segment at 360 Hz, 0.52 s window"),
        ("C1", "#0057B8", "Conv1D (64 filters, kernel=3, ReLU)", "Extracts local morphological features (QRS slopes, P wave contours)"),
        ("C2", "#0057B8", "Conv1D (64 filters, kernel=3, ReLU)", "Deepens feature hierarchy; captures beat-level temporal patterns"),
        ("P1", "#0a9a60", "MaxPooling1D (pool=2)", "Reduces sequence length; retains dominant activations"),
        ("DR", "#7b2d8b", "Dropout (0.3)", "Regularisation — prevents co-adaptation of neurons"),
        ("C3", "#0082f4", "Conv1D (128 filters, kernel=3, ReLU)", "High-level feature extraction over pooled representations"),
        ("P2", "#0a9a60", "MaxPooling1D (pool=2)", "Further spatial compression"),
        ("FL", "#e67e22", "Flatten", "Converts 2D feature maps to 1D vector for dense layers"),
        ("D1", "#D62828", "Dense (64, ReLU)", "Non-linear classification head; learns class-discriminative mappings"),
        ("DR2","#7b2d8b", "Dropout (0.3)", "Additional regularisation before output"),
        ("OU", "#1a1f36", "Dense (5, Softmax)", "Output: probability distribution across 5 heartbeat classes"),
    ]

    for code, color, name, desc in layers:
        st.markdown(f"""
        <div class='arch-layer'>
            <div class='layer-icon' style='background:{color};'>{code}</div>
            <div>
                <div style='font-weight:600; font-size:0.92rem;'>{name}</div>
                <div style='font-size:0.78rem; color:#5a6483; margin-top:2px;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'><h2>📐 Architecture Diagram</h2></div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(13, 3.5), facecolor="#f8faff")
    ax.set_facecolor("#f8faff")
    ax.set_xlim(0, 13); ax.set_ylim(0, 4); ax.axis("off")

    layer_defs = [
        (0.5,  "Input\n(187,1)",  "#4a5568", 0.7, 2.5),
        (1.8,  "Conv1D\n64 k=3",  "#0057B8", 0.7, 2.0),
        (3.1,  "Conv1D\n64 k=3",  "#0057B8", 0.7, 2.0),
        (4.4,  "MaxPool\n/2",     "#0a9a60", 0.7, 1.6),
        (5.5,  "Dropout\n0.3",    "#7b2d8b", 0.7, 1.3),
        (6.6,  "Conv1D\n128 k=3", "#0082f4", 0.7, 2.3),
        (7.9,  "MaxPool\n/2",     "#0a9a60", 0.7, 1.8),
        (9.0,  "Flatten",         "#e67e22", 0.7, 1.0),
        (10.1, "Dense\n64 ReLU",  "#D62828", 0.7, 1.5),
        (11.2, "Softmax\nOutput", "#1a1f36", 0.7, 1.0),
    ]
    prev_x = None
    for (x, label, color, w, h) in layer_defs:
        rect = mpatches.FancyBboxPatch(
            (x, (4-h)/2), w, h,
            boxstyle="round,pad=0.08",
            linewidth=1.5, edgecolor="#fff",
            facecolor=color, alpha=0.9, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x + w/2, 2, label, ha="center", va="center",
                fontsize=7, color="#fff", fontweight="600", zorder=4, linespacing=1.4)
        if prev_x is not None:
            ax.annotate("", xy=(x, 2), xytext=(prev_x, 2),
                        arrowprops=dict(arrowstyle="->", color="#c8d9f5", lw=1.5), zorder=2)
        prev_x = x + w

    plt.tight_layout(pad=0.4)
    st.pyplot(fig)
    plt.close()

    # Model params summary
    st.markdown("<div class='section-header'><h2>🔢 Training Configuration</h2></div>", unsafe_allow_html=True)
    config_cols = st.columns(4)
    configs = [
        ("Optimizer", "Adam"),
        ("Loss Function", "Categorical CE"),
        ("Epochs", "~30–50"),
        ("Batch Size", "32"),
    ]
    for col, (k, v) in zip(config_cols, configs):
        with col:
            st.markdown(f"""
            <div class='metric-chip'>
                <div class='value' style='font-size:1rem;'>{v}</div>
                <div class='label'>{k}</div>
            </div>
            """, unsafe_allow_html=True)

    footer()

# ════════════════════════════════════════════
# PAGE: ABOUT ECG
# ════════════════════════════════════════════
elif nav == "📖  About ECG":
    st.markdown("""
    <div class='hero'>
        <div class='tag'>📖 Cardiology Primer</div>
        <h1>About Electrocardiography</h1>
        <p>A clinical overview of the ECG, its key waveform components, common arrhythmias, and the role of AI in cardiology.</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("<div class='section-header'><h2>⚡ What is an ECG?</h2></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card card-blue'>
            <p style='font-size:0.9rem; color:#1a1f36; line-height:1.75;'>
            An <strong>electrocardiogram (ECG/EKG)</strong> is a non-invasive recording of the 
            heart's electrical activity over time, captured via surface electrodes. 
            Each heartbeat generates a characteristic waveform pattern whose shape, 
            duration, and amplitude reveal the health of the myocardium and its conduction system.
            The standard 12-lead ECG is the cornerstone of cardiovascular diagnostics.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'><h2>🌊 Waveform Components</h2></div>", unsafe_allow_html=True)
        waves = [
            ("P Wave", "#0a9a60", "Atrial depolarisation. Duration: 80–100 ms. Amplitude: <0.25 mV. Absence may indicate atrial fibrillation."),
            ("PR Interval", "#0057B8", "AV conduction time. Normal: 120–200 ms. Prolonged PR indicates first-degree AV block."),
            ("QRS Complex", "#D62828", "Ventricular depolarisation. Normal: <120 ms. Wide QRS (≥120 ms) suggests bundle branch block or ventricular origin."),
            ("ST Segment", "#e67e22", "Ventricular plateau. Elevation ≥1 mm in 2 contiguous leads indicates STEMI until proven otherwise."),
            ("T Wave", "#7b2d8b", "Ventricular repolarisation. Inversion may indicate ischaemia, LVH, or electrolyte disturbance."),
            ("QT Interval", "#4a5568", "Total ventricular electrical activity. Corrected QTc >450 ms (men) / >470 ms (women) indicates Long QT Syndrome risk."),
        ]
        for name, color, desc in waves:
            st.markdown(f"""
            <div class='card' style='border-left:4px solid {color}; padding:0.9rem 1.2rem; margin-bottom:0.6rem;'>
                <strong style='color:{color};'>{name}</strong>
                <p style='font-size:0.84rem; color:#4a5568; margin:0.3rem 0 0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-header'><h2>💓 ECG Waveform Illustration</h2></div>", unsafe_allow_html=True)

        # Synthesise a representative PQRST complex
        t_syn = np.linspace(0, 1, 500)
        def gaussian(t, mu, sigma, amp): return amp * np.exp(-((t - mu)**2) / (2 * sigma**2))
        ecg_syn = (gaussian(t_syn, 0.15, 0.025, 0.25) +          # P
                   gaussian(t_syn, 0.40, 0.008, -0.10) +          # Q
                   gaussian(t_syn, 0.43, 0.010, 1.20) +           # R
                   gaussian(t_syn, 0.46, 0.008, -0.20) +          # S
                   gaussian(t_syn, 0.65, 0.04,  0.35))             # T

        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#f8faff")
        ax.set_facecolor("#f8faff")
        ax.plot(t_syn, ecg_syn, color="#D62828", linewidth=2)
        ax.axhline(0, color="#c8d9f5", linewidth=0.8)
        ax.grid(True, linestyle="--", alpha=0.3, color="#c8d9f5")

        annotations = [
            (0.15, 0.3, "P"),
            (0.43, 1.28, "R"),
            (0.46, -0.28, "S"),
            (0.65, 0.42, "T"),
            (0.40, -0.18, "Q"),
        ]
        for tx, ty, lbl in annotations:
            ax.annotate(lbl, (tx, ty), fontsize=11, fontweight="700",
                        color="#0057B8", ha="center")

        ax.set_xlabel("Time (normalised)", fontsize=9, color="#4a5568")
        ax.set_ylabel("Amplitude (mV)", fontsize=9, color="#4a5568")
        ax.set_title("Typical PQRST Complex", fontsize=10, fontweight="600", color="#1a1f36")
        ax.tick_params(labelsize=8, colors="#4a5568")
        for spine in ax.spines.values(): spine.set_edgecolor("#c8d9f5")
        plt.tight_layout(pad=0.6)
        st.pyplot(fig)
        plt.close()

        st.markdown("<div class='section-header'><h2>🤖 AI in Cardiology</h2></div>", unsafe_allow_html=True)
        ai_points = [
            ("🚀", "Scale", "AI processes thousands of ECGs per hour — enabling population-level screening infeasible manually."),
            ("🎯", "Consistency", "Deep learning models apply the same classification criteria uniformly, eliminating inter-observer variability."),
            ("⚡", "Speed", "Sub-second inference enables real-time arrhythmia alerts in wearables and bedside monitors."),
            ("🌍", "Access", "AI-assisted diagnosis extends specialist-level cardiology to resource-limited settings globally."),
        ]
        for icon, title, desc in ai_points:
            st.markdown(f"""
            <div class='card' style='padding:0.8rem 1rem; margin-bottom:0.5rem;'>
                <span style='font-size:1.2rem;'>{icon}</span>
                <strong style='margin-left:8px; color:#0057B8;'>{title}</strong>
                <p style='font-size:0.83rem; color:#4a5568; margin:0.3rem 0 0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    footer()

# ════════════════════════════════════════════
# PAGE: ABOUT PROJECT
# ════════════════════════════════════════════
elif nav == "ℹ️  About Project":
    st.markdown("""
    <div class='hero'>
        <div class='tag'>ℹ️ Project Documentation</div>
        <h1>AI-Assisted Multi-Class ECG Heartbeat Classification</h1>
        <p>Final-year minor project — Biomedical Engineering with AI/ML specialisation, Osmania University.</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Problem & Objectives", "Methodology", "Technologies", "Applications & Scope"])

    with tabs[0]:
        st.markdown("""
        <div class='card card-red'>
            <strong style='color:#D62828;'>🔴 Problem Statement</strong>
            <p style='font-size:0.9rem; color:#1a1f36; margin-top:0.6rem; line-height:1.75;'>
            Cardiovascular disease is the leading cause of mortality globally, responsible for
            approximately 17.9 million deaths annually (WHO, 2021). Arrhythmias — disorders of
            cardiac rhythm — are frequently under-detected due to the high volume of ECG data
            generated in clinical settings and the shortage of specialist cardiologists,
            particularly in low- and middle-income countries. Manual ECG interpretation is
            time-consuming, subject to fatigue, and inconsistent across practitioners.
            </p>
            <p style='font-size:0.9rem; color:#1a1f36; line-height:1.75; margin-bottom:0;'>
            There is a critical need for an automated, real-time, explainable system that can 
            accurately classify heartbeat morphology and provide structured clinical guidance 
            to support — not replace — cardiologist decision-making.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'><h2>🎯 Objectives</h2></div>", unsafe_allow_html=True)
        objectives = [
            "Develop a 1D CNN classifier for five-class heartbeat categorisation on the MIT-BIH dataset.",
            "Achieve clinically meaningful accuracy (>95%) with balanced performance across minority classes.",
            "Build a hospital-grade CDSS interface with waveform visualisation and structured interpretation.",
            "Provide per-prediction confidence scores, risk stratification, and clinical recommendations.",
            "Document model architecture, dataset characteristics, and evaluation metrics transparently.",
        ]
        for i, obj in enumerate(objectives, 1):
            st.markdown(f"""
            <div class='card' style='display:flex; align-items:flex-start; gap:12px; padding:0.9rem 1.2rem;'>
                <div style='background:#0057B8; color:#fff; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.82rem; flex-shrink:0;'>{i}</div>
                <p style='margin:0; font-size:0.88rem; color:#1a1f36; padding-top:3px;'>{obj}</p>
            </div>
            """, unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("<div class='section-header'><h2>🔁 Workflow</h2></div>", unsafe_allow_html=True)
        steps = [
            ("📥", "Data Acquisition", "MIT-BIH Arrhythmia Database via PhysioNet. 48 half-hour dual-channel ambulatory ECG recordings, pre-segmented into 187-sample beat windows."),
            ("🧹", "Pre-processing", "Z-score normalisation per segment. SMOTE oversampling applied to training set to address severe class imbalance (N class dominates at >82%)."),
            ("🧠", "Model Training", "1D CNN trained with Adam optimiser, categorical cross-entropy loss, early stopping, and learning-rate reduction on plateau."),
            ("📊", "Evaluation", "Test set evaluation on 21,892 samples. Metrics: accuracy, per-class precision, recall, F1-score, confusion matrix."),
            ("🖥️", "Deployment", "Streamlit web application with TensorFlow model serving, interactive waveform visualisation, and structured clinical output."),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div class='card card-blue' style='display:flex; align-items:flex-start; gap:14px;'>
                <div style='font-size:1.6rem; flex-shrink:0;'>{icon}</div>
                <div>
                    <strong style='color:#0057B8;'>{title}</strong>
                    <p style='font-size:0.86rem; color:#4a5568; margin:0.3rem 0 0;'>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tabs[2]:
        tech_cols = st.columns(2)
        techs = [
            ("🐍", "Python 3.10+", "Primary programming language"),
            ("🧠", "TensorFlow / Keras", "CNN model training and inference"),
            ("🎈", "Streamlit", "Clinical web application framework"),
            ("🐼", "Pandas / NumPy", "Data processing and manipulation"),
            ("📊", "Matplotlib", "ECG waveform and statistical visualisation"),
            ("🗄️", "MIT-BIH DB", "Benchmark arrhythmia dataset (PhysioNet)"),
        ]
        for i, (icon, name, desc) in enumerate(techs):
            with tech_cols[i % 2]:
                st.markdown(f"""
                <div class='card' style='display:flex; align-items:center; gap:12px; padding:0.9rem 1.2rem; margin-bottom:0.6rem;'>
                    <div style='font-size:1.8rem;'>{icon}</div>
                    <div>
                        <strong>{name}</strong>
                        <p style='font-size:0.8rem; color:#5a6483; margin:0;'>{desc}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tabs[3]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='section-header'><h2>🏥 Applications</h2></div>", unsafe_allow_html=True)
            apps = ["Bedside cardiac monitoring in ICU/CCU", "Remote patient monitoring and wearable ECG devices",
                    "Population-level cardiac screening programmes", "Pre-operative cardiac risk assessment",
                    "Medical education and ECG training simulators"]
            for a in apps:
                st.markdown(f"<div class='card card-green' style='padding:0.7rem 1rem; margin-bottom:0.5rem; font-size:0.86rem;'>✅ {a}</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='section-header'><h2>🔭 Future Scope</h2></div>", unsafe_allow_html=True)
            futures = ["12-lead multi-channel ECG classification (current: single lead)",
                       "Transformer/attention-based architectures for improved minority class detection",
                       "Federated learning for privacy-preserving multi-hospital training",
                       "Integration with HL7 FHIR electronic health record standards",
                       "Real-time streaming inference from wearable ECG sensors"]
            for f in futures:
                st.markdown(f"<div class='card card-blue' style='padding:0.7rem 1rem; margin-bottom:0.5rem; font-size:0.86rem;'>🔭 {f}</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='card card-orange' style='margin-top:1rem;'>
            <strong>⚠️ Limitations</strong>
            <ul style='font-size:0.86rem; color:#4a5568; margin:0.5rem 0 0; padding-left:1.2rem; line-height:1.8;'>
                <li>Trained and evaluated on a single benchmark dataset (MIT-BIH); generalisation to other ECG recorders requires further validation.</li>
                <li>Single-lead classification — does not capture spatial cardiac electrical patterns of 12-lead ECG.</li>
                <li>Class imbalance in minority classes (S, F, Q) despite SMOTE; real-world performance may differ.</li>
                <li>Not validated in a clinical trial setting. Cannot replace regulatory-approved medical devices.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    footer()