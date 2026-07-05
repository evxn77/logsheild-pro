import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import base64
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import hashlib

# -------------------------
# Page config & styles
# -------------------------
st.set_page_config(
    page_title="LogShield Pro – Breach Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Animated Gradient Text Title */
.main-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(-45deg, #0f4c81, #0275d8, #5a9bd8, #0f4c81);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientBG 15s ease infinite;
    text-align: center;
    margin-bottom: 10px;
}
@keyframes gradientBG {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}
/* Subheader style */
.sub-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1f2937;
    border-bottom: 3px solid #0f4c81;
    padding-bottom: 8px;
    margin-bottom: 20px;
}
/* Uniform overview metric cards - SAME SIZE */
.overview-metric-card {
    background: linear-gradient(145deg, #f8fafc, #e2e8f0);
    padding: 1.2rem;
    border-radius: 12px;
    border-left: 4px solid #0f4c81;
    box-shadow: 0 4px 12px rgba(15, 76, 129, 0.12);
    transition: all 0.3s ease;
    text-align: center;
    height: 110px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    min-width: 180px;
}
.overview-metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(15, 76, 129, 0.2);
}
.metric-title {
    font-weight: 600;
    color: #475569;
    margin: 0 0 8px 0;
    font-size: 0.95rem;
    line-height: 1.2;
}
.metric-value {
    font-weight: 800;
    font-size: 1.6rem;
    margin: 0;
    color: #1e293b;
}
/* Core metric cards */
.metric-card {
    background: linear-gradient(145deg, #f8fafc, #e2e8f0);
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 6px solid #0f4c81;
    box-shadow: 0 6px 16px rgba(15, 76, 129, 0.15);
    transition: all 0.3s ease;
    text-align: center;
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 30px rgba(15, 76, 129, 0.25);
}
/* Progress bar */
.progress-container {
    width: 100%;
    background-color: #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 8px;
    height: 12px;
}
.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #dc3545, #ffc107);
    animation: grow 2s ease forwards;
    border-radius: 10px;
}
@keyframes grow {
    0% {width: 0;}
    100% {width: var(--progress);}
}
/* Attack/Normal percentage containers */
.attack-normal-container {
    display: flex;
    gap: 2rem;
    justify-content: center;
    margin: 2rem 0;
}
.attack-box, .normal-box {
    padding: 1.2rem 1.8rem;
    border-radius: 12px;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    min-width: 140px;
}
.attack-box {
    background: linear-gradient(145deg, #fee2e2, #fecaca);
    border-left: 5px solid #dc3545;
}
.normal-box {
    background: linear-gradient(145deg, #dcfce7, #bbf7d0);
    border-left: 5px solid #16a34a;
}
.attack-box:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(220, 53, 69, 0.3);
}
.normal-box:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(22, 163, 74, 0.3);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">LogShield Pro 🛡️ Breach Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#475569; font-size:1.1rem; margin-bottom: 40px;">A real-time cybersecurity traffic breach scoring and visualization suite</p>', unsafe_allow_html=True)

# -------------------------
# [All functions remain as in your original logic + DNA additions]
# -------------------------

def detect_columns(df):
    cols = [c.lower().strip().replace(" ","_") for c in df.columns]
    def pick(*keywords):
        for k in keywords:
            for i, c in enumerate(cols):
                if k in c:
                    return df.columns[i]
        return None
    timestamp_col = pick("time", "timestamp", "date", "datetime", "log_time", "event_time")
    user_col = pick("user", "device", "host", "ip", "src_ip", "dst_ip", "id")
    attack_col = pick("attack", "label", "target", "intrusion", "malware", "breach", "attack_cat")
    return timestamp_col, user_col, attack_col

def build_features(df):
    X = df.astype(str).sum(axis=1).str.len().values.reshape(-1, 1)
    return X

def train_model(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(Xs, y)
    joblib.dump(clf, "rf_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    return clf, scaler

def load_model():
    try:
        clf = joblib.load("rf_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return clf, scaler
    except:
        return None, None

def compute_metrics(df, timestamp_col, id_col):
    metrics = {}
    if "attack_label" in df.columns and "breach_score" in df.columns:
        metrics["Total Attacks"] = int(df["attack_label"].sum())
        metrics["Average Breach Score (%)"] = round(df["breach_score"].mean() * 100, 2)
    return metrics

def compute_simple_metrics(df, timestamp_col, id_col):
    m = {}
    total_conns = len(df)
    m["Total Connections"] = int(total_conns)

    if "attack_label" in df.columns:
        total_attacks = int(df["attack_label"].sum())
        total_normal = int(total_conns - total_attacks)
        attack_pct = round((total_attacks / total_conns)*100, 2) if total_conns > 0 else 0.0
        normal_pct = round(100 - attack_pct, 2)
        m["Total Attacks (All)"] = total_attacks
        m["Total Normal Connections"] = total_normal
        m["Attack Percentage (%)"] = attack_pct
        m["Normal Percentage (%)"] = normal_pct

    if {"sbytes", "dbytes"}.issubset(df.columns):
        total_bytes = (df["sbytes"] + df["dbytes"]).sum()
        avg_bytes = (df["sbytes"] + df["dbytes"]).mean() if total_conns > 0 else 0.0
        m["Total Data Transferred (bytes)"] = int(total_bytes)
        m["Avg Data per Connection (bytes)"] = round(avg_bytes, 2)

    if {"spkts", "dpkts"}.issubset(df.columns):
        total_pkts = (df["spkts"] + df["dpkts"]).sum()
        m["Total Packets"] = int(total_pkts)

    if "dur" in df.columns:
        m["Avg Duration (s)"] = round(df["dur"].mean(), 3)
        m["Shortest Duration (s)"] = round(df["dur"].min(), 3)
        m["Longest Duration (s)"] = round(df["dur"].max(), 3)

    if id_col and id_col in df.columns:
        if not df[id_col].dropna().empty:
            m["Most Active ID"] = df[id_col].mode()[0]

    if "attack_cat" in df.columns:
        attack_only = df[df["attack_label"] == 1] if "attack_label" in df.columns else df
        if not attack_only.empty:
            top_cat = attack_only["attack_cat"].mode()[0]
            m["Top Attack Category"] = str(top_cat)

    if timestamp_col and timestamp_col in df.columns and "attack_label" in df.columns:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        tmp = df.copy()
        tmp["ts"] = ts
        tmp = tmp.dropna(subset=["ts"])
        if not tmp.empty:
            hourly = tmp.groupby(tmp["ts"].dt.floor("H"))["attack_label"].sum().reset_index()
            if not hourly.empty:
                avg_aph = hourly["attack_label"].mean()
                m["Avg Attacks per Hour"] = round(avg_aph, 2)

    return m

def download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="breach_predictions.csv" style="font-weight:bold; color:#0f4c81;">📥 Download CSV with Scores</a>'

# -------------------------
# DNA generation helpers (NEW, appended without changing rest)
# -------------------------
def score_to_base(score):
    # score: 0..1 (if NaN, treat as 0)
    try:
        s = float(score)
    except:
        s = 0.0
    if s < 0.25:
        return "A"
    elif s < 0.50:
        return "C"
    elif s < 0.75:
        return "G"
    else:
        return "T"

HEX_BASE_MAP = {
    "0":"A","1":"C","2":"G","3":"T",
    "4":"A","5":"C","6":"G","7":"T",
    "8":"A","9":"C","a":"G","b":"T",
    "c":"A","d":"C","e":"G","f":"T"
}

def row_to_dna(row, length=32):
    """
    Deterministic DNA sequence per row:
    - Start with base derived from breach_score
    - Append bases by hashing concatenated row text and mapping hex->base
    """
    # if breach_score missing, default 0.0
    base = score_to_base(row.get("breach_score", 0.0))
    # build row text deterministically
    row_text_items = []
    # preserve column order
    for k in row.index:
        v = row[k]
        # convert timestamps to str without timezone issues
        if pd.api.types.is_datetime64_any_dtype(type(v)) or isinstance(v, (pd.Timestamp,)):
            row_text_items.append(str(v))
        else:
            row_text_items.append(str(v))
    row_text = "|".join(row_text_items)
    h = hashlib.sha256(row_text.encode()).hexdigest()
    dna = base
    for ch in h:
        dna += HEX_BASE_MAP.get(ch.lower(), "A")
        if len(dna) >= length:
            break
    # ensure exact length
    return dna[:length]

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload CSV logs", type=["csv"])
    train_flag = st.checkbox("Train model (needs label column)", value=False)
    st.markdown("""
    <small style="color:#64748b;">
    Upload a network flow log CSV including columns for labels and features.<br>
    The app will auto-detect key columns.
    </small>
    """, unsafe_allow_html=True)

# -------------------------
# Main app logic
# -------------------------
if uploaded:
    with st.spinner("Loading dataset and analyzing..."):
        df = pd.read_csv(uploaded)

    st.markdown('<h2 class="sub-header">📋 Detected Columns</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([3,2])
    with col1:
        st.write(df.columns.tolist())
    with col2:
        timestamp_col, user_col, attack_col = detect_columns(df)
        st.info(f"Timestamp: **{timestamp_col}**\nID Column: **{user_col}**\nLabel Column: **{attack_col}**")

    if not attack_col:
        st.error("❌ No label/attack column detected. Please upload a dataset with that column.")
    else:
        # create attack_label binary: treat 'normal' (case-insensitive) as 0 else 1
        df["attack_label"] = df[attack_col].apply(lambda x: 0 if str(x).strip().lower()=="normal" else 1)

        # build features & labels
        X = build_features(df)
        y = df["attack_label"].values

        clf, scaler = load_model()

        if train_flag:
            with st.spinner("Training model..."):
                clf, scaler = train_model(X, y)
                st.success("✅ Model trained and saved!")

        # predict breach_score safely
        if clf is not None and scaler is not None:
            Xs = scaler.transform(X)
            probs = clf.predict_proba(Xs)
            if probs.shape[1] == 1:
                # single-class trained model, handle gracefully
                if clf.classes_[0] == 0:
                    df["breach_score"] = probs[:, 0].astype(float)
                else:
                    df["breach_score"] = (1.0 - probs[:, 0]).astype(float)
            else:
                df["breach_score"] = probs[:, 1].astype(float)
            df["breach_score_pct"] = (df["breach_score"] * 100).round(2)
        else:
            df["breach_score"] = np.nan
            df["breach_score_pct"] = np.nan
            st.warning("No trained model available. Train one to get breach scores!")

        # -------------------------
        # DNA generation (ADDED)
        # -------------------------
        # generate dna_vector column deterministically for each row
        try:
            # apply row_to_dna using DataFrame row order
            df["dna_vector"] = df.apply(lambda r: row_to_dna(r, length=32), axis=1)
        except Exception as e:
            # fallback: fill with simple repeated base if something goes wrong
            df["dna_vector"] = df["breach_score"].apply(lambda s: (score_to_base(s if not pd.isna(s) else 0.0) * 32)[:32])

        breach_metrics = compute_metrics(df, timestamp_col, user_col)
        simple_metrics = compute_simple_metrics(df, timestamp_col, user_col)

        # -------------------------
        # Core Breach Metrics
        # -------------------------
        st.markdown('<h2 class="sub-header">🎯 Core Breach Metrics</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns([2,3])

        with col1:
            st.markdown(f"""
                <div class="metric-card">
                <h2 style="font-size:2.8rem; margin:0; color:#dc3545;">{breach_metrics.get('Total Attacks', 0)}</h2>
                <p style="font-weight:600; color:#475569; margin:0;">Total Attacks</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            score = breach_metrics.get("Average Breach Score (%)", 0)
            progress_width = f"{min(score, 100)}%"
            st.markdown(
                f"""
                <div class="metric-card">
                    <h2 style="font-size:2.8rem; margin:0; color:#0f4c81;">{score:.1f}%</h2>
                    <p style="font-weight:600; color:#475569; margin:0;">Average Breach Score</p>
                    <div class="progress-container">
                        <div class="progress-bar" style="--progress:{progress_width};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

        # -------------------------
        # Overview Metrics - SAME SIZE BOXES
        # -------------------------
        st.markdown('<h2 class="sub-header">📊 Overview Metrics</h2>', unsafe_allow_html=True)
        
        # 2x4 grid of uniform cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="overview-metric-card">
                    <p class="metric-title">Total Connections</p>
                    <p class="metric-value">{simple_metrics.get('Total Connections', 0):,}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class="overview-metric-card">
                    <p class="metric-title">Total Attacks</p>
                    <p class="metric-value">{simple_metrics.get('Total Attacks (All)', 0):,}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
                <div class="overview-metric-card">
                    <p class="metric-title">Normal Connections</p>
                    <p class="metric-value">{simple_metrics.get('Total Normal Connections', 0):,}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
                <div class="overview-metric-card">
                    <p class="metric-title">Total Packets</p>
                    <p class="metric-value">{simple_metrics.get('Total Packets', 0):,}</p>
                </div>
            """, unsafe_allow_html=True)
            
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.markdown(f"""
                <div class="overview-metric-card">
                    <p class="metric-title">Avg Duration (s)</p>
                    <p class="metric-value">{simple_metrics.get('Avg Duration (s)', 0):.3f}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col6:
            st.markdown(f"""
                <div class="overview-metric-card">
                    <p class="metric-title">Avg Data/Conn (bytes)</p>
                    <p class="metric-value">{simple_metrics.get('Avg Data per Connection (bytes)', 0):,.0f}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col7:
            st.markdown(f"""
                <div class="overview-metric-card">
                    <p class="metric-title">Top Attack Type</p>
                    <p class="metric-value">{simple_metrics.get('Top Attack Category', '-')}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col8:
            st.markdown(f"""
                <div class="overview-metric-card">
                    <p class="metric-title">Most Active ID</p>
                    <p class="metric-value">{simple_metrics.get('Most Active ID', '-')}</p>
                </div>
            """, unsafe_allow_html=True)

        # -------------------------
        # Attack vs Normal Percentage - Side by Side Boxes
        # -------------------------
        st.markdown('<h2 class="sub-header">⚖️ Attack vs Normal Distribution</h2>', unsafe_allow_html=True)
        col_a, col_n = st.columns(2)
        
        with col_a:
            st.markdown(f"""
                <div class="attack-box">
                    <h2 style="font-size:2.2rem; margin:0 0 5px 0; color:#dc3545;">{simple_metrics.get('Attack Percentage (%)', 0)}%</h2>
                    <p style="font-weight:600; color:#b91c1c; margin:0;">ATTACKS</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col_n:
            st.markdown(f"""
                <div class="normal-box">
                    <h2 style="font-size:2.2rem; margin:0 0 5px 0; color:#16a34a;">{simple_metrics.get('Normal Percentage (%)', 0)}%</h2>
                    <p style="font-weight:600; color:#166534; margin:0;">NORMAL</p>
                </div>
            """, unsafe_allow_html=True)

        # -------------------------
        # Pie chart
        # -------------------------
        labels = ['Attacks', 'Normal']
        sizes = [simple_metrics.get("Total Attacks (All)", 0), simple_metrics.get("Total Normal Connections", 0)]
        colors = ['#dc3545', '#0f4c81']

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})
        ax.set_title('Attack vs Normal Connections Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.axis('equal')
        st.pyplot(fig)

        # -------------------------
        # Detailed Analysis
        # -------------------------
        with st.expander("🔍 Detailed Analysis and Predictions", expanded=False):
            st.markdown('<h3 class="sub-header">Predictions Table (First 20 rows)</h3>', unsafe_allow_html=True)
            display_cols = list(df.columns)
            # ensure breach_score_pct is placed after breach_score if it exists
            if "breach_score_pct" in display_cols:
                display_cols = [c for c in display_cols if c != "breach_score_pct"]
                if "breach_score" in display_cols:
                    insert_pos = display_cols.index("breach_score") + 1
                else:
                    insert_pos = len(display_cols)
                display_cols.insert(insert_pos, "breach_score_pct")
            # Insert dna_vector into display columns (NEW)
            if "dna_vector" not in display_cols:
                # place dna_vector near end
                display_cols.append("dna_vector")

            st.dataframe(df[display_cols].head(20), height=350)

            if timestamp_col and timestamp_col in df.columns:
                df["ts"] = pd.to_datetime(df[timestamp_col], errors="coerce")

                if "attack_label" in df.columns:
                    df_timeline = df.groupby(df["ts"].dt.floor("H"))["attack_label"].sum().reset_index()
                    fig1, ax1 = plt.subplots(figsize=(12, 4), facecolor='white')
                    ax1.plot(df_timeline["ts"], df_timeline["attack_label"], marker="o", linewidth=3, markersize=8, color="#dc3545")
                    ax1.fill_between(df_timeline["ts"], df_timeline["attack_label"], alpha=0.3, color="#dc3545")
                    ax1.set_title("🚨 Attacks Over Time (Hourly)", fontsize=16, fontweight='bold')
                    ax1.set_xlabel("Time", fontsize=12)
                    ax1.set_ylabel("Attack Count", fontsize=12)
                    plt.xticks(rotation=45)
                    st.pyplot(fig1)

                if "breach_score" in df.columns:
                    df_score_timeline = df.groupby(df["ts"].dt.floor("H"))["breach_score"].mean().reset_index()
                    df_score_timeline["breach_score_pct"] = df_score_timeline["breach_score"] * 100

                    fig2, ax2 = plt.subplots(figsize=(12, 4), facecolor='white')
                    ax2.plot(df_score_timeline["ts"], df_score_timeline["breach_score_pct"], marker="o", linewidth=3, markersize=8, color="#0275d8")
                    ax2.set_title("📈 Average Breach Score Over Time (%)", fontsize=16, fontweight='bold')
                    ax2.set_xlabel("Time", fontsize=12)
                    ax2.set_ylabel("Breach Score (%)", fontsize=12)
                    ax2.yaxis.set_major_formatter(PercentFormatter(100))
                    plt.xticks(rotation=45)
                    st.pyplot(fig2)

        st.markdown(f"### {download_link(df)}", unsafe_allow_html=True)

else:
    st.info("Upload a CSV network intrusion detection dataset on the left sidebar to start analysis.")
