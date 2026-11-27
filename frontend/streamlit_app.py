"""Streamlit frontend for the Star Type Classifier.

Talks to two backend services:
  * FastAPI (API_URL)    - gets predictions from the registered model
  * MLflow  (MLFLOW_URL) - reads experiment results to explain how the model was chosen
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------- config
API_URL = os.getenv("API_URL", "http://localhost:8000")            # used by this app (inside Docker: http://api:8000)
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")      # used by this app (inside Docker: http://mlflow:5000)
API_PUBLIC_URL = os.getenv("API_PUBLIC_URL", "http://localhost:8000")      # links shown to the user's browser
MLFLOW_PUBLIC_URL = os.getenv("MLFLOW_PUBLIC_URL", "http://localhost:5000")
EXPERIMENT_NAME = "star-type-classification"
DATA_PATH = Path(os.getenv("DATA_PATH", Path(__file__).resolve().parent.parent / "data" / "stars.csv"))

COLORS = ["red", "orange-red", "orange", "yellow", "yellow-white", "white", "blue-white", "blue"]
SPECTRAL = ["O", "B", "A", "F", "G", "K", "M"]
SPECTRAL_HELP = {"O": "hottest, blue", "B": "very hot, blue-white", "A": "hot, white", "F": "warm, yellow-white",
                 "G": "Sun-like, yellow", "K": "cool, orange", "M": "coolest, red"}

STAR_TYPES = {
    "Brown Dwarf": dict(icon="🟤", color="#a0785a",
                        text="A “failed star”: too small to ignite hydrogen fusion. Tiny, cool and very dim."),
    "Red Dwarf": dict(icon="🔴", color="#ff5a4e",
                      text="The most common star in the galaxy. Small, cool and extremely long-lived."),
    "White Dwarf": dict(icon="⚪", color="#d9e4ff",
                        text="The hot, ultra-dense core left after a Sun-like star dies — about the size of Earth."),
    "Main Sequence": dict(icon="🟡", color="#ffd24a",
                          text="A star in the prime of its life, burning hydrogen in its core — just like our Sun."),
    "Supergiant": dict(icon="🟠", color="#ff9a3c",
                       text="A massive star near the end of its life, hundreds of times wider than the Sun."),
    "Hypergiant": dict(icon="💥", color="#b48cff",
                       text="The rarest, most extreme stars — up to a million times brighter than the Sun."),
}
TYPE_ORDER = list(STAR_TYPES)

PRESETS = {
    "☀️ The Sun": dict(desc="Our own star.",
                      values=dict(temperature=5778, luminosity=1.0, radius=1.0, abs_magnitude=4.83,
                                  star_color="yellow", spectral_class="G")),
    "🔴 Betelgeuse": dict(desc="The red shoulder of Orion — could explode as a supernova.",
                         values=dict(temperature=3600, luminosity=126000.0, radius=760.0, abs_magnitude=-5.85,
                                     star_color="red", spectral_class="M")),
    "🔵 Rigel": dict(desc="The bright blue foot of Orion.",
                    values=dict(temperature=12100, luminosity=120000.0, radius=79.0, abs_magnitude=-7.84,
                                star_color="blue-white", spectral_class="B")),
    "⚪ Sirius B": dict(desc="Tiny companion of the brightest star in our night sky.",
                       values=dict(temperature=25000, luminosity=0.056, radius=0.0084, abs_magnitude=11.18,
                                   star_color="white", spectral_class="A")),
    "🟠 Proxima Centauri": dict(desc="The closest star to the Sun, 4.2 light-years away.",
                               values=dict(temperature=3042, luminosity=0.0017, radius=0.154, abs_magnitude=15.5,
                                           star_color="red", spectral_class="M")),
    "✍️ My own star": dict(desc="Enter any values you like.", values=None),
}

# ---------------------------------------------------------------- page + styles
st.set_page_config(page_title="Star Type Classifier", page_icon="✨", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1250px;}
      .hero h1 {font-size: 2.4rem; margin-bottom: .2rem;}
      .hero p  {color: #aab3cf; font-size: 1.05rem; margin-top: 0;}
      .card {border-radius: 18px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
             background: linear-gradient(145deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
             border: 1px solid rgba(255,255,255,.10);}
      .result-icon {font-size: 3rem; line-height: 1;}
      .result-label {color: #aab3cf; font-size: .9rem; text-transform: uppercase; letter-spacing: .08em;}
      .result-name {font-size: 2.1rem; font-weight: 700; margin: .1rem 0 .4rem 0;}
      .bar {height: 10px; border-radius: 6px; background: rgba(255,255,255,.10); overflow: hidden;}
      .bar > div {height: 100%; border-radius: 6px;}
      .muted {color: #8d96b3; font-size: .9rem;}
      .hint {color: #8d96b3; font-size: .82rem; margin: -.6rem 0 .6rem 0;}
      .pill {display:inline-block; padding:.2rem .7rem; border-radius:999px; font-size:.82rem;
             background: rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12); margin-right:.4rem;}
      .dot-ok {color:#4ade80;} .dot-bad {color:#f87171;}
      .type-card h4 {margin: .2rem 0 .4rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- backend helpers
@st.cache_data
def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.rename(columns={"Temperature (K)": "temperature", "Luminosity (L/Lo)": "luminosity",
                              "Radius (R/Ro)": "radius", "Absolute magnitude (Mv)": "abs_magnitude",
                              "Star category": "category", "Spectral Class": "spectral_class"})


@st.cache_data(ttl=30)
def get_model_info():
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def predict(payload: dict):
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30)
def get_experiment_runs():
    """Read the latest run of every model type from the MLflow tracking server (REST API)."""
    try:
        exp = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/experiments/get-by-name",
                           params={"experiment_name": EXPERIMENT_NAME}, timeout=5)
        exp.raise_for_status()
        exp_id = exp.json()["experiment"]["experiment_id"]
        res = requests.post(f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
                            json={"experiment_ids": [exp_id], "max_results": 100,
                                  "order_by": ["attributes.start_time DESC"]}, timeout=5)
        res.raise_for_status()
    except (requests.RequestException, KeyError, ValueError):
        return None

    rows, seen = [], set()
    for run in res.json().get("runs", []):
        data = run.get("data", {})
        params = {p["key"]: p["value"] for p in data.get("params", [])}
        metrics = {m["key"]: m["value"] for m in data.get("metrics", [])}
        name = params.get("model_type")
        if not name or name in seen or "f1_macro" not in metrics:
            continue
        seen.add(name)
        rows.append({"model": name, "accuracy": metrics.get("accuracy"), "f1": metrics.get("f1_macro"),
                     "cv_f1": metrics.get("cv_f1_macro_mean"), "cv_std": metrics.get("cv_f1_macro_std", 0)})
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------- small UI helpers
def times_sun(x: float, more: str, less: str) -> str:
    if x <= 0:
        return ""
    if 0.9 <= x <= 1.1:
        return "about the same as the Sun"
    return f"≈ {x:,.0f}× {more} the Sun" if x > 1 else f"≈ {1 / x:,.0f}× {less} the Sun"


def fmt(x: float) -> str:
    """Human-friendly number: 204,000 / 10.6 / 0.00014 instead of 2.04e+05."""
    if x >= 1000:
        return f"{x:,.0f}"
    if x >= 1:
        return f"{x:.3g}"
    return np.format_float_positional(x, precision=2, unique=False, fractional=False, trim="-")


def nice_model_name(key: str | None) -> str:
    return {"logistic_regression": "Logistic Regression", "knn": "K-Nearest Neighbours", "svm_rbf": "SVM (RBF)",
            "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting"}.get(key or "", key or "—")


def apply_preset():
    values = PRESETS[st.session_state.preset]["values"]
    if values:
        for k, v in values.items():
            st.session_state[k] = v


def plot_layout(fig, height):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    return fig


# ---------------------------------------------------------------- header
info = get_model_info()
st.markdown('<div class="hero"><h1>✨ Star Type Classifier</h1>'
            "<p>Describe a star — a machine-learning model tells you what kind of star it is.</p></div>",
            unsafe_allow_html=True)
if info:
    st.markdown(f'<span class="pill"><span class="dot-ok">●</span> Model online</span>'
                f'<span class="pill">{nice_model_name(info.get("algorithm"))}</span>'
                f'<span class="pill">version {info.get("model_version", "?")}</span>',
                unsafe_allow_html=True)
else:
    st.markdown('<span class="pill"><span class="dot-bad">●</span> Model offline</span>', unsafe_allow_html=True)
    st.error("The prediction service is not reachable right now. Start it with `docker compose up -d api`.")

tab_classify, tab_model, tab_guide = st.tabs(["🔭 Classify a star", "🧠 How the model works", "📖 Star types guide"])

# ================================================================= TAB 1: classify
with tab_classify:
    if "preset" not in st.session_state:
        st.session_state.preset = "☀️ The Sun"
        apply_preset()

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.markdown("#### 1. Choose a star")
        st.selectbox("Famous stars", list(PRESETS), key="preset", on_change=apply_preset,
                     label_visibility="collapsed")
        st.caption(PRESETS[st.session_state.preset]["desc"] + " You can change any value below.")

        st.markdown("#### 2. Its properties")
        c1, c2 = st.columns(2)
        with c1:
            temperature = st.number_input("🌡️ Surface temperature (K)", 1000, 60000, step=100, key="temperature",
                                          help="Kelvin. The Sun's surface is about 5,778 K.")
            t_hint = ("about the same as the Sun" if abs(temperature - 5778) < 300
                      else f'{"hotter" if temperature > 5778 else "cooler"} than the Sun (5,778 K)')
            st.markdown(f'<div class="hint">{t_hint}</div>', unsafe_allow_html=True)
            radius = st.number_input("📏 Radius (× Sun)", 0.0001, 5000.0, format="%g", key="radius",
                                     help="How wide the star is compared to the Sun. 1 = same size as the Sun.")
            st.markdown(f'<div class="hint">{times_sun(radius, "wider than", "smaller than")}</div>',
                        unsafe_allow_html=True)
            star_color = st.selectbox("🎨 Colour", COLORS, key="star_color",
                                      format_func=lambda c: c.replace("-", " ").capitalize())
        with c2:
            luminosity = st.number_input("💡 Luminosity (× Sun)", 0.00001, 1e7, format="%g", key="luminosity",
                                         help="Total light the star emits compared to the Sun. 1 = as bright as the Sun.")
            st.markdown(f'<div class="hint">{times_sun(luminosity, "brighter than", "dimmer than")}</div>',
                        unsafe_allow_html=True)
            abs_magnitude = st.number_input("✴️ Absolute magnitude", -15.0, 25.0, step=0.1, key="abs_magnitude",
                                            help="Brightness on the astronomers' scale. LOWER = brighter. Sun = 4.83.")
            st.markdown('<div class="hint">lower number = brighter star (Sun = 4.83)</div>', unsafe_allow_html=True)
            spectral_class = st.selectbox("🌈 Spectral class", SPECTRAL, key="spectral_class",
                                          format_func=lambda s: f"{s} — {SPECTRAL_HELP[s]}")

        clicked = st.button("🔭 What kind of star is this?", type="primary", width="stretch",
                            disabled=info is None)

    payload = dict(temperature=float(temperature), luminosity=float(luminosity), radius=float(radius),
                   abs_magnitude=float(abs_magnitude), star_color=star_color, spectral_class=spectral_class)

    result = None
    if clicked:
        try:
            result = predict(payload)
            st.session_state.last = (payload, result)
        except requests.RequestException:
            st.error("Couldn't get an answer from the model. Check that the API container is running.")
    elif "last" in st.session_state and st.session_state.last[0] == payload:
        result = st.session_state.last[1]

    with right:
        if result:
            name = result["star_type_name"]
            meta = STAR_TYPES[name]
            conf = result["confidence"]
            sure = "very sure" if conf >= 0.85 else "fairly sure" if conf >= 0.6 else "not very sure"
            st.markdown(
                f"""<div class="card" style="border-color:{meta['color']}66">
                      <div style="display:flex; gap:1rem; align-items:center;">
                        <div class="result-icon">{meta['icon']}</div>
                        <div>
                          <div class="result-label">This star is a</div>
                          <div class="result-name" style="color:{meta['color']}">{name}</div>
                        </div>
                      </div>
                      <p style="margin:.6rem 0 1rem 0">{meta['text']}</p>
                      <div class="muted">The model is <b>{sure}</b> — {conf:.0%} confidence</div>
                      <div class="bar"><div style="width:{conf * 100:.0f}%; background:{meta['color']}"></div></div>
                    </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("##### How the model weighed every option")
            probs = (pd.DataFrame(result["probabilities"].items(), columns=["type", "p"])
                     .sort_values("p", ascending=True))
            fig_p = go.Figure(go.Bar(
                x=probs["p"], y=[f"{STAR_TYPES[t]['icon']} {t}" for t in probs["type"]], orientation="h",
                marker_color=[STAR_TYPES[t]["color"] for t in probs["type"]],
                text=[f"{p:.0%}" for p in probs["p"]], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            fig_p.update_xaxes(range=[0, 1.12], tickformat=".0%", showgrid=False)
            st.plotly_chart(plot_layout(fig_p, 240), width="stretch", config={"displayModeBar": False})
            if conf < 0.6:
                st.info("Low confidence means this star sits on the border between two types in the training data.")
        else:
            st.markdown('<div class="card"><div class="result-icon">🔭</div>'
                        '<p style="margin-top:.6rem">Pick a star on the left and press '
                        '<b>“What kind of star is this?”</b><br>'
                        '<span class="muted">The answer and the model\'s reasoning will appear here.</span></p></div>',
                        unsafe_allow_html=True)

    # ---- HR diagram
    st.markdown("#### Where your star sits among 240 known stars")
    st.caption("This is a Hertzsprung–Russell diagram — astronomers' “map” of stars. Hot stars are on the left, "
               "bright stars at the top, bubble size = star size. Stars of the same type cluster together, "
               "which is exactly what the model learns.")
    df = load_training_data()
    # radius spans 0.008 … 1950 × Sun, so scale bubbles by log(radius) or small stars become invisible
    df = df.assign(bubble=np.log10(df["radius"]).add(2.5).clip(lower=0.4))
    fig = px.scatter(df, x="temperature", y="luminosity", color="category", size="bubble", size_max=22,
                     log_x=True, log_y=True, category_orders={"category": TYPE_ORDER},
                     color_discrete_map={k: v["color"] for k, v in STAR_TYPES.items()},
                     labels={"temperature": "Surface temperature (K)", "luminosity": "Luminosity (× Sun)",
                             "category": "Type"},
                     hover_data={"radius": ":.3g", "luminosity": ":.3g", "temperature": ":,", "bubble": False})
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0)))
    fig.add_trace(go.Scatter(
        x=[temperature], y=[luminosity], mode="markers+text", name="Your star",
        text=[f"  {result['star_type_name']}" if result else "  your star"], textposition="middle right",
        textfont=dict(size=14, color="white"),
        marker=dict(symbol="star", size=26, color="#fff36b", line=dict(width=1.5, color="black"))))
    fig.update_xaxes(autorange="reversed", showgrid=False,
                     tickvals=[40000, 20000, 10000, 5000, 3000, 2000],
                     ticktext=["40,000", "20,000", "10,000", "5,000", "3,000", "2,000"])
    fig.update_yaxes(tickvals=[1e-4, 1e-2, 1, 1e2, 1e4, 1e6],
                     ticktext=["0.0001", "0.01", "1 (Sun)", "100", "10,000", "1,000,000"])
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)")
    st.plotly_chart(plot_layout(fig, 520), width="stretch")

# ================================================================= TAB 2: model
with tab_model:
    st.markdown("#### How this model was built")
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="card"><b>1. Train</b><br><span class="muted">Five different algorithms learn from '
                '240 labelled stars (6 types × 40).</span></div>', unsafe_allow_html=True)
    s2.markdown('<div class="card"><b>2. Track in MLflow</b><br><span class="muted">Every attempt is recorded: '
                'settings, scores and the trained model file.</span></div>', unsafe_allow_html=True)
    s3.markdown('<div class="card"><b>3. Register the best</b><br><span class="muted">The winner is saved in '
                'the MLflow Model Registry and served to this app.</span></div>', unsafe_allow_html=True)

    runs = get_experiment_runs()
    col_a, col_b = st.columns([1.4, 1], gap="large")
    with col_a:
        st.markdown("##### Which algorithm won?")
        if runs is not None:
            runs = runs.sort_values("cv_f1", ascending=True)
            winner = info.get("algorithm") if info else None
            fig_r = go.Figure(go.Bar(
                x=runs["cv_f1"], y=[nice_model_name(m) for m in runs["model"]], orientation="h",
                marker_color=["#fff36b" if m == winner else "#5b6b9a" for m in runs["model"]],
                text=[f"{v:.1%}" for v in runs["cv_f1"]], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            fig_r.update_xaxes(range=[0.9, 1.025], showgrid=False, tickvals=[0.9, 0.92, 0.94, 0.96, 0.98, 1.0], tickformat=".0%", title="Cross-validated F1 score (higher = better)")
            st.plotly_chart(plot_layout(fig_r, 300), width="stretch", config={"displayModeBar": False})
            st.caption("Score = average F1 over 5 cross-validation folds (axis starts at 90% to make small "
                       "differences visible). Highlighted = the model currently serving predictions.")
        else:
            st.warning("Experiment results are not available — is the MLflow server running and has `train.py` been run?")

    with col_b:
        st.markdown("##### The model in use")
        if info:
            source = "MLflow Model Registry" if info.get("source") == "mlflow-registry" else "local model file"
            st.markdown(
                f"""<div class="card">
                      <div class="muted">Name</div><b>{info.get('model_name', '—')}</b><br><br>
                      <div class="muted">Version</div><b>{info.get('model_version', '—')}</b>
                        {'&nbsp;<span class="pill">@' + info['alias'] + '</span>' if info.get('alias') else ''}<br><br>
                      <div class="muted">Algorithm</div><b>{nice_model_name(info.get('algorithm'))}</b><br><br>
                      <div class="muted">Loaded from</div><b>{source}</b>
                    </div>""", unsafe_allow_html=True)
        st.link_button("Open MLflow (experiments & registry) ↗", MLFLOW_PUBLIC_URL, width="stretch")
        st.link_button("Open API documentation ↗", f"{API_PUBLIC_URL}/docs", width="stretch")

    if runs is not None:
        with st.expander("All scores in a table"):
            table = runs.sort_values("cv_f1", ascending=False).assign(model=lambda d: d["model"].map(nice_model_name))
            st.dataframe(table.rename(columns={"model": "Algorithm", "accuracy": "Test accuracy", "f1": "Test F1",
                                               "cv_f1": "CV F1 (mean)", "cv_std": "CV F1 (std)"}),
                         hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.3f")
                                        for c in ["Test accuracy", "Test F1", "CV F1 (mean)", "CV F1 (std)"]})

    st.markdown("##### Good to know")
    st.markdown("- The dataset's star types are separated mostly by **absolute magnitude**, so the task is almost "
                "perfectly learnable — that is why scores are close to 100%.\n"
                "- Real stars can still land between types: Proxima Centauri is really a red dwarf, but it sits right "
                "on the dataset's border with brown dwarfs, so the model hesitates.")

# ================================================================= TAB 3: guide
with tab_guide:
    st.markdown("#### The six types of stars the model knows")
    st.caption("Typical ranges come from the 240 stars the model was trained on.")
    df = load_training_data()
    cols = st.columns(3)
    for i, name in enumerate(TYPE_ORDER):
        meta = STAR_TYPES[name]
        d = df[df["category"] == name]
        with cols[i % 3]:
            st.markdown(
                f"""<div class="card type-card" style="border-left:4px solid {meta['color']}">
                      <div style="font-size:2rem">{meta['icon']}</div>
                      <h4 style="color:{meta['color']}">{name}</h4>
                      <p>{meta['text']}</p>
                      <div class="muted">🌡️ {d.temperature.min():,} – {d.temperature.max():,} K<br>
                      💡 {fmt(d.luminosity.min())} – {fmt(d.luminosity.max())} × Sun<br>
                      📏 {fmt(d.radius.min())} – {fmt(d.radius.max())} × Sun</div>
                    </div>""", unsafe_allow_html=True)
