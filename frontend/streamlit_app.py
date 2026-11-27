"""Streamlit frontend: enter a star's properties, get its type from the FastAPI service,
and see where it sits on the Hertzsprung-Russell diagram."""
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_PATH = Path(os.getenv("DATA_PATH", Path(__file__).resolve().parent.parent / "data" / "stars.csv"))

COLORS = ["red", "orange-red", "orange", "yellow", "yellow-white", "white", "blue-white", "blue"]
SPECTRAL = ["O", "B", "A", "F", "G", "K", "M"]

# Real stars to try (approximate published values)
PRESETS = {
    "☀️ The Sun": dict(temperature=5778, luminosity=1.0, radius=1.0, abs_magnitude=4.83, star_color="yellow", spectral_class="G"),
    "🔴 Betelgeuse": dict(temperature=3600, luminosity=126000.0, radius=760.0, abs_magnitude=-5.85, star_color="red", spectral_class="M"),
    "🔵 Rigel": dict(temperature=12100, luminosity=120000.0, radius=79.0, abs_magnitude=-7.84, star_color="blue-white", spectral_class="B"),
    "⚪ Sirius B": dict(temperature=25000, luminosity=0.056, radius=0.0084, abs_magnitude=11.18, star_color="white", spectral_class="A"),
    "🟠 Proxima Centauri": dict(temperature=3042, luminosity=0.0017, radius=0.154, abs_magnitude=15.5, star_color="red", spectral_class="M"),
    "✍️ Custom": None,
}

st.set_page_config(page_title="Star Type Classifier", page_icon="✨", layout="wide")
st.title("✨ Star Type Classifier")
st.caption("Enter a star's physical properties — the model (served by FastAPI, tracked in MLflow) "
           "predicts its type and plots it on the Hertzsprung–Russell diagram.")


@st.cache_data
def load_training_data():
    df = pd.read_csv(DATA_PATH)
    return df.rename(columns={"Temperature (K)": "temperature", "Luminosity (L/Lo)": "luminosity",
                              "Radius (R/Ro)": "radius", "Star category": "category"})


def api_get(path):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


# ---------- sidebar: API status + inputs ----------
with st.sidebar:
    st.header("API")
    info = api_get("/model-info")
    if info:
        st.success(f"Connected to {API_URL}")
        st.write(f"**Model:** {info.get('model_name', '—')}  v{info.get('model_version', '?')}")
        st.write(f"**Algorithm:** {info.get('algorithm', '—')}")
        st.write(f"**Loaded from:** {info.get('source')}")
    else:
        st.error(f"API not reachable at {API_URL}")

    st.header("Star")
    preset_name = st.selectbox("Pick a real star or enter your own", list(PRESETS))
    p = PRESETS[preset_name] or PRESETS["☀️ The Sun"]
    custom = PRESETS[preset_name] is None

    temperature = st.number_input("Temperature (K)", 1000, 60000, int(p["temperature"]), step=100, disabled=not custom)
    luminosity = st.number_input("Luminosity (L/L☉)", 0.00001, 1e7, float(p["luminosity"]), format="%.5f", disabled=not custom)
    radius = st.number_input("Radius (R/R☉)", 0.0001, 5000.0, float(p["radius"]), format="%.4f", disabled=not custom)
    abs_magnitude = st.number_input("Absolute magnitude (Mv)", -15.0, 25.0, float(p["abs_magnitude"]), step=0.1, disabled=not custom)
    star_color = st.selectbox("Colour", COLORS, index=COLORS.index(p["star_color"]), disabled=not custom)
    spectral_class = st.selectbox("Spectral class", SPECTRAL, index=SPECTRAL.index(p["spectral_class"]), disabled=not custom)
    go_btn = st.button("🔭 Classify", type="primary", width="stretch")

payload = dict(temperature=temperature, luminosity=luminosity, radius=radius,
               abs_magnitude=abs_magnitude, star_color=star_color, spectral_class=spectral_class)

# ---------- prediction ----------
col_res, col_plot = st.columns([1, 2])
result = None
if go_btn:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        result = r.json()
        st.session_state["last"] = (payload, result)
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
elif "last" in st.session_state and st.session_state["last"][0] == payload:
    result = st.session_state["last"][1]

with col_res:
    st.subheader("Prediction")
    if result:
        st.metric("Star type", result["star_type_name"], f"{result['confidence']:.0%} confidence")
        probs = pd.DataFrame(result["probabilities"].items(), columns=["type", "probability"])
        fig_p = px.bar(probs, x="probability", y="type", orientation="h", range_x=[0, 1])
        fig_p.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
        st.plotly_chart(fig_p, width="stretch")
        with st.expander("Raw request / response"):
            st.json({"request": payload, "response": result})
    else:
        st.info("Choose a star in the sidebar and press **Classify**.")

with col_plot:
    st.subheader("Hertzsprung–Russell diagram")
    df = load_training_data()
    fig = px.scatter(
        df, x="temperature", y="luminosity", color="category", size="radius", size_max=25,
        log_x=True, log_y=True, hover_data=["radius", "Spectral Class"],
        category_orders={"category": ["Brown Dwarf", "Red Dwarf", "White Dwarf",
                                      "Main Sequence", "Supergiant", "Hypergiant"]},
        labels={"temperature": "Temperature (K)", "luminosity": "Luminosity (L/L☉)", "category": "Training data"},
    )
    fig.update_traces(marker=dict(opacity=0.55, line=dict(width=0)))
    fig.add_trace(go.Scatter(
        x=[temperature], y=[luminosity], mode="markers+text", name="Your star",
        text=[result["star_type_name"] if result else "your star"], textposition="top center",
        marker=dict(symbol="star", size=24, color="gold", line=dict(width=1.5, color="black")),
    ))
    fig.update_xaxes(autorange="reversed")  # hot stars on the left, as astronomers draw it
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
