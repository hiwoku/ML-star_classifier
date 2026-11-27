"""Shared feature definitions and preprocessing.

Used by train.py (to build the pipeline) and by the API (to know the
expected inputs). Lives in its own package so the pickled pipeline can
find `clean_color` when it is loaded inside the Docker container.
"""
import numpy as np
import pandas as pd

# Raw CSV column -> short snake_case name used everywhere else
COLUMN_MAP = {
    "Temperature (K)": "temperature",
    "Luminosity (L/Lo)": "luminosity",
    "Radius (R/Ro)": "radius",
    "Absolute magnitude (Mv)": "abs_magnitude",
    "Star color": "star_color",
    "Spectral Class": "spectral_class",
    "Star type": "star_type",
}

NUMERIC_FEATURES = ["temperature", "luminosity", "radius", "abs_magnitude"]
# luminosity and radius span ~9 orders of magnitude -> log10 them
LOG_FEATURES = ["luminosity", "radius"]
LINEAR_FEATURES = ["temperature", "abs_magnitude"]
CATEGORICAL_FEATURES = ["star_color", "spectral_class"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "star_type"

CLASS_NAMES = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant",
}

COLORS = ["red", "orange-red", "orange", "yellow", "yellow-white", "white", "blue-white", "blue"]
SPECTRAL_CLASSES = ["O", "B", "A", "F", "G", "K", "M"]

# The raw data spells the same colour many ways ("Blue White", "Blue-white",
# "blue white" ...). Normalise them into the 8 canonical COLORS.
_COLOR_ALIASES = {
    "whitish": "white",
    "yellowish white": "yellow-white",
    "white-yellow": "yellow-white",
    "yellowish": "yellow",
    "pale yellow orange": "orange",
}


def clean_color(value: str) -> str:
    v = " ".join(str(value).strip().lower().split())
    v = _COLOR_ALIASES.get(v, v)
    return v.replace(" ", "-")


def clean_color_column(X):
    """FunctionTransformer-friendly wrapper: works on a DataFrame or 2D array."""
    df = pd.DataFrame(X).copy()
    for col in df.columns:
        df[col] = df[col].map(clean_color)
    return df.to_numpy()


def safe_log10(X):
    return np.log10(np.clip(np.asarray(X, dtype=float), 1e-6, None))


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=COLUMN_MAP)
    df = df[FEATURES + [TARGET]]
    df["star_color"] = df["star_color"].map(clean_color)
    df["spectral_class"] = df["spectral_class"].str.strip().str.upper()
    return df
