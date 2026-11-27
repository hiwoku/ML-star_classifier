# ✨ Star Type Classifier: FastAPI + Docker + Streamlit + MLflow

SIS-3 builds on Practical Task 6. A model classifies a star into one of **6 types** from its physical properties. It is served by **FastAPI**, used through a **Streamlit** UI, tracked and versioned in **MLflow**, and everything runs in **Docker**.

| Brown Dwarf | Red Dwarf | White Dwarf | Main Sequence | Supergiant | Hypergiant |
|---|---|---|---|---|---|

## Dataset

`data/stars.csv` has 240 stars, 40 of each class (perfectly balanced). It is the Kaggle *"Star dataset to predict star types"* (Deepraj Baidya).

| Feature | Meaning | Preprocessing |
|---|---|---|
| `temperature` | surface temperature, K | standard scaling |
| `luminosity` | L / L☉ (1e-4 … 8.5e5) | **log10** + scaling |
| `radius` | R / R☉ (0.008 … 1950) | **log10** + scaling |
| `abs_magnitude` | absolute visual magnitude Mv | standard scaling |
| `star_color` | observed colour | **cleaned** (19 messy spellings → 8 colours) + one-hot |
| `spectral_class` | O B A F G K M | one-hot |

All preprocessing lives inside the sklearn `Pipeline`, so the API sends raw values and the model handles the rest.

## Architecture

```
            ┌────────────┐   POST /predict   ┌───────────┐  models:/star-classifier@champion  ┌──────────────┐
 user ────▶ │ Streamlit  │ ────────────────▶ │  FastAPI  │ ─────────────────────────────────▶ │ MLflow server│
            │  :8501     │ ◀──── JSON ────── │  :8000    │   (fallback: models/model.joblib)  │ :5000        │
            └────────────┘                   └───────────┘                                    └──────▲───────┘
                                                                                                     │ runs, metrics,
                                                                                train.py ────────────┘ artifacts, registry
```

## Project structure

```
star-classifier/
├── data/stars.csv              # dataset
├── starml/features.py          # shared feature definitions + cleaning functions
├── train.py                    # trains 5 models, logs to MLflow, registers best, saves model.joblib
├── app/main.py                 # FastAPI: GET /, GET /health, GET /model-info, POST /predict
├── frontend/streamlit_app.py   # Streamlit UI + Hertzsprung–Russell diagram
├── models/model.joblib         # trained pipeline (local fallback for the API)
├── models/model_meta.json      # version / metrics of the saved model
├── Dockerfile                  # API image
├── Dockerfile.frontend         # Streamlit image
├── Dockerfile.mlflow           # MLflow tracking server + registry image
├── docker-compose.yml          # runs all three services
├── requirements.txt            # training + API
└── requirements-frontend.txt   # Streamlit
```

## MLflow integration

**Experiment tracking.** Experiment `star-type-classification` gets one run per model: Logistic Regression, KNN, SVM (RBF), Random Forest, Gradient Boosting. Each run logs:
- **parameters**: model type, hyperparameters, test size, CV folds, train/test sizes
- **metrics**: `accuracy`, `f1_macro`, `precision_macro`, `recall_macro`, `cv_f1_macro_mean`, `cv_f1_macro_std`
- **artifacts**: the model (with signature + input example), confusion matrix plot, classification report JSON

**Model registry.** The best model (highest 5-fold CV macro-F1) is registered as **`star-classifier`**, gets a new **version**, and the alias **`@champion`** points at it. The version also gets a description and an `algorithm` tag. The API loads `models:/star-classifier@champion`.

Results on our run:

| model | test accuracy | test F1 | CV F1 (5-fold) |
|---|---|---|---|
| logistic_regression | 0.967 | 0.967 | 0.966 ± 0.028 |
| knn | 0.983 | 0.983 | 0.972 ± 0.031 |
| svm_rbf | 1.000 | 1.000 | 0.989 ± 0.014 |
| **random_forest** ✅ | **1.000** | **1.000** | **1.000 ± 0.000** |
| gradient_boosting | 1.000 | 1.000 | 0.994 ± 0.011 |

## How to run

### Option A: everything in Docker (recommended)

```bash
# 1. start the MLflow server
docker compose up -d --build mlflow            # UI: http://localhost:5000

# 2. train locally, logging into that server
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
MLFLOW_TRACKING_URI=http://localhost:5000 python train.py
#   Windows PowerShell:  $env:MLFLOW_TRACKING_URI="http://localhost:5000"; python train.py

# 3. start API + frontend (the API pulls the @champion model from the registry)
docker compose up -d --build api frontend
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI Swagger | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

If you retrain later, run `docker compose restart api` so the API picks up the new champion.

### Option B: no Docker

```bash
pip install -r requirements.txt -r requirements-frontend.txt
python train.py                                  # tracks into ./mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
uvicorn app.main:app --reload                    # http://localhost:8000/docs
streamlit run frontend/streamlit_app.py          # http://localhost:8501
```

## API

`POST /predict`
```json
{"temperature": 5778, "luminosity": 1.0, "radius": 1.0, "abs_magnitude": 4.83,
 "star_color": "yellow", "spectral_class": "G"}
```
returns
```json
{"star_type": 3, "star_type_name": "Main Sequence", "confidence": 0.9726,
 "probabilities": {"Brown Dwarf": 0.0, "Red Dwarf": 0.0, "White Dwarf": 0.0,
                   "Main Sequence": 0.9726, "Supergiant": 0.0035, "Hypergiant": 0.0239},
 "model_version": "1"}
```
Invalid inputs (negative radius, unknown colour like `"purple"`) are rejected with HTTP 422.

## Streamlit UI

- The sidebar shows the API status, which model version is served, and where it was loaded from (registry or file).
- Pick a real star (Sun, Betelgeuse, Rigel, Sirius B, Proxima Centauri) or enter custom values.
- The result panel shows the predicted type, confidence and a probability bar chart.
- A **Hertzsprung–Russell diagram** shows all 240 training stars with your star as a ⭐.

## Observations / limitations

- **~100% accuracy is real, but expected.** The classes in this dataset are almost exactly bands of absolute magnitude: Brown Dwarf Mv > 16, Red/White Dwarf 10…15, Supergiant −7.45…−5.2, Hypergiant < −7.6. The task is close to separable, and tree models learn these thresholds.
- **Real stars expose it.** Proxima Centauri (Mv ≈ 15.5) falls between the Red Dwarf and Brown Dwarf bands, and the model splits about 58 / 41. It is really a red dwarf. Rigel (Mv ≈ −7.8) comes out as a "Hypergiant" by this dataset's thresholds, although astronomers call it a supergiant. The model learns the dataset's labelling rules, not astrophysics.
- 240 rows is small, so the test set is only 60 stars. We pick the model by 5-fold CV rather than test score.
