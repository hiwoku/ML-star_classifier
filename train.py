"""Train several classifiers, track every run in MLflow, register the best one.

Usage:
    python train.py                          # local sqlite tracking (mlflow.db)
    MLFLOW_TRACKING_URI=http://localhost:5000 python train.py   # docker-compose MLflow server
"""
import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from starml.features import (
    CATEGORICAL_FEATURES,
    CLASS_NAMES,
    FEATURES,
    LINEAR_FEATURES,
    LOG_FEATURES,
    TARGET,
    clean_color_column,
    load_dataset,
    safe_log10,
)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "stars.csv"
MODEL_DIR = ROOT / "models"
EXPERIMENT_NAME = "star-type-classification"
REGISTERED_MODEL_NAME = "star-classifier"
RANDOM_STATE = 42

# MLflow 3 saves sklearn models with skops, which refuses to (de)serialise types
# it doesn't know. These are our own preprocessing functions + sklearn's tree
# storage (RandomForest / GradientBoosting); we trained the models ourselves.
SKOPS_TRUSTED_TYPES = [
    "starml.features.clean_color_column",
    "starml.features.safe_log10",
    "sklearn.tree._tree.Tree",
]

CANDIDATES = {
    "logistic_regression": (LogisticRegression, {"C": 1.0, "max_iter": 2000}),
    "knn": (KNeighborsClassifier, {"n_neighbors": 5, "weights": "distance"}),
    "svm_rbf": (SVC, {"C": 10.0, "gamma": "scale", "probability": True, "random_state": RANDOM_STATE}),
    "random_forest": (RandomForestClassifier, {"n_estimators": 300, "max_depth": 8, "random_state": RANDOM_STATE}),
    "gradient_boosting": (GradientBoostingClassifier, {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3, "random_state": RANDOM_STATE}),
}


def build_pipeline(estimator) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("log_num", Pipeline([("log10", FunctionTransformer(safe_log10)), ("scale", StandardScaler())]), LOG_FEATURES),
            ("num", StandardScaler(), LINEAR_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("clean", FunctionTransformer(clean_color_column)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def save_confusion_matrix(y_true, y_pred, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=list(CLASS_NAMES.values()), ax=ax, xticks_rotation=45, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {tracking_uri}")

    df = load_dataset(str(DATA_PATH))
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    MODEL_DIR.mkdir(exist_ok=True)

    results = []
    for name, (cls, params) in CANDIDATES.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = build_pipeline(cls(**params))

            cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "precision_macro": precision_score(y_test, y_pred, average="macro"),
                "recall_macro": recall_score(y_test, y_pred, average="macro"),
                "cv_f1_macro_mean": cv_f1.mean(),
                "cv_f1_macro_std": cv_f1.std(),
            }

            # --- params ---
            mlflow.log_param("model_type", name)
            mlflow.log_params(params)
            mlflow.log_params({"test_size": 0.25, "cv_folds": 5, "n_train": len(X_train), "n_test": len(X_test)})
            mlflow.set_tag("dataset", "Stars (240 stars, 6 classes)")

            # --- metrics ---
            mlflow.log_metrics(metrics)

            # --- artifacts ---
            cm_path = MODEL_DIR / f"cm_{name}.png"
            save_confusion_matrix(y_test, y_pred, cm_path, f"{name} — test set")
            mlflow.log_artifact(str(cm_path), artifact_path="plots")
            report = classification_report(y_test, y_pred, target_names=list(CLASS_NAMES.values()), output_dict=True)
            mlflow.log_dict(report, "classification_report.json")

            signature = infer_signature(X_train, pipe.predict(X_train))
            model_info = mlflow.sklearn.log_model(
                pipe,
                name="model",
                signature=signature,
                input_example=X_train.head(3),
                code_paths=[str(ROOT / "starml")],
                                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )

            cm_path.unlink()
            results.append({"name": name, "run_id": run.info.run_id, "model_uri": model_info.model_uri, "pipeline": pipe, **metrics})
            print(f"{name:20s} acc={metrics['accuracy']:.3f} f1={metrics['f1_macro']:.3f} "
                  f"cv_f1={metrics['cv_f1_macro_mean']:.3f}±{metrics['cv_f1_macro_std']:.3f}")

    # Best = highest CV F1 (more reliable than a 60-row test set), tie-break by test F1
    best = max(results, key=lambda r: (round(r["cv_f1_macro_mean"], 4), r["f1_macro"]))
    print(f"\nBest model: {best['name']} (run {best['run_id']})")

    # --- Model Registry ---
    mv = mlflow.register_model(best["model_uri"], REGISTERED_MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)
    client.update_registered_model(
        REGISTERED_MODEL_NAME, description="Classifies a star into one of 6 types from its physical properties."
    )
    client.update_model_version(
        REGISTERED_MODEL_NAME, mv.version,
        description=f"{best['name']}: cv_f1={best['cv_f1_macro_mean']:.3f}, test_f1={best['f1_macro']:.3f}",
    )
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "algorithm", best["name"])
    print(f"Registered '{REGISTERED_MODEL_NAME}' version {mv.version} with alias @champion")

    # --- Local artifact for the API (same as Practical Task 6) ---
    joblib.dump(best["pipeline"], MODEL_DIR / "model.joblib")
    meta = {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": str(mv.version),
        "algorithm": best["name"],
        "run_id": best["run_id"],
        "metrics": {k: round(float(best[k]), 4) for k in ("accuracy", "f1_macro", "cv_f1_macro_mean")},
        "classes": CLASS_NAMES,
    }
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
