"""Entraîne le modèle et le sauvegarde dans models/.

Usage :  uv run python src/train.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Chemins (relatifs à la racine du projet, donc valables partout, y compris dans Docker) ---
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "car_price.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

# --- Colonnes : listes explicites, réutilisées par l'API ---
TARGET = "Selling_Price"
NUMERIC = ["Year", "Present_Price", "Driven_kms", "Owner"]
CATEGORICAL = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]
FEATURES = NUMERIC + CATEGORICAL

MIN_R2 = 0.90  # sécurité : on refuse de sauvegarder un modèle trop mauvais


def clean_car_name(name: str) -> str:
    """Enlève les espaces en trop. Utilisée à l'entraînement ET dans l'API (même traitement)."""
    return " ".join(str(name).split())


def build_model() -> Pipeline:
    """Prétraitement + modèle dans un seul objet : l'API n'a rien à refaire."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    return Pipeline(
        [("preprocessor", preprocessor), ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))]
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["Car_Name"] = df["Car_Name"].map(clean_car_name)

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }
    print(f"MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.4f}")

    if metrics["r2"] < MIN_R2:
        raise SystemExit(f"R2 trop bas (< {MIN_R2}) : modèle non sauvegardé.")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Modèle sauvegardé : {MODEL_PATH}")


if __name__ == "__main__":
    main()
