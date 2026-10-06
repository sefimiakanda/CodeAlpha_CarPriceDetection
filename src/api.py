"""API de prédiction.

Lancer :  uv run uvicorn api:app --app-dir src --reload
Docs interactives : http://localhost:8000/docs
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from train import FEATURES, METRICS_PATH, MODEL_PATH, clean_car_name

STATIC_DIR = Path(__file__).resolve().parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api")


class Car(BaseModel):
    """Entrée de l'API. Pydantic rejette automatiquement (erreur 422) toute valeur invalide."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "Car_Name": "ritz",
                    "Year": 2014,
                    "Present_Price": 5.59,
                    "Driven_kms": 27000,
                    "Fuel_Type": "Petrol",
                    "Selling_type": "Dealer",
                    "Transmission": "Manual",
                    "Owner": 0,
                }
            ]
        },
    )

    Car_Name: str = Field(min_length=1, max_length=100)
    Year: int = Field(ge=1990, le=date.today().year + 1)
    Present_Price: float = Field(gt=0, le=1000, description="Prix actuel du neuf")
    Driven_kms: int = Field(ge=0, le=2_000_000)
    Fuel_Type: Literal["Petrol", "Diesel", "CNG"]
    Selling_type: Literal["Dealer", "Individual"]
    Transmission: Literal["Manual", "Automatic"]
    Owner: int = Field(default=0, ge=0, le=10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le modèle est chargé UNE fois au démarrage, pas à chaque requête.
    if not MODEL_PATH.exists():
        raise RuntimeError("Modèle introuvable. Lancez d'abord : uv run python src/train.py")
    app.state.model = joblib.load(MODEL_PATH)
    # Pour l'interface web : noms de véhicules connus du modèle + erreur moyenne mesurée au test
    encoder = app.state.model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    app.state.car_names = sorted(str(name) for name in encoder.categories_[0])  # Car_Name = 1re colonne catégorielle
    app.state.mae = json.loads(METRICS_PATH.read_text())["mae"] if METRICS_PATH.exists() else None
    logger.info("Modèle chargé depuis %s", MODEL_PATH)
    yield


app = FastAPI(title="Car Price API", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def home():
    """Page d'accueil : le formulaire destiné aux utilisateurs."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/options", include_in_schema=False)
def options(request: Request):
    """Données pour pré-remplir le formulaire (liste des modèles, erreur moyenne)."""
    return {"car_names": request.app.state.car_names, "mae": request.app.state.mae}


@app.get("/health")
def health():
    """Utilisé par les plateformes pour savoir si l'API tourne."""
    return {"status": "ok"}


@app.post("/predict")
def predict(car: Car, request: Request):
    start = time.perf_counter()
    row = car.model_dump()
    row["Car_Name"] = clean_car_name(row["Car_Name"])
    frame = pd.DataFrame([row], columns=FEATURES)  # même ordre de colonnes qu'à l'entraînement
    try:
        price = float(request.app.state.model.predict(frame)[0])
    except Exception:
        logger.exception("Échec de la prédiction")
        raise HTTPException(status_code=500, detail="Échec de la prédiction") from None

    logger.info("prediction=%.3f duree_ms=%.1f entree=%s", price, (time.perf_counter() - start) * 1000, row)
    return {"predicted_selling_price": round(price, 3)}
