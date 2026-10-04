import pytest
from fastapi.testclient import TestClient

import train
from api import app

CAR = {
    "Car_Name": "ritz",
    "Year": 2014,
    "Present_Price": 5.59,
    "Driven_kms": 27000,
    "Fuel_Type": "Petrol",
    "Selling_type": "Dealer",
    "Transmission": "Manual",
    "Owner": 0,
}


@pytest.fixture(scope="module")
def client():
    if not train.MODEL_PATH.exists():  # entraîne le modèle si ce n'est pas déjà fait
        train.main()
    with TestClient(app) as test_client:  # `with` déclenche le chargement du modèle
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_plausible_price(client):
    response = client.post("/predict", json=CAR)
    assert response.status_code == 200
    assert 1 < response.json()["predicted_selling_price"] < 8  # valeur réelle : 3.35


def test_stray_whitespace_in_name_gives_same_price(client):
    clean = client.post("/predict", json=CAR).json()
    padded = client.post("/predict", json={**CAR, "Car_Name": "  ritz "}).json()
    assert clean == padded


def test_invalid_fuel_type_is_rejected(client):
    assert client.post("/predict", json={**CAR, "Fuel_Type": "Hydrogen"}).status_code == 422


def test_negative_price_is_rejected(client):
    assert client.post("/predict", json={**CAR, "Present_Price": -1}).status_code == 422
