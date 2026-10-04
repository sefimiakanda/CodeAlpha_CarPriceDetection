# 🛻 Car Price Prediction

A production-ready machine learning pipeline and web service predicting used car selling prices. 

## 🎯 Project Overview & Objectives

Developed as part of an internship project at CodeAlpha, built with scikit-learn, served via FastAPI, and containerized using Docker.

## Structure

```
CodeAlpha_CarPriceDetection/
├── data/car_price.csv        # ataset containing used car attributes
├── notebooks/exploration.ipynb  # Exploratory Data Analysis & feature engineering notes
├── src/
│   ├── train.py              # model training -> models/model.joblib
│   └── api.py                # API : POST /predict, GET /health
├── tests/test_api.py         # API tests 
├── models/                   # Serialized model directory
├── pyproject.toml            # Project metadata and dependencies 
├── Dockerfile                # Container definition
└── .gitignore
```


## 🚀 Local Development

Prerequisites: Make sure you have uv installed.

```bash
uv sync                                        # Install dependencies and generate uv.lock
uv run python src/train.py                     # Train the model and save artifacts to models/
uv run pytest                                  # Run API and pipeline tests
uv run uvicorn api:app --app-dir src --reload    # Launch local development server on port 8000
```

Once running, open http://localhost:8000/docs to interact with the Swagger UI: choose POST /predict → Try it out → Execute.

To launch the interactive notebook environment: `uv run jupyter lab`, ten open `notebooks/exploration.ipynb`.

## 🐳 Docker Containerization

To package and run the application inside a container:

```bash
uv sync                                  # Generate uv.lock locally first
docker build -t car-price .              # Build image & train model during the build phase
docker run --rm -p 8000:8000 car-price   # Run container; access API on http://localhost:8000/docs
```

## ☁️ Deployment

Render Deployment Instructions:

1. Push your repository to GitHub.
2. Go to Render Dashboard → New → Web Service and select your repository.
3. Set Language to Docker.
4. Choose your Instance Type (e.g., Free tier).
5. Set /health as the Health Check Path.

Note on Free Tier limitations: The service spins down after 15 minutes of inactivity, resulting in an approximate 1-minute cold start upon the next incoming request.

## 💎 The 6 Pillars of a Production-Ready Project

1. Self-Contained Preprocessing: Data cleaning and transformations are embedded directly within the saved scikit-learn pipeline, ensuring the API can never bypass or forget preprocessing steps.

2. Locked Dependencies: Precise dependency locking (uv.lock) guarantees exact environmental consistency across all machines and deployment targets.

3. Strict Input Validation: Payloads are validated via Pydantic; invalid data formats or outlier values immediately return an HTTP 422 Unprocessable Entity rather than generating corrupted predictions.

4. Health Check Endpoint: A dedicated /health route allows cloud orchestrators and monitoring tools to continuously verify service uptime.

5. Standardized Containerization: The Dockerfile abstracts away system-level dependencies, ensuring identical behavior in local development and production.

6. Automated Testing: Integration tests explicitly validate HTTP behavior and model inference routes.

## 👤 Author

**Fidèle Miakanda Sefi** — *Data Science Intern at CodeAlpha*
  * [GitHub](https://github.com/sefimiakanda)
  * [LinkedIn](https://www.linkedin.com/in/fidèle-miakanda)



