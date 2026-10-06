# 🛻 Car Price Prediction

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange?logo=scikitlearn&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-purple)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=white)

A production-ready machine learning pipeline and web service that predicts used car selling prices.

🔗 **Live demo:** [https://car-price-prediction-qqrm.onrender.com/](https://car-price-prediction-qqrm.onrender.com/)

## 🎯 Project Overview & Objectives

Developed as part of an internship project at CodeAlpha. The model is built with scikit-learn, served through FastAPI, and ships with a simple web form so that non-technical users can get an estimate without touching the API.

- **Train** a Random Forest regression pipeline (preprocessing + model saved together).
- **Serve** predictions through a validated REST API.
- **Use** a friendly web interface at `/` (no technical knowledge required).
- **Deploy** for free on Render using a `render.yaml` Blueprint, with Docker kept ready for a future containerized release.


## 📂 Structure

```
CodeAlpha_CarPriceDetection/
├── data/car_price.csv           # Dataset containing used car attributes
├── notebooks/exploration.ipynb  # Exploratory Data Analysis & feature engineering notes
├── src/
│   ├── train.py                 # Model training -> models/model.joblib
│   ├── api.py                   # API: POST /predict, GET /health, serves the web form
│   └── static/index.html        # Web interface (form + result display)
├── tests/test_api.py            # API tests
├── models/                      # Serialized model directory (generated, not versioned)
├── pyproject.toml               # Project metadata and dependencies
├── uv.lock                      # Locked dependency versions (must be committed)
├── .python-version              # Python version used locally and by Render
├── render.yaml                  # Render Blueprint (native Python deployment)
├── Dockerfile                   # Container definition (optional, for future use)
└── .gitignore
```

## 🚀 Local Development

Prerequisites: make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

```bash
uv sync                                          # Install dependencies and generate uv.lock
uv run python src/train.py                       # Train the model and save artifacts to models/
uv run pytest                                    # Run API and pipeline tests
uv run uvicorn api:app --app-dir src --reload    # Launch local development server on port 8000
```

Once running:

- open **http://localhost:8000** for the web form (click *Fill an example* to try it quickly);
- open **http://localhost:8000/docs** for the Swagger UI: choose `POST /predict` → *Try it out* → *Execute*.

To launch the interactive notebook environment:

```bash
uv run jupyter lab
```

Then open `notebooks/exploration.ipynb`.

### Customizing the web interface

The page is a single file: `src/static/index.html`. The price unit shown next to the result is controlled by one line near the end of the file:

```js
const UNIT = "";   // e.g. "lakhs ₹". Leave empty if the dataset's unit is unknown.
```

## 🐳 Docker Containerization

To package and run the application inside a container:

```bash
uv sync                                  # Generate uv.lock locally first
docker build -t car-price .              # Build image & train model during the build phase
docker run --rm -p 8000:8000 car-price   # Run container; access the app on http://localhost:8000
```

If the build stays at `uv sync` on Docker Desktop, retry with host networking:

```bash
docker build --network=host -t car-price .
```

This requires host networking to be enabled in Docker Desktop. If it is not available, restart Docker Desktop and check its network, VPN, firewall, or proxy configuration.


## ☁️ Deployment on Render

The service is deployed with docker, described entirely by `render.yaml`. Render installs the dependencies with uv, trains the model during the build, then starts the API. No Docker image is needed.

### 1. Deploy

1. Push the repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com), click **New +** → **Blueprint**.
3. Connect your GitHub account and select this repository. Render reads `render.yaml` and shows the service it will create.
4. Click **Apply** (or **Deploy Blueprint**).
5. Open the service's **Logs** tab. You should see, in order:
   1. the installation of the dependencies by uv;
   2. the training metrics (`MAE=… RMSE=… R2=…`) followed by `Modèle sauvegardé`;
   3. `Uvicorn running on http://0.0.0.0:…`.
6. When the status becomes **Live**, open the URL shown at the top of the page (`https://car-price-api-xxxx.onrender.com`).

Check that it works:

- `/` → the web form;
- `/health` → `{"status":"ok"}`;
- `/docs` → interactive API documentation.

### 2. Updating the deployed app

Every `git push` to the linked branch triggers a new build and deployment automatically:

```bash
git add .
git commit -m "Describe your change"
git push
```

If you change a dependency, run `uv add <package>` locally and commit both `pyproject.toml` and `uv.lock`.

### 3. Free tier limitations

Free-tier limits change regularly; check Render's pricing page before relying on them.

- The service **spins down after 15 minutes of inactivity**; the next request takes about a minute to wake it up (the web form displays a waiting message when this happens).
- Roughly **512 MB of RAM**, which is enough for this model.
- The filesystem is **ephemeral**: anything written at runtime is lost on restart or redeploy. This is not an issue here because the model is rebuilt on every deployment.
- The URL is **public and unauthenticated**. Do not send sensitive data.


## 💎 The 6 Pillars of a Production-Ready Project

1. **Self-Contained Preprocessing**: Data cleaning and transformations are embedded directly within the saved scikit-learn pipeline, ensuring the API can never bypass or forget preprocessing steps.

2. **Locked Dependencies**: Precise dependency locking (`uv.lock`) guarantees exact environmental consistency across all machines and deployment targets.

3. **Strict Input Validation**: Payloads are validated via Pydantic; invalid data formats or outlier values immediately return an HTTP 422 Unprocessable Entity rather than generating corrupted predictions.

4. **Health Check Endpoint**: A dedicated `/health` route allows cloud platforms and monitoring tools to continuously verify service uptime.

5. **Reproducible Deployment**: `render.yaml` describes the whole deployment as code (build, start, health check), and the `Dockerfile` provides an identical, containerized alternative for future releases.

6. **Automated Testing**: Integration tests explicitly validate the HTTP behavior, the web page, and the model inference routes.

## 👤 Author

**Fidèle Miakanda** — *Data Science Intern at CodeAlpha*
  * [GitHub](https://github.com/sefimiakanda)
  * [LinkedIn](https://www.linkedin.com/in/fidèle-miakanda)
