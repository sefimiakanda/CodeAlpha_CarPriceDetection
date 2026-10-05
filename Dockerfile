FROM python:3.12-slim

# Utiliser la dernière version stable d'uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Variables d'environnement pour uv et Python
ENV UV_LINK_MODE=copy PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 1) Copier uniquement les fichiers de dépendances d'abord
COPY pyproject.toml uv.lock ./

# 2) Installer les dépendances verrouillées, sans les outils de développement
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# 3) Copier le code source et les données, puis entraîner le modèle
COPY src ./src
COPY data ./data
RUN python src/train.py

# 4) Exécution sous un compte non privilégié
RUN useradd --system app
USER app

EXPOSE 8000

# Commande de lancement de l'API
CMD ["sh", "-c", "exec uvicorn api:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}"]