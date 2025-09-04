FROM python:3.11-slim-bookworm

# Bonnes pratiques Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances système minimales + make + dos2unix
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential make dos2unix \
 && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'appli
COPY . .

# Convertir CRLF -> LF et rendre exécutable le script de démarrage
RUN dos2unix /app/app/run.sh || true \
 && chmod +x /app/app/run.sh

# Port FastAPI
EXPOSE 8000

# Démarrage
CMD ["bash", "-lc", "./app/run.sh"]
