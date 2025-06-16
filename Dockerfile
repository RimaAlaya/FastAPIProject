# Utiliser une image légère avec Python
FROM python:3.11-slim

# Créer le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY ./app ./app
COPY ./models ./models
COPY requirements.txt .

# Installer les dépendances
RUN pip install --upgrade pip && pip install -r requirements.txt

# Exposer le port utilisé par uvicorn
EXPOSE 8085

# Lancer l'API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8085"]
