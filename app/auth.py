# app/auth.py

import os
import secrets
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# Charger les variables d'environnement (une seule fois suffit, mais on le met par sécurité ici)
load_dotenv(".env.local")

BASIC_USERNAME = os.getenv("BASIC_AUTH_USERNAME")
BASIC_PASSWORD = os.getenv("BASIC_AUTH_PASSWORD")

# Créer l'objet HTTPBasic
security = HTTPBasic()

def check_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Protège les routes (comme Swagger) avec Basic Auth.
    Compare les identifiants aux variables d'environnement.
    """
    valid_user = secrets.compare_digest(credentials.username, BASIC_USERNAME)
    valid_pass = secrets.compare_digest(credentials.password, BASIC_PASSWORD)

    if not (valid_user and valid_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username
