# app/security.py
from fastapi import Header, HTTPException, status
from dotenv import load_dotenv
import os

load_dotenv(".env.local")
API_KEY = os.getenv("API_KEY")

def verify_api_key(x_api_key: str = Header(..., description="Clé API")) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide"
        )
