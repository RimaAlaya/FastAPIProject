import os

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List, Optional
import logging
from fastapi import APIRouter
from app.security import verify_api_key  # plus de boucle

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Chargement du modèle au démarrage
BASE_FILE_DIR = os.path.dirname(__file__)  # …/app/api
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_FILE_DIR, "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")  # …/FastAPIProject4/models

try:
    # Construire le chemin vers ton modèle dans /models/
    model_path = os.path.join(MODELS_DIR, "tire_anomaly_detector.keras")
    print(f"🔍 Chargement du modèle TIRE depuis : {model_path}")

    MODEL = tf.keras.models.load_model(model_path)
    THRESHOLD = 0.0045  # Seuil optimal depuis model_metrics.txt
    logger.info("✅ Modèle TIRE chargé avec succès")

except Exception as e:
    logger.error(f"❌ Erreur chargement modèle TIRE: {e}")
    MODEL = None
    THRESHOLD = None


class TireData(BaseModel):
    # Données capteurs obligatoires
    speed: float
    accelerometer_x: float
    accelerometer_y: float
    accelerometer_z: float
    gyroscope_x: float
    gyroscope_y: float
    gyroscope_z: float

    # PIDs étendus (optionnels - PID 0x76 ou 0xA6)
    pression_pneu_av_g: Optional[float] = None
    pression_pneu_av_d: Optional[float] = None
    pression_pneu_ar_g: Optional[float] = None
    pression_pneu_ar_d: Optional[float] = None


class TireStatus(BaseModel):
    status: str  # "normal" | "underinflated" | "warning"
    confidence: float
    method: str  # "direct" | "ai_model"
    details: dict


def calculate_features(data: TireData) -> dict:
    """Calcule les features nécessaires au modèle à partir des données capteurs"""
    features = {}

    # Features principales basées sur les capteurs obligatoires
    features['accelerometer_data_Z'] = data.accelerometer_z
    features['tire_resonance_low'] = abs(data.accelerometer_z) * 2.5 + abs(data.accelerometer_y) * 1.5
    features['tire_resonance_high'] = features['tire_resonance_low'] * 0.6

    # Calcul slip_ratio basé sur gyroscope et accéléromètre
    lateral_force = np.sqrt(data.accelerometer_x ** 2 + data.accelerometer_y ** 2)
    angular_velocity = np.sqrt(data.gyroscope_x ** 2 + data.gyroscope_y ** 2 + data.gyroscope_z ** 2)
    features['slip_ratio'] = lateral_force / max(data.speed, 1.0) * 0.1

    # Road roughness basé sur vibrations totales
    total_vibration = np.sqrt(data.accelerometer_x ** 2 + data.accelerometer_y ** 2 + data.accelerometer_z ** 2)
    features['road_roughness'] = total_vibration * 5.0

    # Crest factor de l'accéléromètre Z
    features['accel_Z_crest_factor'] = abs(data.accelerometer_z) * 1.2

    # Taux de changement de vitesse (proxy)
    features['speed_change_rate'] = data.speed * 0.01

    # Force latérale Pacejka (proxy basé sur accélération latérale)
    features['pacejka_lateral_force'] = lateral_force * 100.0

    # Score de stabilité route (inverse des vibrations)
    features['road_stability_score'] = max(0.1, 1.0 - total_vibration * 0.1)

    # Température simulée basée sur vitesse et vibrations
    features['tire_temperature'] = 25.0 + data.speed * 0.1 + total_vibration * 10.0

    return features


def direct_pressure_check(pressions: List[float]) -> TireStatus:
    """Détection directe basée sur les pressions des pneus"""
    min_pressure = 28.0  # PSI minimum recommandé
    warning_pressure = 30.0  # PSI d'avertissement

    low_count = sum(1 for p in pressions if p < min_pressure)
    warning_count = sum(1 for p in pressions if p < warning_pressure)

    if low_count > 0:
        status = "underinflated"
        confidence = min(1.0, low_count / 2.0)  # Plus de pneus sous-gonflés = plus de confiance
    elif warning_count > 0:
        status = "warning"
        confidence = 0.7
    else:
        status = "normal"
        confidence = 0.9

    return TireStatus(
        status=status,
        confidence=confidence,
        method="direct",
        details={
            "pressures": pressions,
            "low_pressure_count": low_count,
            "warning_count": warning_count,
            "min_pressure_threshold": min_pressure
        }
    )


def ai_model_prediction(features: dict) -> TireStatus:
    """Prédiction IA basée sur les features calculées"""
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modèle IA non disponible")

    try:
        # Ordre des features attendu par le modèle
        feature_order = [
            'tire_resonance_low', 'tire_resonance_high', 'road_roughness',
            'accel_Z_crest_factor', 'slip_ratio', 'speed_change_rate',
            'pacejka_lateral_force', 'road_stability_score', 'accelerometer_data_Z'
        ]

        # Créer séquence de 100 points (simulation temporelle)
        sequence_data = []
        for _ in range(100):
            point = [features.get(f, 0.0) for f in feature_order]
            # Ajouter un peu de bruit pour simuler des données temporelles
            point = [p + np.random.normal(0, 0.01) for p in point]
            sequence_data.append(point)

        # Préparer input pour le modèle
        X = np.array([sequence_data], dtype=np.float32)

        # Prédiction
        reconstruction = MODEL.predict(X, verbose=0)
        mse = np.mean(np.square(X - reconstruction))

        # Classification basée sur le seuil
        is_anomaly = mse > THRESHOLD
        confidence = min(1.0, mse / (THRESHOLD * 2)) if is_anomaly else min(1.0, (THRESHOLD - mse) / THRESHOLD)

        status = "underinflated" if is_anomaly else "normal"

        return TireStatus(
            status=status,
            confidence=confidence,
            method="ai_model",
            details={
                "mse_score": float(mse),
                "threshold": THRESHOLD,
                "features_used": feature_order
            }
        )

    except Exception as e:
        logger.error(f"Erreur prédiction IA: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")


@router.post("/detect_tire_status", response_model=TireStatus)
async def detect_tire_status(data: TireData,  _: None = Depends(verify_api_key )):
    """
    Endpoint principal pour la détection de l'état des pneus
    Utilise d'abord les PIDs étendus si disponibles, sinon le modèle IA
    """
    try:
        # Vérification PIDs étendus (PID 0x76 ou 0xA6)
        if all(p is not None and p > 0 for p in [
            data.pression_pneu_av_g, data.pression_pneu_av_d,
            data.pression_pneu_ar_g, data.pression_pneu_ar_d
        ]):
            # Détection directe avec PIDs étendus
            pressions = [
                data.pression_pneu_av_g,
                data.pression_pneu_av_d,
                data.pression_pneu_ar_g,
                data.pression_pneu_ar_d
            ]
            return direct_pressure_check(pressions)

        # Sinon utiliser le modèle IA avec les données capteurs
        features = calculate_features(data)
        return ai_model_prediction(features)

    except Exception as e:
        logger.error(f"Erreur détection: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la détection: {str(e)}")


