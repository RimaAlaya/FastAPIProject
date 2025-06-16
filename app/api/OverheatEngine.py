from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Optional
import logging
from pathlib import Path
from fastapi import APIRouter
from app.security import verify_api_key  # plus de boucle

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'temp_critical': 100,
    'temp_warning': 92,
    'temp_attention': 85,
}

router = APIRouter()

APP_API_DIR = os.path.dirname(__file__)

PROJECT_ROOT = os.path.abspath(os.path.join(APP_API_DIR, "..", ".."))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Modèles Pydantic pour l'API
class OBDData(BaseModel):
    """Données d'entrée OBD-II (PIDs standards)"""
    engine_coolant_temp: float = Field(..., ge=-40, le=150, description="Température liquide de refroidissement (°C)")
    engine_rpm: float = Field(..., ge=0, le=8000, description="Régime moteur (RPM)")
    vehicle_speed: float = Field(..., ge=0, le=300, description="Vitesse véhicule (km/h)")
    throttle_position: float = Field(..., ge=0, le=100, description="Position papillon des gaz (%)")
    intake_air_temp: Optional[float] = Field(25, ge=-40, le=100, description="Température air admission (°C)")
    ambient_air_temp: Optional[float] = Field(20, ge=-40, le=60, description="Température ambiante (°C)")
    mass_air_flow: Optional[float] = Field(5.0, ge=0, le=1000, description="Débit d'air massique (g/s)")
    intake_manifold_pressure: Optional[float] = Field(100, ge=0, le=300,
                                                      description="Pression collecteur admission (kPa)")


class PredictionResponse(BaseModel):
    """Réponse de l'API"""
    predicted_temp: float = Field(..., description="Température prédite dans 10 minutes (°C)")
    risk_message: str = Field(..., description="Message d'alerte")
    recommendations: List[str] = Field(..., description="Recommandations d'action")


class OverheatPredictor:
    """Classe de prédiction optimisée pour l'API"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_loaded = False

    def load_model(self, filename: str = "api_overheat_model.pkl"):
        """Charge le modèle pré-entraîné depuis le dossier models/ à la racine du projet"""
        try:

            model_path = os.path.join(MODELS_DIR, filename)
            print(f"🔍 Chargement du modèle depuis : {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modèle non trouvé: {model_path}")

            model_data = joblib.load(model_path)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']
            self.is_loaded = True
            logger.info(f"✅ Modèle chargé avec succès: {model_path}")

        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur chargement modèle: {e}")

    def create_features(self, obd_data: OBDData) -> pd.DataFrame:
        """Crée les features à partir des données OBD-II"""
        try:
            # Extraction des valeurs de base
            coolant_temp = obd_data.engine_coolant_temp
            rpm = obd_data.engine_rpm
            speed = obd_data.vehicle_speed
            throttle = obd_data.throttle_position
            intake_temp = obd_data.intake_air_temp
            ambient_temp = obd_data.ambient_air_temp

            # Calcul des features dérivées
            engine_load_factor = (rpm / 6000) * (throttle / 100)
            temp_rpm_ratio = coolant_temp / max(rpm, 1)  # Éviter division par zéro
            temp_ambient_diff = coolant_temp - ambient_temp

            # Features de base
            features = {
                'engine_coolant_temp': coolant_temp,
                'engine_rpm': rpm,
                'vehicle_speed': speed,
                'throttle_position': throttle,
                'intake_air_temp': intake_temp,
                'ambient_air_temp': ambient_temp,
                'condition_encoded': 0,  # Normal par défaut

                # Features calculées
                'engine_load_factor': engine_load_factor,
                'temp_rpm_ratio': temp_rpm_ratio,
                'temp_ambient_diff': temp_ambient_diff,

                # Features temporelles (approximées pour temps réel)
                'coolant_temp_ma_30s': coolant_temp,
                'coolant_temp_ma_60s': coolant_temp,
                'coolant_temp_ma_300s': coolant_temp,
                'coolant_temp_trend_1min': 0,  # Pas d'historique en temps réel

                # Indicateurs de risque
                'high_temp_indicator': 1 if coolant_temp > 90 else 0,
                'extreme_load': 1 if (rpm > 4000 and throttle > 80) else 0,
                'heat_stress_score': (
                        (coolant_temp - 80) * 0.3 +
                        (engine_load_factor * 20) +
                        (temp_ambient_diff * 0.2)
                )
            }

            # Créer DataFrame avec toutes les features nécessaires
            df = pd.DataFrame([features])

            # S'assurer que toutes les features du modèle sont présentes
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0  # Valeur par défaut

            # Réorganiser selon l'ordre d'entraînement
            df = df[self.feature_names]

            return df

        except Exception as e:
            logger.error(f"Erreur création features: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Erreur traitement données: {str(e)}")

    def predict(self, features_df: pd.DataFrame, current_temp: float) -> dict:
        """Effectue la prédiction de surchauffe"""
        try:
            if not self.is_loaded:
                raise HTTPException(status_code=500, detail="Modèle non chargé")

            # Normalisation
            X_scaled = self.scalers['robust'].transform(features_df)

            # Prédictions des modèles
            pred_main = self.models['xgb_main'].predict(X_scaled)[0]
            pred_rf = self.models['rf'].predict(X_scaled)[0]

            # Prédiction ensemble
            if 'xgb_high_temp' in self.models and current_temp > 90:
                pred_high = self.models['xgb_high_temp'].predict(X_scaled)[0]
                predicted_temp = 0.4 * pred_main + 0.6 * pred_high
            else:
                predicted_temp = 0.6 * pred_main + 0.4 * pred_rf

            # Contraintes réalistes
            predicted_temp = np.clip(predicted_temp, current_temp - 5, 118)

            # Si température actuelle déjà élevée, ajuster la prédiction
            if current_temp > predicted_temp and current_temp > 95:
                predicted_temp = max(predicted_temp, current_temp + 1)

            # Détermination du niveau de risque
            if predicted_temp >= CONFIG['temp_critical']:
                risk_level = 3
            elif predicted_temp >= CONFIG['temp_warning']:
                risk_level = 2
            elif predicted_temp >= CONFIG['temp_attention']:
                risk_level = 1
            else:
                risk_level = 0

            # Ajustements basés sur température actuelle
            if current_temp >= 100:
                risk_level = max(risk_level, 3)
            elif current_temp >= 95:
                risk_level = max(risk_level, 2)
            elif current_temp >= 90:
                risk_level = max(risk_level, 1)

            return {
                'predicted_temp': float(predicted_temp),
                'risk_level': risk_level
            }

        except Exception as e:
            logger.error(f"Erreur prédiction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")

    def get_risk_message_and_recommendations(self, risk_level: int, current_temp: float) -> tuple:
        """Retourne le message de risque et les recommandations"""

        if risk_level == 3:
            message = "🚨 CRITIQUE: Surchauffe imminente - ARRÊTEZ-VOUS IMMÉDIATEMENT !"
            recommendations = [
                "ARRÊTEZ-VOUS IMMÉDIATEMENT et éteignez le moteur",
                "Attendez le refroidissement complet avant de redémarrer",
                "Vérifiez le niveau de liquide de refroidissement",
                "Contactez un mécanicien d'urgence",
                "Ne conduisez pas tant que le problème n'est pas résolu"
            ]

        elif risk_level == 2:
            message = "🔥 ALERTE: Risque de surchauffe détecté - Action immédiate requise !"
            recommendations = [
                "Réduisez immédiatement la charge du moteur",
                "Diminuez votre vitesse et évitez les accélérations",
                "Activez le chauffage au maximum pour évacuer la chaleur",
                "Surveillez constamment la température",
                "Préparez-vous à vous arrêter si la température continue de monter"
            ]

        elif risk_level == 1:
            message = "⚠️ ATTENTION: Température élevée - Surveillez attentivement"
            recommendations = [
                "Surveillez la température de près",
                "Évitez les accélérations brusques et les hauts régimes",
                "Réduisez l'utilisation de la climatisation",
                "Vérifiez prochainement le système de refroidissement",
                "Conduisez de manière plus douce"
            ]

        else:
            message = "✅ Température normale - Conduite normale possible"
            recommendations = [
                "Température dans la plage normale",
                "Continuez votre conduite normalement",
                "Surveillez périodiquement les paramètres moteur"
            ]

        return message, recommendations


# Instance globale du prédicteur
predictor = OverheatPredictor()

async def startup_event():
    """Initialisation au démarrage de l'API"""
    try:
        # Construire le chemin vers models/ à la racine
        FILE_DIR = os.path.dirname(__file__)
        PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
        MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

        # Liste des fichiers à essayer
        model_files = [
            "enhanced_overheat_predictor_v2.pkl",
            "enhanced_overheat_predictor.pkl",
            "api_overheat_model.pkl"
        ]

        # Tenter de charger le premier existant
        for filename in model_files:
            model_path = os.path.join(MODELS_DIR, filename)
            print(f"🔍 Vérification existence : {model_path}")
            if os.path.exists(model_path):
                predictor.load_model(filename)
                logger.info(f"✅ API démarrée avec le modèle: {filename}")
                return

        # Aucun trouvé
        logger.warning("⚠️ Aucun modèle trouvé - API en mode dégradé")

    except Exception as e:
        logger.error(f"❌ Erreur démarrage API: {e}")
        raise

@router.post("/predict_OverHeat", response_model=PredictionResponse)
async def predict_overheat(obd_data: OBDData ,  _: None = Depends(verify_api_key )):


    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible - Service temporairement indisponible"
        )

    try:
        # Validation des données d'entrée
        current_temp = obd_data.engine_coolant_temp

        # Vérifications de cohérence
        if current_temp < -40 or current_temp > 150:
            raise HTTPException(status_code=400, detail="Température liquide de refroidissement hors limites")

        if obd_data.engine_rpm < 0 or obd_data.engine_rpm > 8000:
            raise HTTPException(status_code=400, detail="Régime moteur hors limites")

        # Création des features
        features_df = predictor.create_features(obd_data)

        # Prédiction
        result = predictor.predict(features_df, current_temp)

        # Message et recommandations
        risk_message, recommendations = predictor.get_risk_message_and_recommendations(
            result['risk_level'], current_temp
        )

        # Réponse formatée
        response = PredictionResponse(
            predicted_temp=round(result['predicted_temp'], 1),
            risk_message=risk_message,
            recommendations=recommendations
        )

        logger.info(f"Prédiction réussie - Temp actuelle: {current_temp}°C, "
                    f"Prédite: {result['predicted_temp']:.1f}°C, Risque: {result['risk_level']}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")




