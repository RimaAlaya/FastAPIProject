import os

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import logging
from typing import List, Optional
from scipy import stats
from scipy.fft import fft
import warnings
from joblib import load
from fastapi import APIRouter
from app.security import verify_api_key  # plus de boucle

warnings.filterwarnings('ignore')

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Global variables pour les modèles et preprocessing
model = None
scaler = None
label_encoder = None

# Obtenir le chemin du dossier courant (là où ce fichier est exécuté)
# dossier contenant ce fichier (app/api)
APP_API_DIR = os.path.dirname(__file__)

# remonter deux fois : app/api → app → FastAPIProject4
PROJECT_ROOT = os.path.abspath(os.path.join(APP_API_DIR, "..", ".."))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

class WindowedOBDData(BaseModel):
    """Données OBD-II agrégées sur 5 secondes (PIDs Standard)"""
    engine_load_mean: float
    engine_load_std: float
    engine_load_max: float
    engine_load_min: float
    engine_rpm_mean: float
    engine_rpm_std: float
    engine_rpm_max: float
    engine_rpm_min: float
    vehicle_speed_mean: float
    vehicle_speed_std: float
    vehicle_speed_max: float
    vehicle_speed_min: float
    coolant_temp_mean: float
    coolant_temp_std: float
    coolant_temp_max: float
    coolant_temp_min: float
    map_pressure_mean: float
    map_pressure_std: float
    map_pressure_max: float
    map_pressure_min: float
    maf_rate_mean: float
    maf_rate_std: float
    maf_rate_max: float
    maf_rate_min: float
    intake_air_temp_mean: float
    intake_air_temp_std: float
    intake_air_temp_max: float
    intake_air_temp_min: float


class WindowedSmartphoneData(BaseModel):
    """Données capteurs smartphone agrégées sur 5 secondes"""
    altitude_change_mean: float
    altitude_change_std: float
    altitude_change_max: float
    altitude_change_min: float
    vertical_acceleration_mean: float
    vertical_acceleration_std: float
    vertical_acceleration_max: float
    vertical_acceleration_min: float
    gps_speed_variation_mean: float
    gps_speed_variation_std: float
    gps_speed_variation_max: float
    gps_speed_variation_min: float
    longitudinal_acceleration_mean: float  # Calculé au niveau smartphone
    longitudinal_acceleration_std: float
    longitudinal_acceleration_max: float
    longitudinal_acceleration_min: float


class WindowedDrivingDataPoint(BaseModel):
    """Point de données agrégé sur 5 secondes"""
    window_start_timestamp: float
    window_end_timestamp: float
    obd: WindowedOBDData
    smartphone: WindowedSmartphoneData
    trip_id: str
    vehicle_id: str


class DrivingSequence(BaseModel):
    """Séquence de 24 fenêtres de 5 secondes (total: 2 minutes)"""
    windowed_data_points: List[WindowedDrivingDataPoint]


class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    driving_style: str
    confidence: float
    probabilities: dict

@router.on_event("startup")
async def load_models():
    """Charger le modèle et les preprocessors au démarrage"""
    global model, scaler, label_encoder

    try:
        # Charger le modèle TCN
        model_path = os.path.join(MODELS_DIR, "best_enhanced_tcn_3_classes.keras")
        model = tf.keras.models.load_model(model_path)

        # Charger les preprocessors
        scaler = load(os.path.join(MODELS_DIR, "robust_scaler (3).pkl"))
        label_encoder = load(os.path.join(MODELS_DIR, "label_encoder (3).pkl"))

        print("✅ Modèle et preprocessors chargés avec succès")

    except Exception as e:
        print(f"❌ Erreur lors du chargement des modèles: {e}")
        raise


def calculate_derived_features_from_windowed(df_windowed):
    """
    Calcule les features dérivées à partir des données déjà agrégées par fenêtres
    """
    logger.info("🔧 Calcul des features dérivées à partir des données agrégées...")

    df_features = df_windowed.copy()

    # ===== BASIC DERIVED FEATURES adaptées aux données agrégées =====
    # RPM efficiency (utilise les moyennes)
    df_features['RPM_Speed_Ratio_mean'] = df_features['EngineRPM_mean'] / (
                df_features['VehicleSpeedInstantaneous_mean'] + 1e-6)

    # Engine stress indicators
    df_features['Engine_Stress_mean'] = df_features['EngineLoad_mean'] * df_features['EngineRPM_mean'] / 1000
    df_features['Power_Demand_mean'] = df_features['EngineLoad_mean'] * df_features['VehicleSpeedInstantaneous_mean']

    # Fuel efficiency (utilise MAF pour estimation)
    fuel_density = 0.74  # kg/L essence
    df_features['FuelConsumptionAverage_mean'] = (df_features['MAFRate_mean'] * 3.6) / (
                fuel_density * 1000)  # L/100km approx
    df_features['Fuel_Efficiency_mean'] = df_features['VehicleSpeedInstantaneous_mean'] / (
            df_features['FuelConsumptionAverage_mean'] + 1e-6)

    # Temperature ratios
    df_features['Temp_Efficiency_mean'] = df_features['EngineCoolantTemperature_mean'] / (
            df_features['IntakeAirTemperature_mean'] + 273.15)

    # ===== CONTEXT-AWARE FEATURES basées sur les moyennes =====
    df_features['is_accelerating'] = (df_features['LongitudinalAcceleration_mean'] > 0.5).astype(int)
    df_features['is_braking'] = (df_features['LongitudinalAcceleration_mean'] < -0.5).astype(int)
    df_features['is_cruising'] = ((abs(df_features['LongitudinalAcceleration_mean']) <= 0.5) &
                                  (df_features['VehicleSpeedInstantaneous_mean'] > 5)).astype(int)
    df_features['is_stopped'] = (df_features['VehicleSpeedInstantaneous_mean'] <= 1).astype(int)

    # ===== AGGRESSIVE DRIVING INDICATORS =====
    df_features['Harsh_Acceleration'] = (df_features['LongitudinalAcceleration_max'] > 2.5).astype(int)
    df_features['Harsh_Braking'] = (df_features['LongitudinalAcceleration_min'] < -2.5).astype(int)

    # Speed variability directement des données agrégées
    df_features['VehicleSpeedAverage_mean'] = df_features['VehicleSpeedInstantaneous_mean']
    df_features['VehicleSpeedVariance_mean'] = df_features['VehicleSpeedInstantaneous_std'] ** 2
    df_features['Speed_Variability_mean'] = df_features['VehicleSpeedVariance_mean'] / (
                df_features['VehicleSpeedAverage_mean'] + 1e-6)

    # ===== TEMPORAL FEATURES sur la séquence de 24 fenêtres =====
    window_sizes = [5, 10, 20]

    for window in window_sizes:
        if len(df_features) >= window:
            # Speed patterns
            df_features[f'Speed_Rolling_Mean_{window}'] = df_features['VehicleSpeedInstantaneous_mean'].rolling(window,
                                                                                                                min_periods=1).mean()
            df_features[f'Speed_Rolling_Std_{window}'] = df_features['VehicleSpeedInstantaneous_mean'].rolling(window,
                                                                                                               min_periods=1).std().fillna(
                0)

            # Acceleration patterns
            df_features[f'Accel_Rolling_Mean_{window}'] = df_features['LongitudinalAcceleration_mean'].rolling(window,
                                                                                                               min_periods=1).mean()
            df_features[f'Accel_Rolling_Std_{window}'] = df_features['LongitudinalAcceleration_mean'].rolling(window,
                                                                                                              min_periods=1).std().fillna(
                0)

            # RPM patterns
            df_features[f'RPM_Rolling_Mean_{window}'] = df_features['EngineRPM_mean'].rolling(window,
                                                                                              min_periods=1).mean()
            df_features[f'RPM_Rolling_Std_{window}'] = df_features['EngineRPM_mean'].rolling(window,
                                                                                             min_periods=1).std().fillna(
                0)

    # Engine behavior
    df_features['RPM_Variability_mean'] = df_features['EngineRPM_mean'] / (df_features['RPM_Rolling_Mean_10'] + 1e-6)

    # ===== STATISTICAL MOMENTS sur la séquence =====
    key_vars = ['VehicleSpeedInstantaneous_mean', 'LongitudinalAcceleration_mean', 'EngineRPM_mean', 'EngineLoad_mean']

    for var in key_vars:
        if var in df_features.columns and len(df_features) >= 20:
            df_features[f'{var}_Skewness_Trip'] = df_features[var].rolling(20, min_periods=1).apply(
                lambda x: stats.skew(x.dropna()) if len(x.dropna()) > 1 else 0
            ).fillna(0)
            df_features[f'{var}_Kurtosis_Trip'] = df_features[var].rolling(20, min_periods=1).apply(
                lambda x: stats.kurtosis(x.dropna()) if len(x.dropna()) > 1 else 0
            ).fillna(0)

    # ===== SPECTRAL FEATURES sur la séquence =====
    if len(df_features) > 10:
        for var in ['VehicleSpeedInstantaneous_mean', 'LongitudinalAcceleration_mean', 'EngineRPM_mean']:
            if var in df_features.columns:
                signal_clean = df_features[var].fillna(df_features[var].median())
                if len(signal_clean) > 1:
                    fft_values = np.abs(fft(signal_clean))[:len(signal_clean) // 2]
                    if len(fft_values) > 0:
                        df_features[f'{var}_Spectral_Centroid'] = np.mean(fft_values)
                        df_features[f'{var}_Spectral_Spread'] = np.std(fft_values)
                        df_features[f'{var}_Spectral_Skewness'] = stats.skew(fft_values) if len(fft_values) > 1 else 0
                        df_features[f'{var}_Spectral_Kurtosis'] = stats.kurtosis(fft_values) if len(
                            fft_values) > 1 else 0

    # ===== HANDLE NaN AND INFINITE VALUES =====
    numeric_cols = df_features.select_dtypes(include=np.number).columns
    cols_to_fill = [col for col in numeric_cols if col not in ['trip_id', 'vehicle_id']]

    # Replace infinite values with NaN first
    df_features[cols_to_fill] = df_features[cols_to_fill].replace([np.inf, -np.inf], np.nan)

    # Fill NaN values with median
    df_features[cols_to_fill] = df_features[cols_to_fill].fillna(df_features[cols_to_fill].median())

    logger.info(f"✅ Features calculées: {df_features.shape}")
    return df_features


def create_sequences_for_prediction(df_features, sequence_length=24):
    """
    Créer les séquences pour prédiction directement à partir des 24 fenêtres
    """
    logger.info(f"🔄 Création des séquences TCN (longueur={sequence_length})...")

    # Features critiques agrégées (adaptées aux données pré-agrégées)
    critical_features_agg = [
        'VehicleSpeedInstantaneous_mean', 'LongitudinalAcceleration_mean', 'EngineRPM_mean',
        'EngineLoad_mean', 'VehicleSpeedAverage_mean', 'VehicleSpeedVariance_mean',
        'FuelConsumptionAverage_mean', 'RPM_Speed_Ratio_mean', 'Engine_Stress_mean'
    ]

    # Sélectionner les features numériques
    exclude_cols = ['trip_id', 'vehicle_id']
    all_numeric_cols = df_features.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [col for col in all_numeric_cols if col not in exclude_cols]

    # Utiliser les features critiques si disponibles
    available_critical_agg = [col for col in critical_features_agg if col in feature_cols]

    if len(available_critical_agg) >= 3:
        other_features = [col for col in feature_cols if col not in available_critical_agg]
        feature_cols = available_critical_agg + other_features[:max(0, 50 - len(available_critical_agg))]
    else:
        feature_cols = feature_cols[:50]

    logger.info(f"Utilisation de {len(feature_cols)} features")

    # Préparer les données
    X_data = df_features[feature_cols].values

    # Imputation des NaN
    imputer = SimpleImputer(strategy='median')
    X_data = imputer.fit_transform(X_data)

    # Vérifier que nous avons exactement 24 fenêtres
    if len(X_data) != sequence_length:
        raise ValueError(f"Nombre incorrect de fenêtres: {len(X_data)} != {sequence_length}")

    # Créer la séquence
    X_sequences = np.array([X_data])

    logger.info(f"✅ Séquence créée: {X_sequences.shape}")

    return X_sequences, feature_cols


def convert_windowed_input_to_dataframe(sequence_data: DrivingSequence):
    """
    Convertir les données agrégées d'entrée au format DataFrame
    """
    logger.info("🔄 Conversion des données agrégées d'entrée...")

    data_rows = []

    for point in sequence_data.windowed_data_points:
        # Mapper toutes les données agrégées selon les noms attendus par le modèle
        row = {
            # OBD Data aggregated
            'VehicleSpeedInstantaneous_mean': point.obd.vehicle_speed_mean,
            'VehicleSpeedInstantaneous_std': point.obd.vehicle_speed_std,
            'VehicleSpeedInstantaneous_max': point.obd.vehicle_speed_max,
            'VehicleSpeedInstantaneous_min': point.obd.vehicle_speed_min,
            'EngineRPM_mean': point.obd.engine_rpm_mean,
            'EngineRPM_std': point.obd.engine_rpm_std,
            'EngineRPM_max': point.obd.engine_rpm_max,
            'EngineRPM_min': point.obd.engine_rpm_min,
            'EngineLoad_mean': point.obd.engine_load_mean,
            'EngineLoad_std': point.obd.engine_load_std,
            'EngineLoad_max': point.obd.engine_load_max,
            'EngineLoad_min': point.obd.engine_load_min,
            'EngineCoolantTemperature_mean': point.obd.coolant_temp_mean,
            'EngineCoolantTemperature_std': point.obd.coolant_temp_std,
            'EngineCoolantTemperature_max': point.obd.coolant_temp_max,
            'EngineCoolantTemperature_min': point.obd.coolant_temp_min,
            'IntakeAirTemperature_mean': point.obd.intake_air_temp_mean,
            'IntakeAirTemperature_std': point.obd.intake_air_temp_std,
            'IntakeAirTemperature_max': point.obd.intake_air_temp_max,
            'IntakeAirTemperature_min': point.obd.intake_air_temp_min,
            'MAFRate_mean': point.obd.maf_rate_mean,
            'MAFRate_std': point.obd.maf_rate_std,
            'MAFRate_max': point.obd.maf_rate_max,
            'MAFRate_min': point.obd.maf_rate_min,
            'MAPPressure_mean': point.obd.map_pressure_mean,
            'MAPPressure_std': point.obd.map_pressure_std,
            'MAPPressure_max': point.obd.map_pressure_max,
            'MAPPressure_min': point.obd.map_pressure_min,

            # Smartphone Data aggregated
            'LongitudinalAcceleration_mean': point.smartphone.longitudinal_acceleration_mean,
            'LongitudinalAcceleration_std': point.smartphone.longitudinal_acceleration_std,
            'LongitudinalAcceleration_max': point.smartphone.longitudinal_acceleration_max,
            'LongitudinalAcceleration_min': point.smartphone.longitudinal_acceleration_min,
            'AltitudeChange_mean': point.smartphone.altitude_change_mean,
            'AltitudeChange_std': point.smartphone.altitude_change_std,
            'AltitudeChange_max': point.smartphone.altitude_change_max,
            'AltitudeChange_min': point.smartphone.altitude_change_min,
            'VerticalAcceleration_mean': point.smartphone.vertical_acceleration_mean,
            'VerticalAcceleration_std': point.smartphone.vertical_acceleration_std,
            'VerticalAcceleration_max': point.smartphone.vertical_acceleration_max,
            'VerticalAcceleration_min': point.smartphone.vertical_acceleration_min,
            'GPSSpeedVariation_mean': point.smartphone.gps_speed_variation_mean,
            'GPSSpeedVariation_std': point.smartphone.gps_speed_variation_std,
            'GPSSpeedVariation_max': point.smartphone.gps_speed_variation_max,
            'GPSSpeedVariation_min': point.smartphone.gps_speed_variation_min,

            # Metadata
            'trip_id': point.trip_id,
            'vehicle_id': point.vehicle_id,
            'window_start_timestamp': point.window_start_timestamp,
            'window_end_timestamp': point.window_end_timestamp
        }

        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    logger.info(f"✅ DataFrame créé: {df.shape}")

    return df


@router.post("/predict_driver_behaviour", response_model=PredictionResponse)
async def predict_driving_style(sequence_data: DrivingSequence ,  _: None = Depends(verify_api_key )):
    """
    Prédire le style de conduite à partir d'une séquence de 24 fenêtres de 5 secondes
    """
    try:
        logger.info("🚀 Début de la prédiction...")

        # Vérifications
        if len(sequence_data.windowed_data_points) != 24:
            raise HTTPException(
                status_code=400,
                detail=f"Exactement 24 fenêtres de 5 secondes requises, reçu: {len(sequence_data.windowed_data_points)}"
            )

        if model is None or scaler is None or label_encoder is None:
            raise HTTPException(status_code=500, detail="Modèles non chargés")

        # 1. Convertir en DataFrame (données déjà agrégées)
        df_windowed = convert_windowed_input_to_dataframe(sequence_data)

        # 2. Calculer les features dérivées à partir des données agrégées
        df_features = calculate_derived_features_from_windowed(df_windowed)

        # 3. Créer les séquences directement (pas besoin de re-fenêtrer)
        X_sequences, feature_names = create_sequences_for_prediction(df_features)

        # 4. Normalisation (identique à l'entraînement)
        n_sequences, seq_len, n_features = X_sequences.shape
        X_reshaped = X_sequences.reshape(-1, n_features)
        X_scaled = scaler.transform(X_reshaped)
        X_sequences_scaled = X_scaled.reshape(n_sequences, seq_len, n_features)

        # 5. Prédiction
        predictions = model.predict(X_sequences_scaled, verbose=0)

        # 6. Interpréter les résultats
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        # Décoder la classe prédite
        predicted_style = label_encoder.inverse_transform([predicted_class_idx])[0]

        # Créer le dictionnaire des probabilités
        probabilities = {}
        for i, class_name in enumerate(label_encoder.classes_):
            probabilities[class_name] = float(predictions[0][i])

        logger.info(f"✅ Prédiction: {predicted_style} (confiance: {confidence:.3f})")

        return PredictionResponse(
            driving_style=predicted_style,
            confidence=confidence,
            probabilities=probabilities
        )

    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))





