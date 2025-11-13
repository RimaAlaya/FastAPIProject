# 🚗 Intelligent Vehicle Assistant API

A comprehensive FastAPI-based system providing real-time vehicle diagnostics, driver behavior analysis, and environmental impact monitoring using advanced machine learning models.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)


## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Machine Learning Models](#machine-learning-models)
- [API Endpoints](#api-endpoints)
- [Installation](#installation)
- [Usage](#usage)
- [Security](#security)
- [Research Publication](#research-publication)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## 🎯 Overview

This API system integrates multiple AI-powered diagnostic modules that analyze OBD-II vehicle data and smartphone sensors to provide:
- **Real-time engine health monitoring**
- **Predictive maintenance alerts**
- **Driver behavior classification**
- **Environmental impact assessment**
- **Personalized weekly performance reports**

## ✨ Features

### 🔥 Engine Overheat Prediction
- Predicts coolant temperature 10 minutes in advance
- Hybrid ensemble model (XGBoost + Random Forest)
- Risk classification with actionable recommendations
- **Accuracy: 95.7%** | **MAE: 0.9°C** | **R²: 0.952**

### 🛞 Tire Pressure Monitoring
- Detects underinflation through vibration pattern analysis
- Convolutional Autoencoder for anomaly detection
- Supports both direct TPMS (PID 0x76) and AI-based inference
- **Recall: 94%** for critical anomaly detection

### 👤 Driver Behavior Classification
- Temporal Convolutional Network (TCN) architecture
- Classifies driving styles: Peaceful, Aggressive, Eco-Friendly
- Real-time analysis over 2-minute sequences (24 x 5s windows)
- **Accuracy: 98.73%** - Surpasses state-of-the-art benchmarks

### 🌍 CO₂ Emissions Calculator
- Real-time emission rate calculation (g/s and g/km)
- Multiple calculation methods: MAF, Fuel Rate, MAP-based estimation
- Fuel type auto-detection (gasoline/diesel)
- Confidence scoring for each calculation method

### 📊 Weekly Performance Reports
- Comprehensive driver scoring system (0-1000 points)
- Environmental impact analysis with carbon footprint
- Safety metrics and performance trends
- Personalized recommendations and achievements
- Community rankings and comparative analytics

## 🧠 Machine Learning Models

### Driving Style Classification (TCN)
Our **Temporal Convolutional Network** achieves state-of-the-art performance on the Mafalda dataset:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Our TCN** | **98.73%** | **98.80%** | **98.73%** | **98.75%** |
| Al-Refai et al. (2024) | 92.00% | 86.00% | 91.00% | 85.00% |
| Random Forest | 97.94% | 97.97% | 97.94% | 97.94% |
| Standard ANN | 97.66% | 97.69% | 97.66% | 97.66% |

**Key Innovations:**
- Dilated causal convolutions for long-range temporal dependencies
- Integrated attention mechanism for dynamic feature weighting
- Spatial dropout and residual connections for robustness
- Enhanced global pooling (average + max)

> 📄 **Research Paper**: "Advanced Driving Style Classification using Temporal Convolutional Networks on Vehicle Data" - Full methodology and results available in project documentation.

### Engine Overheat Predictor
**Hybrid Ensemble Architecture:**
- Primary XGBoost model for general conditions
- Specialized XGBoost for high-temperature scenarios (>90°C)
- Random Forest for decision reinforcement
- Dynamic weighting based on current temperature

**Training Details:**
- Features: 17 engineered features from OBD-II PIDs
- Dataset: Synthetic + real-world driving conditions
- Optimization: Hyperparameter tuning with cross-validation

### Tire Anomaly Detector
**Convolutional Autoencoder (Conv1D):**
- Architecture: 3 encoder + 3 decoder layers
- Compression ratio: 11.25:1 (900 → 80 features)
- Optimal reconstruction threshold: 1.113
- Input: 100-timestep sequences of 9 vibration features

**Performance Focus:**
- Prioritizes high recall (94%) for safety-critical detection
- Acceptable false positive rate to prevent missed anomalies
- Trained on synthetic anomaly data with augmentation

## 🔌 API Endpoints

### Engine Diagnostics
```http
POST /predict_OverHeat
```
Predicts engine temperature 10 minutes ahead with risk assessment.

### Tire Monitoring
```http
POST /detect_tire_status
```
Detects tire underinflation using sensor data or direct TPMS readings.

### Driver Analysis
```http
POST /predict_driver_behaviour
```
Classifies driving style from 2-minute OBD-II + smartphone data sequences.

### Emissions
```http
POST /calculate_co2
```
Calculates real-time CO₂ emissions with multiple estimation methods.

### Reports
```http
POST /generate_weekly_report
POST /calculate_driver_score_only
GET /community_stats
```
Generates comprehensive weekly performance analytics and rankings.

## 🚀 Installation

### Prerequisites
- Python 3.11+
- Docker (optional)
- 8GB+ RAM recommended

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/vehicle-assistant-api.git
cd vehicle-assistant-api
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.local.example .env.local
# Edit .env.local with your credentials:
# BASIC_AUTH_USERNAME=admin
# BASIC_AUTH_PASSWORD=your_password
# API_KEY=your_api_key
```

5. **Download models** (if not using Git LFS)
```bash
# Models are tracked with Git LFS
git lfs pull
```

### Docker Deployment

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8085`

## 📖 Usage

### Authentication

All API endpoints require an API key in the header:
```bash
curl -X POST "http://localhost:8085/predict_OverHeat" \
  -H "x-api-key: your_api_key" \
  -H "Content-Type: application/json" \
  -d @request.json
```

### Interactive Documentation

Access Swagger UI (protected by Basic Auth):
```
http://localhost:8085/docs
```

### Example Request - Driver Behavior
```json
{
  "windowed_data_points": [
    {
      "window_start_timestamp": 1638360000.0,
      "window_end_timestamp": 1638360005.0,
      "obd": {
        "engine_load_mean": 45.2,
        "engine_rpm_mean": 2100.0,
        "vehicle_speed_mean": 60.5,
        // ... (32 total OBD features)
      },
      "smartphone": {
        "altitude_change_mean": 0.5,
        "vertical_acceleration_mean": 0.1,
        "longitudinal_acceleration_mean": 0.3,
        // ... (16 total smartphone features)
      },
      "trip_id": "trip_001",
      "vehicle_id": "vehicle_123"
    }
    // ... (24 total windows for 2-minute sequence)
  ]
}
```

### Example Response
```json
{
  "driving_style": "peaceful_style",
  "confidence": 0.956,
  "probabilities": {
    "aggressive_style": 0.022,
    "peaceful_style": 0.956,
    "eco_friendly": 0.022
  }
}
```

## 🔒 Security

### Multi-Layer Security
1. **API Key Authentication**: Required header `x-api-key` for all endpoints
2. **Basic Auth**: Protects Swagger/ReDoc documentation
3. **Environment Variables**: Sensitive credentials stored in `.env.local`
4. **Docker Isolation**: Containerized deployment for production

### Best Practices
- Never commit `.env.local` to version control
- Rotate API keys regularly
- Use HTTPS in production
- Implement rate limiting (recommended: 100 req/hour per user)

## 📄 Research Publication

Our **TCN-based driving style classification** model has been documented in a research paper demonstrating:
- **1.07% improvement** over Al-Refai et al. (2024) baseline
- Perfect recall (100%) for aggressive driving detection
- Superior temporal dependency modeling
- Robust generalization across diverse driving scenarios

> The model and methodology are described in: *"Advanced Driving Style Classification using Temporal Convolutional Networks on Vehicle Data"* (2024)

**Dataset**: Mafalda public dataset - time-series OBD-II data with annotated driving styles

## 📁 Project Structure

```
FastAPIProject4/
├── app/
│   ├── api/
│   │   ├── CO2emission.py          # Emissions calculator
│   │   ├── DriverProfile.py        # TCN-based driver classification
│   │   ├── OverheatEngine.py       # Hybrid ensemble predictor
│   │   ├── TirePressure.py         # Autoencoder anomaly detector
│   │   └── WeeklyReport.py         # Analytics & reports
│   ├── auth.py                     # Basic Auth for docs
│   ├── security.py                 # API key verification
│   └── main.py                     # FastAPI application
├── models/
│   ├── best_enhanced_tcn_3_classes.keras
│   ├── api_overheat_model.pkl
│   ├── tire_anomaly_detector.keras
│   └── *.pkl                       # Scalers & encoders
├── .env.local                      # Environment config (not tracked)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🎯 Why This Project Matters

This system prevents engine failures before they happen and helps drivers reduce their carbon footprint by up to 15% through intelligent behavior analysis. Built end-to-end from research to production, it demonstrates how AI can make automotive systems safer and more sustainable.

### What I Bring to the Table
- **ML Research → Production**: Published research (98.73% accuracy) deployed as production APIs
- **Full-Stack ML**: Data engineering, model training, API development, Docker deployment
- **Tech Stack**: Python • FastAPI • TensorFlow • XGBoost • Docker • Git LFS

### Highlights
✅ 5 ML models in production  
✅ State-of-the-art driving classification (beats benchmarks by 1.07%)  
✅ Real-time inference (<1s response time)  
✅ Research paper: [`docs/research_paper.pdf`](docs/research_paper.pdf)

---

## 📬 Let's Connect

**Rima ALAYA**  
🎓 National School of Engineers of Carthage  
📧 rima.alaya@enicar.ucar.tn  
🔗 [LinkedIn](https://linkedin.com/in/rima-alaya) • [GitHub](https://github.com/RimaAlaya)

*Open to opportunities in ML Engineering, Backend Development, or Automotive AI*

> 💡 Full training notebooks and additional projects available upon request!

## 🙏 Acknowledgments

- **Supervisors**: Dr. Faouzi JAIDI (Sup'Com), Aymen TOUHENT (XELERO)
- **Dataset**: Mafalda public dataset for driving behavior research
- **Research**: Based on thesis work in Intelligent Transport Systems (ITS)
- **Technologies**: FastAPI, TensorFlow, scikit-learn, XGBoost

---

**⚠️ Note**: This API is designed for research and development purposes. For production deployment in safety-critical automotive systems, additional validation and certification are required.

**🔧 Model Training**: The training pipelines, preprocessing scripts, and experimental notebooks are maintained in a separate repository. Contact the author for access to the complete research codebase.