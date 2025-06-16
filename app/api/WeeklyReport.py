from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
import math
from enum import Enum
from fastapi import APIRouter
from app.security import verify_api_key  # plus de boucle


router = APIRouter()


class DrivingStyle(str, Enum):
    PEACEFUL = "peaceful_style"
    AGGRESSIVE = "aggressive_style"
    ECO_FRIENDLY = "eco_friendly"
    SPORTIVE = "sportive"


class TripData(BaseModel):
    trip_id: str
    timestamp: datetime
    driving_style: DrivingStyle
    driving_style_confidence: float = Field(ge=0.0, le=1.0)
    co2_emission_rate: float  # g/s - Calculé par l'API CO₂
    co2_per_km: Optional[float] = None  # g/km - Calculé par l'API CO₂
    distance_km: float
    duration_minutes: float
    avg_speed_kmh: float
    max_speed_kmh: float
    fuel_consumption_l: Optional[float] = None


class WeeklyDrivingData(BaseModel):
    user_id: str
    week_start: datetime
    week_end: datetime
    trips: List[TripData]
    total_distance_km: float
    total_duration_hours: float

@dataclass
class EnvironmentalImpact:
    co2_kg_week: float
    co2_vs_average: float  # % par rapport à la moyenne
    trees_needed: float  # Nombre d'arbres pour compenser
    fuel_cost_euros: float
    eco_rank_percentile: float
    carbon_footprint_rating: str
    eco_score: float
    efficiency_level: str
    cost_savings_potential: float




@dataclass
class SafetyMetrics:
    aggressive_episodes: int
    smooth_driving_score: float  # 0-100
    speed_violations: int
    safety_rank_percentile: float


@dataclass
class PerformanceMetrics:
    efficiency_score: float  # 0-100
    consistency_score: float  # 0-100
    improvement_trend: float  # -100 à +100
    peak_performance_moments: int


@dataclass
class SmartInsights:
    dominant_style: DrivingStyle
    style_confidence: float
    style_evolution: str  # "improving", "stable", "declining"
    best_day: str
    worst_day: str
    recommendations: List[str]


class DriverScore(BaseModel):
    overall_score: float = Field(ge=0, le=1000)  # Score sur 1000
    eco_score: float = Field(ge=0, le=100)
    safety_score: float = Field(ge=0, le=100)
    performance_score: float = Field(ge=0, le=100)
    consistency_score: float = Field(ge=0, le=100)
    rank_position: Optional[int] = None
    percentile: float = Field(ge=0, le=100)
    badge: str = ""
    level: str = ""


class WeeklyReport(BaseModel):
    user_id: str
    week_period: str
    generation_date: datetime

    # Métriques principales
    driver_score: DriverScore
    environmental_impact: Dict
    safety_metrics: Dict
    performance_metrics: Dict
    smart_insights: Dict

    # Données comparatives
    vs_last_week: Dict
    vs_community: Dict

    # Recommandations personnalisées
    personalized_tips: List[str]
    achievements: List[str]
    challenges: List[str]


class WeeklyReportGenerator:
    def __init__(self):
        # Données de référence pour les comparaisons
        self.avg_co2_per_km = 120  # g/km moyenne
        self.avg_fuel_price = 1.65  # €/L
        self.tree_co2_absorption = 22  # kg CO2/an par arbre
        self.target_fuel_consumption = 7.0  # L/100km - AJOUTÉ

        # Système de badges et niveaux
        self.badges = {
            (900, 1000): "🏆 Driving Master",
            (800, 899): "🥇 Eco Champion",
            (700, 799): "🥈 Road Warrior",
            (600, 699): "🥉 Safe Driver",
            (500, 599): "🚗 Cruiser",
            (0, 499): "🌱 Learner"
        }

        self.levels = {
            (900, 1000): "LEGENDARY",
            (800, 899): "EXPERT",
            (700, 799): "ADVANCED",
            (600, 699): "INTERMEDIATE",
            (500, 599): "NOVICE",
            (0, 499): "BEGINNER"
        }

    def calculate_environmental_impact(self, trips: List[TripData]) -> EnvironmentalImpact:
        # Calcul CO2 hebdomadaire
        co2_kg_week = sum(trip.co2_emission_rate * trip.duration_minutes * 60 for trip in trips) / 1000

        # Calcul du carburant consommé
        total_fuel_l = sum(trip.fuel_consumption_l or 0 for trip in trips)
        fuel_cost = total_fuel_l * self.avg_fuel_price

        # Calcul des arbres équivalents
        trees_equivalent = (co2_kg_week * 52) / self.tree_co2_absorption  # 52 semaines par an

        # Calcul du pourcentage vs moyenne
        total_distance = sum(trip.distance_km or 0 for trip in trips)
        if total_distance > 0:
            avg_co2_per_km = (co2_kg_week * 1000) / total_distance
            co2_vs_avg_percent = ((avg_co2_per_km - self.avg_co2_per_km) / self.avg_co2_per_km) * 100
            eco_score = max(0, min(100, int(100 - ((avg_co2_per_km - 80) / 2))))
        else:
            co2_vs_avg_percent = 0
            eco_score = 50

        # Calcul du percentile écologique
        eco_rank_percentile = max(0, min(100, eco_score))

        # Niveau d'efficacité
        if eco_score >= 80:
            efficiency_level = "Excellent"
        elif eco_score >= 65:
            efficiency_level = "Good"
        elif eco_score >= 50:
            efficiency_level = "Average"
        else:
            efficiency_level = "Poor"

        # Rating empreinte carbone
        if co2_kg_week < 50:
            rating = "Excellent"
        elif co2_kg_week < 100:
            rating = "Good"
        elif co2_kg_week < 150:
            rating = "Average"
        else:
            rating = "Poor"

        # Potentiel d'économies corrigé
        if total_distance > 0 and total_fuel_l > 0:
            current_consumption = (total_fuel_l / total_distance) * 100  # L/100km
            potential_savings = max(0, (
                        current_consumption - self.target_fuel_consumption) * total_distance * self.avg_fuel_price / 100)
        else:
            potential_savings = 0

        return EnvironmentalImpact(
            co2_kg_week=co2_kg_week,
            co2_vs_average=co2_vs_avg_percent,
            trees_needed=trees_equivalent,
            fuel_cost_euros=fuel_cost,
            eco_rank_percentile=eco_rank_percentile,
            carbon_footprint_rating=rating,
            eco_score=eco_score,
            efficiency_level=efficiency_level,
            cost_savings_potential=potential_savings
        )

    @staticmethod
    def calculate_safety_metrics(trips: List[TripData]) -> SafetyMetrics:
        """Calcule les métriques de sécurité"""

        aggressive_episodes = sum(1 for trip in trips if trip.driving_style == DrivingStyle.AGGRESSIVE)

        # Score de conduite douce basé sur la variabilité des styles
        style_consistency = len(set(trip.driving_style for trip in trips))
        smooth_score = max(0, 100 - (aggressive_episodes * 20) - (style_consistency * 10))

        # Violations de vitesse (simulées)
        speed_violations = sum(1 for trip in trips if trip.max_speed_kmh > 130)

        # Rang de sécurité
        safety_rank = max(0, smooth_score - (speed_violations * 5))

        return SafetyMetrics(
            aggressive_episodes=aggressive_episodes,
            smooth_driving_score=smooth_score,
            speed_violations=speed_violations,
            safety_rank_percentile=safety_rank
        )

    @staticmethod
    def calculate_performance_metrics(trips: List[TripData]) -> PerformanceMetrics:
        """Calcule les métriques de performance"""

        # Efficacité basée sur CO2 déjà calculé
        co2_values = [trip.co2_per_km for trip in trips if trip.co2_per_km]
        efficiency_score = max(0, 100 - np.mean(co2_values) / 2) if co2_values else 50

        # Consistance basée sur la variance des performances
        speed_variance = np.var([trip.avg_speed_kmh for trip in trips])
        consistency_score = max(0, 100 - speed_variance)

        # Tendance d'amélioration (simulée)
        if len(trips) >= 7:
            first_half = trips[:len(trips) // 2]
            second_half = trips[len(trips) // 2:]

            first_avg_co2 = np.mean([t.co2_per_km for t in first_half if t.co2_per_km])
            second_avg_co2 = np.mean([t.co2_per_km for t in second_half if t.co2_per_km])

            improvement_trend = ((first_avg_co2 - second_avg_co2) / first_avg_co2) * 100 if first_avg_co2 else 0
        else:
            improvement_trend = 0

        # Moments de performance de pointe
        peak_moments = sum(1 for trip in trips
                           if trip.driving_style == DrivingStyle.PEACEFUL and
                           trip.driving_style_confidence > 0.8)

        return PerformanceMetrics(
            efficiency_score=efficiency_score,
            consistency_score=consistency_score,
            improvement_trend=improvement_trend,
            peak_performance_moments=peak_moments
        )

    def generate_smart_insights(self, trips: List[TripData]) -> SmartInsights:
        """Génère des insights intelligents"""

        # Style dominant
        style_counts = {}
        total_confidence = {}

        for trip in trips:
            style = trip.driving_style
            style_counts[style] = style_counts.get(style, 0) + 1
            total_confidence[style] = total_confidence.get(style, 0) + trip.driving_style_confidence

        dominant_style = max(style_counts.items(), key=lambda x: x[1])[0]
        style_confidence = total_confidence[dominant_style] / style_counts[dominant_style]

        # Évolution du style (simulée)
        recent_aggressive = sum(1 for trip in trips[-3:] if trip.driving_style == DrivingStyle.AGGRESSIVE)
        if recent_aggressive == 0:
            style_evolution = "improving"
        elif recent_aggressive >= 2:
            style_evolution = "declining"
        else:
            style_evolution = "stable"

        # Meilleur et pire jour
        trips_by_day = {}
        for trip in trips:
            day = trip.timestamp.strftime("%A")
            trips_by_day[day] = trips_by_day.get(day, []) + [trip]

        day_scores = {}
        for day, day_trips in trips_by_day.items():
            co2_values = [t.co2_per_km for t in day_trips if t.co2_per_km]
            avg_co2 = np.mean(co2_values) if co2_values else 120
            peaceful_ratio = sum(1 for t in day_trips if t.driving_style == DrivingStyle.PEACEFUL) / len(day_trips)
            day_scores[day] = peaceful_ratio * 100 - avg_co2

        best_day = max(day_scores.items(), key=lambda x: x[1])[0] if day_scores else "Monday"
        worst_day = min(day_scores.items(), key=lambda x: x[1])[0] if day_scores else "Friday"

        # Recommandations personnalisées
        recommendations = self.generate_recommendations(dominant_style, style_evolution, trips)

        return SmartInsights(
            dominant_style=dominant_style,
            style_confidence=style_confidence,
            style_evolution=style_evolution,
            best_day=best_day,
            worst_day=worst_day,
            recommendations=recommendations
        )

    @staticmethod
    def generate_recommendations(dominant_style: DrivingStyle, evolution: str, trips: List[TripData]) -> List[
        str]:
        """Génère des recommandations personnalisées"""

        recommendations = []

        if dominant_style == DrivingStyle.AGGRESSIVE:
            recommendations.extend([
                "🧘 Essayez la respiration profonde avant de démarrer",
                "⏰ Partez 10 minutes plus tôt pour éviter le stress",
                "🎵 Écoutez de la musique relaxante pendant la conduite"
            ])
        elif dominant_style == DrivingStyle.PEACEFUL:
            recommendations.extend([
                "🌟 Excellent travail ! Maintenez cette conduite zen",
                "📊 Partagez vos astuces avec la communauté",
                "🎯 Défi : Réduisez encore de 5% vos émissions CO2"
            ])

        if evolution == "declining":
            recommendations.append("📈 Analysez vos trajets difficiles pour identifier les déclencheurs")
        elif evolution == "improving":
            recommendations.append("🚀 Vous progressez ! Continuez sur cette lancée")

        # Recommandations basées sur les émissions CO₂ déjà calculées
        co2_values = [t.co2_per_km for t in trips if t.co2_per_km]
        avg_co2 = np.mean(co2_values) if co2_values else 120

        if avg_co2 > 140:
            recommendations.append("🔧 Vérifiez la pression de vos pneus pour optimiser la consommation")

        return recommendations[:5]  # Limiter à 5 recommandations

    def calculate_driver_score(self, env_impact: EnvironmentalImpact,
                               safety: SafetyMetrics,
                               performance: PerformanceMetrics) -> DriverScore:
        """Calcule le score global du conducteur"""

        # Scores composants (0-100)
        eco_score = min(100, max(0, 100 - abs(env_impact.co2_vs_average)))
        safety_score = safety.smooth_driving_score
        performance_score = (performance.efficiency_score + performance.consistency_score) / 2
        consistency_score = max(0, 100 - abs(performance.improvement_trend))

        # Score global pondéré (sur 1000)
        weights = {
            'eco': 0.3,
            'safety': 0.35,
            'performance': 0.25,
            'consistency': 0.1
        }

        overall_score = (
                eco_score * weights['eco'] * 10 +
                safety_score * weights['safety'] * 10 +
                performance_score * weights['performance'] * 10 +
                consistency_score * weights['consistency'] * 10
        )

        # Badge et niveau
        badge = ""
        level = ""
        for (min_score, max_score), badge_name in self.badges.items():
            if min_score <= overall_score <= max_score:
                badge = badge_name
                break

        for (min_score, max_score), level_name in self.levels.items():
            if min_score <= overall_score <= max_score:
                level = level_name
                break

        # Percentile simulé
        percentile = min(99, overall_score / 10)

        return DriverScore(
            overall_score=round(overall_score, 1),
            eco_score=round(eco_score, 1),
            safety_score=round(safety_score, 1),
            performance_score=round(performance_score, 1),
            consistency_score=round(consistency_score, 1),
            percentile=round(percentile, 1),
            badge=badge,
            level=level
        )

    @staticmethod
    def generate_achievements(trips: List[TripData], driver_score: DriverScore) -> List[str]:
        """Génère les achievements de la semaine"""

        achievements = []

        # Achievements basés sur les données
        peaceful_trips = sum(1 for trip in trips if trip.driving_style == DrivingStyle.PEACEFUL)
        if peaceful_trips >= len(trips) * 0.8:
            achievements.append("🧘 Zen Master - 80% de conduite peaceful")

        if driver_score.eco_score >= 90:
            achievements.append("🌱 Eco Warrior - Score écologique exceptionnel")

        if driver_score.safety_score >= 95:
            achievements.append("🛡️ Safety Guardian - Conduite ultra sécurisée")

        total_distance = sum(trip.distance_km for trip in trips)
        if total_distance >= 500:
            achievements.append("🛣️ Road Runner - Plus de 500km parcourus")

        if len(set(trip.driving_style for trip in trips)) == 1:
            achievements.append("🎯 Style Consistency - Style de conduite constant")

        return achievements

    def generate_weekly_report(self, weekly_data: WeeklyDrivingData) -> WeeklyReport:
        """Génère le rapport hebdomadaire complet"""

        # Calculs des métriques
        env_impact = self.calculate_environmental_impact(weekly_data.trips)
        safety = self.calculate_safety_metrics(weekly_data.trips)
        performance = self.calculate_performance_metrics(weekly_data.trips)
        insights = self.generate_smart_insights(weekly_data.trips)
        driver_score = self.calculate_driver_score(env_impact, safety, performance)

        # Génération des achievements
        achievements = self.generate_achievements(weekly_data.trips, driver_score)

        # Défis pour la semaine suivante
        challenges = [
            f"🎯 Objectif : Atteindre {driver_score.overall_score + 50} points",
            "🌿 Réduire les émissions CO2 de 10%",
            "🏆 Maintenir un style peaceful sur 5 trajets consécutifs"
        ]

        return WeeklyReport(
            user_id=weekly_data.user_id,
            week_period=f"{weekly_data.week_start.strftime('%d/%m')} - {weekly_data.week_end.strftime('%d/%m/%Y')}",
            generation_date=datetime.now(),
            driver_score=driver_score,
            environmental_impact={
                "co2_kg_week": round(env_impact.co2_kg_week, 2),
                "vs_average_percent": round(env_impact.co2_vs_average, 1),
                "trees_needed_annual": round(env_impact.trees_needed, 1),
                "fuel_cost_euros": round(env_impact.fuel_cost_euros, 2),
                "eco_percentile": round(env_impact.eco_rank_percentile, 1)
            },
            safety_metrics={
                "aggressive_episodes": safety.aggressive_episodes,
                "smooth_score": round(safety.smooth_driving_score, 1),
                "speed_violations": safety.speed_violations,
                "safety_percentile": round(safety.safety_rank_percentile, 1)
            },
            performance_metrics={
                "efficiency_score": round(performance.efficiency_score, 1),
                "consistency_score": round(performance.consistency_score, 1),
                "improvement_trend": round(performance.improvement_trend, 1),
                "peak_moments": performance.peak_performance_moments
            },
            smart_insights={
                "dominant_style": insights.dominant_style.value,
                "style_confidence": round(insights.style_confidence, 2),
                "evolution": insights.style_evolution,
                "best_day": insights.best_day,
                "worst_day": insights.worst_day
            },
            vs_last_week={
                "score_change": "+23.5 points",  # Simulé
                "co2_change": "-8.2%",
                "safety_change": "+5.1 points"
            },
            vs_community={
                "better_than": f"{driver_score.percentile}% des conducteurs",
                "rank_position": f"Top {100 - driver_score.percentile:.0f}%"
            },
            personalized_tips=insights.recommendations,
            achievements=achievements,
            challenges=challenges
        )


# Instance globale du générateur
report_generator = WeeklyReportGenerator()


@router.post("/generate_weekly_report", response_model=WeeklyReport)
async def generate_weekly_report(weekly_data: WeeklyDrivingData,  _: None = Depends(verify_api_key )):
    """
    Génère un rapport hebdomadaire impressionnant pour un utilisateur
    """
    try:
        if not weekly_data.trips:
            raise HTTPException(status_code=400, detail="Aucune donnée de trajet fournie")

        report = report_generator.generate_weekly_report(weekly_data)
        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération du rapport: {str(e)}")


@router.post("/calculate_driver_score")
async def calculate_driver_score_only(weekly_data: WeeklyDrivingData,  _: None = Depends(verify_api_key )):
    """
    Calcule uniquement le score du conducteur pour le ranking
    """
    try:
        env_impact = report_generator.calculate_environmental_impact(weekly_data.trips)
        safety = report_generator.calculate_safety_metrics(weekly_data.trips)
        performance = report_generator.calculate_performance_metrics(weekly_data.trips)

        driver_score = report_generator.calculate_driver_score(env_impact, safety, performance)

        return {
            "user_id": weekly_data.user_id,
            "driver_score": driver_score,
            "calculation_date": datetime.now()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul du score: {str(e)}")


@router.get("/community_stats")
async def get_community_stats(user_score: float = 0,  _: None = Depends(verify_api_key )):
    """
    Statistiques communautaires anonymisées
    """
    # Simulation basée sur distribution normale
    community_scores = np.random.normal(650, 150, 1000)  # Moyenne 650, écart-type 150

    percentile = (np.sum(community_scores < user_score) / len(community_scores)) * 100

    return {
        "your_percentile": round(percentile, 1),
        "your_position_description": f"Vous êtes meilleur que {percentile:.1f}% des conducteurs",
        "average_community_score": 650.3,
        "top_10_percent_threshold": 850.0,
        "top_25_percent_threshold": 750.0,
        "your_ranking_tier": get_tier_from_score(user_score),
        "total_active_users": 1247
    }


def get_tier_from_score(score: float) -> str:
    """Détermine le tier basé sur le score"""
    if score >= 900:
        return "LEGENDARY"
    elif score >= 800:
        return "EXPERT"
    elif score >= 700:
        return "ADVANCED"
    elif score >= 600:
        return "INTERMEDIATE"
    elif score >= 500:
        return "NOVICE"
    else:
        return "BEGINNER"


