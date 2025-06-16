from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import math
from fastapi import APIRouter

from app.security import verify_api_key  # plus de boucle

router = APIRouter()



class OBDData(BaseModel):
    """Input model for OBD-II PIDs data"""
    # Core PIDs for CO₂ calculation
    maf: Optional[float] = Field(None, description="Mass Air Flow (g/s) - PID 0x10")
    fuel_rate: Optional[float] = Field(None, description="Fuel Flow Rate (L/h) - PID 0x5E")

    # Alternative PIDs when MAF not available
    rpm: Optional[float] = Field(None, description="Engine RPM - PID 0x0C")
    map_pressure: Optional[float] = Field(None, description="Manifold Absolute Pressure (kPa) - PID 0x0B")
    iat: Optional[float] = Field(None, description="Intake Air Temperature (°C) - PID 0x0F")

    # Fuel type detection PIDs
    fuel_type: Optional[int] = Field(None, description="Fuel Type - PID 0x51")
    fuel_system_status: Optional[int] = Field(None, description="Fuel System Status - PID 0x03")

    # Vehicle speed for CO₂/km calculation
    speed: Optional[float] = Field(None, description="Vehicle Speed (km/h) - PID 0x0D")

    # Manual fuel type override
    manual_fuel_type: Optional[str] = Field(None, description="Manual fuel type: 'gasoline' or 'diesel'")

    # Engine specifications (if available)
    engine_displacement: Optional[float] = Field(None, description="Engine displacement in liters")


class CO2Result(BaseModel):
    """Output model for CO₂ emissions result"""
    co2_emission_rate: float = Field(description="CO₂ emission rate (g/s)")
    co2_per_km: Optional[float] = Field(None, description="CO₂ emissions per km (g/km)")
    fuel_type_detected: str = Field(description="Detected fuel type")
    calculation_method: str = Field(description="Method used for calculation")
    confidence: str = Field(description="Confidence level of the result")


# Constants
FUEL_CONSTANTS = {
    'gasoline': {
        'co2_factor': 2340,  # g/L
        'afr': 14.7,  # Air-Fuel Ratio
        'density': 750  # g/L
    },
    'diesel': {
        'co2_factor': 2690,  # g/L
        'afr': 14.5,  # Air-Fuel Ratio
        'density': 840  # g/L
    }
}

# Fuel type detection mappings (based on PID 0x51)
FUEL_TYPE_MAPPING = {
    1: 'gasoline',  # Gasoline
    2: 'diesel',  # Diesel
    3: 'gasoline',  # CNG (treat as gasoline for calculation)
    4: 'gasoline',  # Propane
    # Add more mappings as needed
}


def detect_fuel_type(obd_data: OBDData) -> str:
    """Detect fuel type from OBD-II data"""

    # Priority 1: Manual override
    if obd_data.manual_fuel_type:
        if obd_data.manual_fuel_type.lower() in ['gasoline', 'diesel']:
            return obd_data.manual_fuel_type.lower()

    # Priority 2: Direct fuel type PID
    if obd_data.fuel_type is not None:
        return FUEL_TYPE_MAPPING.get(obd_data.fuel_type, 'gasoline')

    # Priority 3: Heuristics based on other PIDs
    # Diesel engines typically have higher MAP at idle and different RPM characteristics
    if obd_data.map_pressure and obd_data.rpm:
        if obd_data.rpm < 1000 and obd_data.map_pressure > 80:
            return 'diesel'  # Diesel engines often have higher compression

    # Default to gasoline if uncertain
    return 'gasoline'


def calculate_fuel_flow_from_maf(maf: float, fuel_type: str) -> float:
    """Calculate fuel flow rate from MAF sensor data"""
    constants = FUEL_CONSTANTS[fuel_type]

    # Convert MAF (g/s) to fuel flow (L/s)
    # V_fuel = m_air / (AFR * rho_fuel)
    fuel_flow_ls = maf / (constants['afr'] * constants['density'])

    return fuel_flow_ls


def estimate_airflow_from_map(rpm: float, map_pressure: float, iat: float,
                              displacement: float = 2.0, ve: float = 0.85) -> float:
    """Estimate airflow using MAP, RPM, and IAT (when MAF not available)"""

    # Convert temperature to Kelvin
    temp_k = iat + 273.15

    # Air density calculation using ideal gas law
    # ρ = P / (R_specific * T)
    R_specific = 287  # J/(kg·K) for air
    air_density = (map_pressure * 1000) / (R_specific * temp_k)  # kg/m³

    # Volumetric flow rate (m³/s)
    volumetric_flow = (displacement / 1000) * (rpm / 60) * (ve / 2)  # /2 for 4-stroke

    # Mass flow rate (kg/s) -> convert to g/s
    mass_flow = air_density * volumetric_flow * 1000

    return mass_flow


@router.post("/calculate_co2", response_model=CO2Result)
async def calculate_co2_emissions(obd_data: OBDData,  _: None = Depends(verify_api_key )):
    """
    Calculate CO₂ emissions from OBD-II PIDs data
    """

    # Detect fuel type
    fuel_type = detect_fuel_type(obd_data)
    constants = FUEL_CONSTANTS[fuel_type]

    fuel_flow_ls = None
    method = ""
    confidence = "high"

    try:
        # Method 1: Direct fuel rate PID (most accurate)
        if obd_data.fuel_rate is not None:
            fuel_flow_ls = obd_data.fuel_rate / 3600  # Convert L/h to L/s
            method = "direct_fuel_rate"
            confidence = "very_high"

        # Method 2: MAF sensor (high accuracy)
        elif obd_data.maf is not None:
            fuel_flow_ls = calculate_fuel_flow_from_maf(obd_data.maf, fuel_type)
            method = "maf_based"
            confidence = "high"

        # Method 3: MAP-based estimation (medium accuracy)
        elif all([obd_data.rpm, obd_data.map_pressure, obd_data.iat]):
            displacement = obd_data.engine_displacement or 2.0  # Default 2.0L
            estimated_maf = estimate_airflow_from_map(
                obd_data.rpm, obd_data.map_pressure, obd_data.iat, displacement
            )
            fuel_flow_ls = calculate_fuel_flow_from_maf(estimated_maf, fuel_type)
            method = "map_estimation"
            confidence = "medium"

        else:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data: Need either MAF, Fuel Rate, or (RPM + MAP + IAT)"
            )

        # Calculate CO₂ emission rate (g/s)
        co2_rate = fuel_flow_ls * constants['co2_factor']

        # Calculate CO₂ per km if speed is available
        co2_per_km = None
        if obd_data.speed is not None and obd_data.speed > 0:
            speed_ms = obd_data.speed / 3.6  # Convert km/h to m/s
            co2_per_km = (co2_rate / speed_ms) * 1000  # g/km

        return CO2Result(
            co2_emission_rate=round(co2_rate, 4),
            co2_per_km=round(co2_per_km, 2) if co2_per_km else None,
            fuel_type_detected=fuel_type,
            calculation_method=method,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")


