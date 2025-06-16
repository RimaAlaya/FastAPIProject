from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import secrets
import os
from dotenv import load_dotenv

from app.api.OverheatEngine import startup_event
from app.auth import check_credentials
from app.security import verify_api_key  # plus de boucle
from fastapi.responses import JSONResponse

from app.api import CO2emission, DriverProfile, OverheatEngine, TirePressure, WeeklyReport


# Créer l'app (désactiver docs/redoc par défaut)
app = FastAPI(
    title="Intelligent Vehicle Assistant",
    version="1.0",
    docs_url=None,          # désactive /docs par défaut
    redoc_url=None,         # désactive /redoc par défaut
    openapi_url=None        # désactive /openapi.json par défaut
)

# Swagger UI protégé
@app.get("/docs", include_in_schema=False)
def custom_docs(username: str = Depends(check_credentials)):
    return get_swagger_ui_html(
        openapi_url="/openapi.json",  # c’est ici qu’elle va chercher ta spec
        title="Docs sécurisées"
    )

# ReDoc protégé
@app.get("/redoc", include_in_schema=False)
def custom_redoc(username: str = Depends(check_credentials)):
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="ReDoc sécurisées"
    )

# OpenAPI JSON protégé
@app.get("/openapi.json", include_in_schema=False)
def custom_openapi():
    return JSONResponse(
        get_openapi(
            title=app.title,
            version="1.0.0",
            routes=app.routes
        )
    )


# Routers
app.include_router(CO2emission.router)
app.include_router(DriverProfile.router)
app.include_router(OverheatEngine.router)
app.include_router(TirePressure.router)
app.include_router(WeeklyReport.router)

@app.on_event("startup")
async def on_startup():
    await startup_event()