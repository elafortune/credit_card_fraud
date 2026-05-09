from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import eda, evaluation, retraining, training

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Professional ML pipeline for fraud detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes API — enregistrées EN PREMIER pour avoir la priorité sur le catch-all SPA
app.include_router(eda.router,         prefix="/api/eda",        tags=["EDA"])
app.include_router(training.router,    prefix="/api/training",   tags=["Training"])
app.include_router(evaluation.router,  prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(retraining.router,  prefix="/api/retraining", tags=["Retraining"])


@app.get("/api/health", tags=["Health"])
def health():
    return {"status": "ok", "version": "1.0.0"}


# Fichiers statiques du frontend (frontend/dist/) — enregistrés EN DERNIER
# main.py est dans backend/src/api/ → remonter 4 niveaux pour atteindre la racine du repo
FRONTEND_DIST = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"

if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Sert le fichier s'il existe (favicon, icons…), sinon index.html pour React Router
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
