from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

app.include_router(eda.router,         prefix="/api/eda",        tags=["EDA"])
app.include_router(training.router,    prefix="/api/training",   tags=["Training"])
app.include_router(evaluation.router,  prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(retraining.router,  prefix="/api/retraining", tags=["Retraining"])


@app.get("/api/health", tags=["Health"])
def health():
    return {"status": "ok", "version": "1.0.0"}
