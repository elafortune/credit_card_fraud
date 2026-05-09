import shutil

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from ...config import DATA_DIR, TARGET_COL
from ...evaluation import evaluate_model
from ...state import state

router = APIRouter()


@router.post("/evaluate-dataset")
async def evaluate_new_dataset(file: UploadFile = File(...)):
    if state.best_model is None:
        raise HTTPException(status_code=400, detail="No trained model available. Train first.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    dest = DATA_DIR / "eval_dataset.csv"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    df = pd.read_csv(dest)

    if TARGET_COL not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset must contain '{TARGET_COL}' column for evaluation.",
        )

    if state.feature_names:
        missing = [c for c in state.feature_names if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing feature columns: {missing}",
            )
        X = df[state.feature_names]
    else:
        X = df.drop(columns=[TARGET_COL])

    y = df[TARGET_COL]

    result = evaluate_model(
        state.best_model, X, y,
        model_name=state.best_model_name or "Best Model",
    )
    return result


@router.get("/current-model")
def current_model_info():
    if state.best_model_name is None:
        raise HTTPException(status_code=404, detail="No trained model available.")
    return {
        "best_model":   state.best_model_name,
        "val_metrics":  (
            state.training_results["best_model_val_metrics"]
            if state.training_results else None
        ),
        "feature_names": state.feature_names,
    }
