import shutil
from typing import Dict

import joblib
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from scipy import stats

from ...config import DATA_DIR, MODELS_DIR, TARGET_COL
from ...evaluation import plot_metrics_comparison, plot_pr_curves, plot_roc_curves
from ...preprocessing import get_split_info, split_data
from ...state import state
from ...training import run_training_pipeline

router = APIRouter()

DRIFT_P_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def _detect_drift(X_ref: pd.DataFrame, X_new: pd.DataFrame) -> Dict:
    feature_drift: Dict = {}
    drifted: list = []

    for col in X_ref.columns:
        if col not in X_new.columns:
            continue
        ks_stat, p_value = stats.ks_2samp(
            X_ref[col].dropna().values,
            X_new[col].dropna().values,
        )
        is_drifted = p_value < DRIFT_P_THRESHOLD
        if is_drifted:
            drifted.append(col)
        feature_drift[col] = {
            "ks_statistic": round(float(ks_stat), 6),
            "p_value":      round(float(p_value), 6),
            "drifted":      is_drifted,
        }

    return {
        "feature_drift":    feature_drift,
        "drifted_features": drifted,
        "n_drifted":        len(drifted),
        "n_total_features": len(X_ref.columns),
        "drift_score":      round(len(drifted) / max(len(X_ref.columns), 1), 4),
        "drift_detected":   len(drifted) > 0,
    }


@router.post("/detect-drift")
async def detect_drift(file: UploadFile = File(...)):
    if state.X_train is None:
        raise HTTPException(
            status_code=400,
            detail="No training reference data. Train a model first.",
        )
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    dest = DATA_DIR / "drift_check.csv"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    df_new = pd.read_csv(dest)
    X_new = df_new.drop(columns=[TARGET_COL], errors="ignore")

    common_cols = [c for c in state.feature_names if c in X_new.columns]
    if not common_cols:
        raise HTTPException(
            status_code=422,
            detail="No matching feature columns found in the uploaded file.",
        )

    return _detect_drift(state.X_train[common_cols], X_new[common_cols])


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

def _do_retraining(new_data_path: str):
    try:
        state.update(
            training_status="running",
            training_progress=5,
            training_message="Merging original + new data…",
        )

        df_original = pd.read_csv(state.data_path)
        df_new = pd.read_csv(new_data_path)
        df_combined = (
            pd.concat([df_original, df_new], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

        combined_path = DATA_DIR / "dataset.csv"
        df_combined.to_csv(combined_path, index=False)
        state.update(data_path=str(combined_path))

        X_train, X_test, X_val, y_train, y_test, y_val = split_data(df_combined)
        split_info = get_split_info(y_train, y_test, y_val)

        state.update(
            X_train=X_train,
            y_train=y_train,
            feature_names=list(X_train.columns),
            class_ratio=float((y_train == 0).sum() / (y_train == 1).sum()),
            training_progress=15,
            training_message="Retraining models…",
        )

        def _cb(pct: int, msg: str):
            state.update(training_progress=pct, training_message=msg)

        results = run_training_pipeline(
            X_train, y_train, X_test, y_test, X_val, y_val,
            progress_callback=_cb,
        )

        _cb(91, "Generating comparison plots…")
        all_models = joblib.load(MODELS_DIR / "all_models.pkl")
        models_probs_val = {
            name: m.predict_proba(X_val)[:, 1]
            for name, m in all_models.items()
        }
        models_probs_test = {
            name: m.predict_proba(X_test)[:, 1]
            for name, m in all_models.items()
        }
        best_model = joblib.load(MODELS_DIR / "best_model.pkl")

        state.update(
            best_model=best_model,
            best_model_name=results["best_model_name"],
            training_status="completed",
            training_progress=100,
            training_message="Retraining complete!",
            training_results={
                **results,
                "split_info": split_info,
                "plots": {
                    "metrics_comparison": plot_metrics_comparison(results["model_results"]),
                    "roc_curves_val":     plot_roc_curves(models_probs_val,  y_val),
                    "pr_curves_val":      plot_pr_curves(models_probs_val,   y_val),
                    "roc_curves_test":    plot_roc_curves(models_probs_test, y_test),
                    "pr_curves_test":     plot_pr_curves(models_probs_test,  y_test),
                },
            },
        )

    except Exception as exc:
        state.update(
            training_status="failed",
            training_message=str(exc),
            training_progress=0,
        )
        raise


@router.post("/retrain")
async def retrain_with_new_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if state.data_path is None:
        raise HTTPException(status_code=400, detail="No base dataset. Upload data first.")
    if state.training_status == "running":
        raise HTTPException(status_code=400, detail="Training already in progress.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    dest = DATA_DIR / "retrain_new.csv"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(_do_retraining, str(dest))
    return {"message": "Retraining started with merged dataset."}


@router.get("/status")
def retraining_status():
    return {
        "status":   state.training_status,
        "progress": state.training_progress,
        "message":  state.training_message,
    }
