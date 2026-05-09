import joblib
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException

from ...config import MODELS_DIR
from ...evaluation import (
    plot_metrics_comparison,
    plot_pr_curves,
    plot_roc_curves,
)
from ...preprocessing import get_split_info, split_data
from ...state import state
from ...training import run_training_pipeline

router = APIRouter()


def _do_training():
    try:
        state.update(training_status="running", training_progress=5,
                     training_message="Loading dataset…")

        df = pd.read_csv(state.data_path)

        state.update(training_progress=10, training_message="Splitting data (70/15/15)…")
        X_train, X_test, X_val, y_train, y_test, y_val = split_data(df)
        split_info = get_split_info(y_train, y_test, y_val)

        state.update(
            X_train=X_train,
            y_train=y_train,
            feature_names=list(X_train.columns),
            class_ratio=float((y_train == 0).sum() / (y_train == 1).sum()),
            training_progress=15,
            training_message="Starting hyperparameter search…",
        )

        def _cb(pct: int, msg: str):
            state.update(training_progress=pct, training_message=msg)

        results = run_training_pipeline(
            X_train, y_train, X_test, y_test, X_val, y_val,
            progress_callback=_cb,
        )

        # --- post-training plots (on val set) ---
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
            training_message="Training complete!",
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


@router.post("/start")
async def start_training(background_tasks: BackgroundTasks):
    if state.data_path is None:
        raise HTTPException(status_code=400, detail="Upload a CSV dataset first.")
    if state.training_status == "running":
        raise HTTPException(status_code=400, detail="Training already in progress.")

    background_tasks.add_task(_do_training)
    return {"message": "Training started in background."}


@router.get("/status")
def training_status():
    return {
        "status":   state.training_status,
        "progress": state.training_progress,
        "message":  state.training_message,
    }


@router.get("/results")
def training_results():
    if state.training_results is None:
        raise HTTPException(status_code=404, detail="No training results yet.")
    return state.training_results
