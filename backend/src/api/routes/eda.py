import shutil

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from ...config import DATA_DIR
from ...eda import clean_dataframe, generate_eda_report, validate_dataframe
from ...state import state

router = APIRouter()


@router.post("/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    dest = DATA_DIR / "dataset.csv"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    df = pd.read_csv(dest)
    valid, error = validate_dataframe(df)
    if not valid:
        raise HTTPException(status_code=422, detail=f"Invalid dataset: {error}")

    df, n_removed = clean_dataframe(df)
    df.to_csv(dest, index=False)

    report = generate_eda_report(df)
    report["summary"]["duplicates_removed"] = n_removed

    state.update(
        data_path=str(dest),
        eda_report=report,
        training_status="idle",
        training_progress=0,
        training_message="",
        training_results=None,
        best_model=None,
        best_model_name=None,
        feature_names=None,
        X_train=None,
        y_train=None,
    )

    return {
        "message": f"Dataset uploaded. {n_removed} duplicate(s) removed.",
        "report": report,
    }


@router.get("/report")
def get_eda_report():
    if state.eda_report is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded yet.")
    return state.eda_report
