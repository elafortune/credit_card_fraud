from typing import Callable, Dict, List, Optional

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import CV_FOLDS, FEATURES_TO_SCALE, MODELS_DIR, N_ITER_SEARCH, RANDOM_STATE


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _make_preprocessor(feature_names: List[str]) -> ColumnTransformer:
    scale_cols = [f for f in FEATURES_TO_SCALE if f in feature_names]
    return ColumnTransformer(
        [("scaler", StandardScaler(), scale_cols)],
        remainder="passthrough",
    )


def _build_pipelines(feature_names: List[str]) -> Dict[str, ImbPipeline]:
    smote = lambda: SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.5)

    return {
        "random_forest": ImbPipeline([
            ("preprocessor", _make_preprocessor(feature_names)),
            ("smote", smote()),
            ("classifier", RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "logistic_regression": ImbPipeline([
            ("preprocessor", _make_preprocessor(feature_names)),
            ("smote", smote()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "xgboost": ImbPipeline([
            ("preprocessor", _make_preprocessor(feature_names)),
            ("smote", smote()),
            ("classifier", XGBClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="aucpr",
                verbosity=0,
            )),
        ]),
    }


def _param_grids() -> Dict:
    return {
        "random_forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__max_features": ["sqrt", "log2"],
            "smote__k_neighbors": [3, 5],
        },
        "logistic_regression": {
            "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["liblinear", "saga"],
            "smote__k_neighbors": [3, 5],
        },
        "xgboost": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.3],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
            "smote__k_neighbors": [3, 5],
        },
    }


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def _metrics(y_true, y_prob, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "recall":  round(float(recall_score(y_true, y_pred)), 4),
        "f1":      round(float(f1_score(y_true, y_pred)), 4),
        "pr_auc":  round(float(average_precision_score(y_true, y_prob)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training_pipeline(
    X_train, y_train,
    X_test, y_test,
    X_val, y_val,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict:

    def _cb(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    feature_names = list(X_train.columns)
    pipelines = _build_pipelines(feature_names)
    param_grids = _param_grids()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    model_results: Dict = {}
    fitted_models: Dict = {}
    n_models = len(pipelines)

    for i, (name, pipe) in enumerate(pipelines.items()):
        base_pct = 15 + int(i / n_models * 65)
        _cb(base_pct, f"[{i+1}/{n_models}] Hyperparameter search — {name}…")

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grids[name],
            n_iter=N_ITER_SEARCH,
            scoring="average_precision",   # PR-AUC for CV tuning (test set proxy)
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=0,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        fitted_models[name] = best

        test_prob = best.predict_proba(X_test)[:, 1]
        val_prob  = best.predict_proba(X_val)[:, 1]

        model_results[name] = {
            "best_params": {
                k: (v if not isinstance(v, np.generic) else v.item())
                for k, v in search.best_params_.items()
            },
            "cv_pr_auc": round(float(search.best_score_), 4),
            "test_metrics": _metrics(y_test, test_prob),
            "val_metrics":  _metrics(y_val,  val_prob),
        }

    _cb(82, "Selecting best model on validation set (PR-AUC)…")

    best_model_name = max(
        model_results,
        key=lambda n: model_results[n]["val_metrics"]["pr_auc"],
    )
    best_model = fitted_models[best_model_name]

    joblib.dump(best_model,   MODELS_DIR / "best_model.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    joblib.dump(fitted_models, MODELS_DIR / "all_models.pkl")

    _cb(90, "Models saved.")

    return {
        "model_results": model_results,
        "best_model_name": best_model_name,
        "best_model_val_metrics": model_results[best_model_name]["val_metrics"],
    }
