from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TARGET_COL, TEST_SIZE, VAL_SIZE


def split_data(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # val_ratio relative to the trainval portion
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )
    return X_train, X_test, X_val, y_train, y_test, y_val


def get_split_info(y_train, y_test, y_val) -> Dict:
    def _counts(y) -> Dict:
        return {
            "total": len(y),
            "n_fraud": int((y == 1).sum()),
            "n_legit": int((y == 0).sum()),
            "fraud_rate": round(float(y.mean()), 6),
        }

    return {
        "train": _counts(y_train),
        "test": _counts(y_test),
        "val": _counts(y_val),
    }
