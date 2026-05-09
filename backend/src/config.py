from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

for _dir in [DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
N_ITER_SEARCH = 10
CV_FOLDS = 3
TARGET_COL = "Class"
FEATURES_TO_SCALE = ["Amount", "Time"]

EXPECTED_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Class"]
