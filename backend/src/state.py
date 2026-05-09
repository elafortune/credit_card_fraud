import threading
from typing import Any, Dict, List, Optional


class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.data_path: Optional[str] = None
            self.eda_report: Optional[Dict] = None
            self.training_status: str = "idle"
            self.training_progress: int = 0
            self.training_message: str = ""
            self.training_results: Optional[Dict] = None
            self.best_model: Any = None
            self.best_model_name: Optional[str] = None
            self.feature_names: Optional[List[str]] = None
            self.X_train: Any = None
            self.y_train: Any = None
            self.class_ratio: Optional[float] = None

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)


state = AppState()
