import { useCallback, useEffect, useRef, useState } from "react";
import {
  getTrainingResults,
  getTrainingStatus,
  startTraining,
} from "../api/client";
import Badge from "../components/Badge";
import PlotImage from "../components/PlotImage";
import ProgressBar from "../components/ProgressBar";
import { AlertCircle, Play, Trophy } from "lucide-react";

const MODEL_LABELS = {
  random_forest:      "Random Forest",
  logistic_regression: "Régression Logistique",
  xgboost:            "XGBoost",
};

const METRIC_COLS = ["recall", "f1", "pr_auc", "roc_auc"];
const METRIC_HEADS = ["Recall", "F1", "PR-AUC", "ROC-AUC"];

const PLOT_LABELS = {
  metrics_comparison: "Comparaison des modèles (validation)",
  pr_curves_val:      "Courbes Précision-Rappel (validation)",
  roc_curves_val:     "Courbes ROC (validation)",
  pr_curves_test:     "Courbes Précision-Rappel (test)",
  roc_curves_test:    "Courbes ROC (test)",
};

export default function TrainingPage() {
  const [status, setStatus]   = useState({ status: "idle", progress: 0, message: "" });
  const [results, setResults] = useState(null);
  const [error, setError]     = useState(null);
  const pollRef = useRef(null);

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };

  const poll = useCallback(async () => {
    try {
      const { data } = await getTrainingStatus();
      setStatus(data);
      if (data.status === "completed") {
        stopPolling();
        const { data: res } = await getTrainingResults();
        setResults(res);
      } else if (data.status === "failed") {
        stopPolling();
        setError(data.message);
      }
    } catch (e) {
      stopPolling();
    }
  }, []);

  useEffect(() => () => stopPolling(), []);

  async function handleStart() {
    setError(null);
    setResults(null);
    try {
      await startTraining();
      setStatus({ status: "running", progress: 0, message: "Démarrage…" });
      stopPolling();
      pollRef.current = setInterval(poll, 1500);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    }
  }

  const isRunning = status.status === "running";
  const isDone    = status.status === "completed";
  const isFailed  = status.status === "failed";

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-xl font-semibold text-white mb-1">Pipeline d'Entraînement</h1>
      <p className="text-sm text-slate-500 mb-6">
        RandomForest · Régression Logistique · XGBoost — SMOTE + hyperparameter search (70/15/15)
      </p>

      {/* Launch button */}
      <div className="bg-[#141920] rounded-xl border border-[#1e2535] p-5 mb-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-sm font-medium text-slate-200">Lancer l'entraînement</p>
            <p className="text-xs text-slate-500 mt-0.5">
              Le dataset EDA doit avoir été importé au préalable.
            </p>
          </div>
          <button
            onClick={handleStart}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors"
          >
            <Play size={15} />
            {isRunning ? "En cours…" : "Entraîner"}
          </button>
        </div>

        {(isRunning || isDone || isFailed) && (
          <ProgressBar
            progress={status.progress}
            message={status.message}
            status={status.status}
          />
        )}

        {error && (
          <div className="mt-3 flex items-center gap-2 text-red-400 text-sm">
            <AlertCircle size={14} />
            {error}
          </div>
        )}
      </div>

      {/* Results */}
      {results && (
        <>
          {/* Best model banner */}
          <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-xl p-4 mb-5 flex items-center gap-3">
            <Trophy className="text-yellow-400" size={22} />
            <div>
              <p className="text-sm font-semibold text-white">
                Meilleur modèle : {MODEL_LABELS[results.best_model_name] || results.best_model_name}
              </p>
              <p className="text-xs text-slate-400 mt-0.5">
                Sélectionné sur la PR-AUC du jeu de validation.
                PR-AUC = {results.best_model_val_metrics.pr_auc} ·
                Recall = {results.best_model_val_metrics.recall} ·
                F1 = {results.best_model_val_metrics.f1}
              </p>
            </div>
          </div>

          {/* Split info */}
          {results.split_info && (
            <div className="grid grid-cols-3 gap-3 mb-5">
              {Object.entries(results.split_info).map(([split, info]) => (
                <div key={split} className="bg-[#141920] border border-[#1e2535] rounded-xl p-3">
                  <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">
                    {split === "train" ? "Entraînement" : split === "test" ? "Test (tuning)" : "Validation (sélection)"}
                  </p>
                  <p className="text-lg font-semibold text-white">{info.total.toLocaleString("fr")}</p>
                  <p className="text-xs text-slate-500">
                    {info.n_fraud} fraudes ({(info.fraud_rate * 100).toFixed(3)}%)
                  </p>
                </div>
              ))}
            </div>
          )}

          {/* Comparison table */}
          <div className="bg-[#141920] border border-[#1e2535] rounded-xl overflow-hidden mb-5">
            <div className="px-4 py-3 border-b border-[#1e2535]">
              <p className="text-sm font-medium text-slate-200">Résultats sur validation</p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-[#1a1f2e] text-slate-400 text-xs">
                    <th className="text-left px-4 py-2">Modèle</th>
                    <th className="text-center px-4 py-2">CV PR-AUC</th>
                    {METRIC_HEADS.map((h) => (
                      <th key={h} className="text-center px-4 py-2">{h} (test)</th>
                    ))}
                    {METRIC_HEADS.map((h) => (
                      <th key={h+"v"} className="text-center px-4 py-2">{h} (val)</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(results.model_results).map(([name, r]) => (
                    <tr
                      key={name}
                      className={`border-t border-[#1e2535] ${
                        name === results.best_model_name ? "bg-blue-500/5" : "hover:bg-white/[0.02]"
                      }`}
                    >
                      <td className="px-4 py-2 font-medium text-slate-200 flex items-center gap-2">
                        {MODEL_LABELS[name] || name}
                        {name === results.best_model_name && (
                          <Badge variant="success">Meilleur</Badge>
                        )}
                      </td>
                      <td className="px-4 py-2 text-center font-mono text-slate-400">
                        {r.cv_pr_auc}
                      </td>
                      {METRIC_COLS.map((m) => (
                        <td key={m} className="px-4 py-2 text-center font-mono text-slate-300">
                          {(r.test_metrics[m] * 100).toFixed(2)}%
                        </td>
                      ))}
                      {METRIC_COLS.map((m) => (
                        <td
                          key={m+"v"}
                          className={`px-4 py-2 text-center font-mono ${
                            name === results.best_model_name ? "text-blue-300" : "text-slate-300"
                          }`}
                        >
                          {(r.val_metrics[m] * 100).toFixed(2)}%
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Plots */}
          {results.plots && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {Object.entries(results.plots).map(([key, b64]) => (
                <div key={key} className="bg-[#141920] border border-[#1e2535] rounded-xl p-4">
                  <p className="text-xs font-medium text-slate-400 mb-3 uppercase tracking-wider">
                    {PLOT_LABELS[key] || key}
                  </p>
                  <PlotImage b64={b64} alt={key} />
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
