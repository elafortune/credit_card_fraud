import { useEffect, useState } from "react";
import { evaluateDataset, getCurrentModel } from "../api/client";
import MetricsTable from "../components/MetricsTable";
import PlotImage from "../components/PlotImage";
import StatCard from "../components/StatCard";
import UploadZone from "../components/UploadZone";
import { AlertCircle, CheckCircle, Loader } from "lucide-react";

export default function EvaluationPage() {
  const [model, setModel]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [result, setResult]   = useState(null);

  useEffect(() => {
    getCurrentModel()
      .then(({ data }) => setModel(data))
      .catch(() => {});
  }, []);

  async function handleFile(file) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const { data } = await evaluateDataset(file);
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }

  const MODEL_LABELS = {
    random_forest:       "Random Forest",
    logistic_regression: "Régression Logistique",
    xgboost:             "XGBoost",
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-xl font-semibold text-white mb-1">Évaluation de Nouveau Dataset</h1>
      <p className="text-sm text-slate-500 mb-6">
        Évaluez le meilleur modèle entraîné sur un nouveau jeu de données (mêmes colonnes).
      </p>

      {/* Model info */}
      {model ? (
        <div className="bg-[#141920] border border-[#1e2535] rounded-xl p-4 mb-5">
          <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Modèle actif</p>
          <p className="text-sm font-semibold text-white">
            {MODEL_LABELS[model.best_model] || model.best_model}
          </p>
          {model.val_metrics && (
            <div className="flex gap-4 mt-2">
              {Object.entries(model.val_metrics).map(([k, v]) => (
                <span key={k} className="text-xs text-slate-400">
                  <span className="text-blue-400 font-mono">{(v * 100).toFixed(2)}%</span>
                  {" "}{k.toUpperCase().replace("_", "-")}
                </span>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-4 mb-5 text-yellow-400 text-sm">
          Aucun modèle entraîné. Allez dans "Entraînement" pour créer un modèle.
        </div>
      )}

      {/* Upload */}
      <UploadZone
        onFile={handleFile}
        disabled={loading || !model}
        label="Importer un CSV à évaluer (avec colonne 'Class')"
      />

      {loading && (
        <div className="mt-4 flex items-center gap-2 text-blue-400 text-sm">
          <Loader size={15} className="animate-spin" />
          Évaluation en cours…
        </div>
      )}
      {error && (
        <div className="mt-4 flex items-center gap-2 text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg p-3">
          <AlertCircle size={14} />
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6 space-y-5">
          <div className="flex items-center gap-2 text-green-400 text-sm">
            <CheckCircle size={15} />
            Évaluation terminée sur {result.model_name}
          </div>

          {/* Confusion matrix summary */}
          {result.confusion_matrix && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard label="Vrais Négatifs (TN)" value={result.confusion_matrix.tn.toLocaleString("fr")} color="green" />
              <StatCard label="Faux Positifs (FP)"  value={result.confusion_matrix.fp.toLocaleString("fr")} color="yellow" />
              <StatCard label="Faux Négatifs (FN)"  value={result.confusion_matrix.fn.toLocaleString("fr")} color="red" />
              <StatCard label="Vrais Positifs (TP)" value={result.confusion_matrix.tp.toLocaleString("fr")} color="blue" />
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div className="bg-[#141920] border border-[#1e2535] rounded-xl p-4">
              <p className="text-xs text-slate-400 uppercase tracking-wider mb-3">Métriques</p>
              <MetricsTable metrics={result.metrics} />
            </div>
            {result.plots?.confusion_matrix && (
              <div className="bg-[#141920] border border-[#1e2535] rounded-xl p-4">
                <p className="text-xs text-slate-400 uppercase tracking-wider mb-3">Matrice de Confusion</p>
                <PlotImage b64={result.plots.confusion_matrix} alt="confusion matrix" />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
