import { useCallback, useRef, useState } from "react";
import {
  detectDrift,
  getRetrainingStatus,
  retrainModel,
} from "../api/client";
import Badge from "../components/Badge";
import ProgressBar from "../components/ProgressBar";
import UploadZone from "../components/UploadZone";
import { Activity, AlertCircle, AlertTriangle, CheckCircle, Loader, RefreshCw } from "lucide-react";

export default function RetrainingPage() {
  // Drift detection
  const [driftLoading, setDriftLoading] = useState(false);
  const [driftError, setDriftError]     = useState(null);
  const [driftReport, setDriftReport]   = useState(null);

  // Retraining
  const [retrainStatus, setRetrainStatus] = useState({ status: "idle", progress: 0, message: "" });
  const [retrainError, setRetrainError]   = useState(null);
  const [retrainDone, setRetrainDone]     = useState(false);
  const pollRef = useRef(null);

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };

  async function handleDriftFile(file) {
    setDriftLoading(true);
    setDriftError(null);
    setDriftReport(null);
    try {
      const { data } = await detectDrift(file);
      setDriftReport(data);
    } catch (e) {
      setDriftError(e.response?.data?.detail || e.message);
    } finally {
      setDriftLoading(false);
    }
  }

  const poll = useCallback(async () => {
    try {
      const { data } = await getRetrainingStatus();
      setRetrainStatus(data);
      if (data.status === "completed") { stopPolling(); setRetrainDone(true); }
      else if (data.status === "failed") { stopPolling(); setRetrainError(data.message); }
    } catch { stopPolling(); }
  }, []);

  async function handleRetrainFile(file) {
    setRetrainError(null);
    setRetrainDone(false);
    try {
      await retrainModel(file);
      setRetrainStatus({ status: "running", progress: 0, message: "Initialisation…" });
      stopPolling();
      pollRef.current = setInterval(poll, 1500);
    } catch (e) {
      setRetrainError(e.response?.data?.detail || e.message);
    }
  }

  const isRetraining = retrainStatus.status === "running";

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-xl font-semibold text-white mb-1">Réapprentissage & Data Drift</h1>
      <p className="text-sm text-slate-500 mb-6">
        Détectez le glissement de données (KS-test) et réentraînez le modèle avec de nouvelles données.
      </p>

      {/* ── Drift Detection ── */}
      <section className="bg-[#141920] border border-[#1e2535] rounded-xl p-5 mb-6">
        <div className="flex items-center gap-2 mb-1">
          <Activity size={17} className="text-purple-400" />
          <h2 className="text-sm font-semibold text-white">Détection de Data Drift</h2>
        </div>
        <p className="text-xs text-slate-500 mb-4">
          Test de Kolmogorov-Smirnov (p &lt; 0.05) sur chaque feature par rapport aux données d'entraînement.
        </p>

        <UploadZone
          onFile={handleDriftFile}
          disabled={driftLoading}
          label="Importer un nouveau CSV pour analyse de drift"
        />

        {driftLoading && (
          <div className="mt-3 flex items-center gap-2 text-purple-400 text-sm">
            <Loader size={14} className="animate-spin" />
            Analyse KS-test…
          </div>
        )}
        {driftError && (
          <div className="mt-3 flex items-center gap-2 text-red-400 text-sm">
            <AlertCircle size={14} />
            {driftError}
          </div>
        )}

        {driftReport && (
          <div className="mt-4 space-y-4">
            {/* Summary */}
            <div className={`flex items-start gap-3 p-3 rounded-lg border ${
              driftReport.drift_detected
                ? "bg-red-500/10 border-red-500/25 text-red-400"
                : "bg-green-500/10 border-green-500/25 text-green-400"
            }`}>
              {driftReport.drift_detected
                ? <AlertTriangle size={16} className="flex-shrink-0 mt-0.5" />
                : <CheckCircle  size={16} className="flex-shrink-0 mt-0.5" />}
              <div>
                <p className="text-sm font-medium">
                  {driftReport.drift_detected
                    ? `Drift détecté sur ${driftReport.n_drifted}/${driftReport.n_total_features} features`
                    : "Aucun drift significatif détecté"}
                </p>
                <p className="text-xs opacity-75 mt-0.5">
                  Score de drift global : {(driftReport.drift_score * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Drifted features list */}
            {driftReport.drifted_features.length > 0 && (
              <div className="bg-[#1a1f2e] rounded-lg p-3">
                <p className="text-xs text-slate-400 mb-2 uppercase tracking-wider">Features en drift</p>
                <div className="flex flex-wrap gap-2">
                  {driftReport.drifted_features.map((f) => (
                    <Badge key={f} variant="danger">{f}</Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Full KS table */}
            <details className="group">
              <summary className="text-xs text-slate-400 cursor-pointer hover:text-slate-300 select-none">
                Voir tous les résultats KS-test ({Object.keys(driftReport.feature_drift).length} features)
              </summary>
              <div className="mt-2 overflow-hidden rounded-lg border border-[#1e2535] max-h-72 overflow-y-auto">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-[#1a1f2e]">
                    <tr className="text-slate-400 uppercase tracking-wider">
                      <th className="text-left px-3 py-2">Feature</th>
                      <th className="text-right px-3 py-2">KS stat</th>
                      <th className="text-right px-3 py-2">p-value</th>
                      <th className="text-center px-3 py-2">Drift</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(driftReport.feature_drift).map(([feat, info]) => (
                      <tr key={feat} className={`border-t border-[#1e2535] ${info.drifted ? "bg-red-500/5" : ""}`}>
                        <td className="px-3 py-1.5 text-slate-300">{feat}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-slate-400">{info.ks_statistic.toFixed(4)}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-slate-400">{info.p_value.toFixed(4)}</td>
                        <td className="px-3 py-1.5 text-center">
                          <Badge variant={info.drifted ? "danger" : "success"}>
                            {info.drifted ? "Oui" : "Non"}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </div>
        )}
      </section>

      {/* ── Retraining ── */}
      <section className="bg-[#141920] border border-[#1e2535] rounded-xl p-5">
        <div className="flex items-center gap-2 mb-1">
          <RefreshCw size={17} className="text-blue-400" />
          <h2 className="text-sm font-semibold text-white">Réentraînement</h2>
        </div>
        <p className="text-xs text-slate-500 mb-4">
          Les nouvelles données sont fusionnées avec le dataset original avant de relancer
          la pipeline complète (3 modèles + hyperparameter search).
        </p>

        <UploadZone
          onFile={handleRetrainFile}
          disabled={isRetraining}
          label="Importer de nouvelles données (même format CSV)"
        />

        {(isRetraining || retrainStatus.status === "completed" || retrainStatus.status === "failed") && (
          <div className="mt-4">
            <ProgressBar
              progress={retrainStatus.progress}
              message={retrainStatus.message}
              status={retrainStatus.status}
            />
          </div>
        )}
        {retrainError && (
          <div className="mt-3 flex items-center gap-2 text-red-400 text-sm">
            <AlertCircle size={14} />
            {retrainError}
          </div>
        )}
        {retrainDone && (
          <div className="mt-3 flex items-center gap-2 text-green-400 text-sm">
            <CheckCircle size={14} />
            Réentraînement terminé. Consultez l'onglet <strong>Entraînement</strong> pour les nouveaux résultats.
          </div>
        )}
      </section>
    </div>
  );
}
