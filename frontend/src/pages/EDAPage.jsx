import { useState } from "react";
import { uploadDataset } from "../api/client";
import PlotImage from "../components/PlotImage";
import StatCard from "../components/StatCard";
import UploadZone from "../components/UploadZone";
import { AlertCircle, CheckCircle, Loader } from "lucide-react";

const PLOT_LABELS = {
  class_distribution: "Distribution des classes",
  amount_distribution: "Distribution des montants",
  time_distribution: "Distribution temporelle",
  correlation_heatmap: "Corrélation avec la fraude",
  top_features: "Top 6 features discriminantes",
};

export default function EDAPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [report, setReport]   = useState(null);
  const [fileName, setFileName] = useState(null);

  async function handleFile(file) {
    setLoading(true);
    setError(null);
    setReport(null);
    setFileName(file.name);
    try {
      const { data } = await uploadDataset(file);
      setReport(data.report);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }

  const s = report?.summary;

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-xl font-semibold text-white mb-1">Analyse Exploratoire (EDA)</h1>
      <p className="text-sm text-slate-500 mb-6">
        Importez le dataset Kaggle Credit Card Fraud CSV pour analyser et visualiser les données.
      </p>

      <UploadZone onFile={handleFile} disabled={loading} label="Importer creditcard.csv" />

      {/* Status */}
      {loading && (
        <div className="mt-4 flex items-center gap-2 text-blue-400 text-sm">
          <Loader size={16} className="animate-spin" />
          Analyse en cours…
        </div>
      )}
      {error && (
        <div className="mt-4 flex items-center gap-2 text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg p-3">
          <AlertCircle size={16} />
          {error}
        </div>
      )}
      {report && !loading && (
        <div className="mt-4 flex items-center gap-2 text-green-400 text-sm">
          <CheckCircle size={16} />
          <span><strong>{fileName}</strong> analysé — {s.duplicates_removed} doublon(s) supprimé(s)</span>
        </div>
      )}

      {/* Stats grid */}
      {s && (
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard
            label="Total transactions"
            value={s.total_transactions.toLocaleString("fr")}
            color="blue"
          />
          <StatCard
            label="Fraudes"
            value={s.n_fraud.toLocaleString("fr")}
            sub={`${(s.fraud_rate * 100).toFixed(3)}% du total`}
            color="red"
          />
          <StatCard
            label="Légitimes"
            value={s.n_legitimate.toLocaleString("fr")}
            color="green"
          />
          <StatCard
            label="Montant moyen"
            value={`$${s.amount_all.mean.toFixed(2)}`}
            sub={`Fraude: $${s.amount_fraud.mean.toFixed(2)}`}
            color="yellow"
          />
        </div>
      )}

      {/* Plots */}
      {report?.plots && (
        <div className="mt-6 space-y-6">
          {Object.entries(report.plots).map(([key, b64]) => (
            <div key={key} className="bg-[#141920] rounded-xl border border-[#1e2535] p-4">
              <h2 className="text-sm font-medium text-slate-300 mb-3">
                {PLOT_LABELS[key] || key}
              </h2>
              <PlotImage b64={b64} alt={key} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
