const METRIC_LABELS = {
  recall:   "Recall",
  f1:       "F1 Score",
  pr_auc:   "PR-AUC",
  roc_auc:  "ROC-AUC",
  precision: "Précision",
  specificity: "Spécificité",
  false_positive_rate: "Taux FP",
};

function pct(v) {
  return typeof v === "number" ? (v * 100).toFixed(2) + "%" : v;
}

export default function MetricsTable({ metrics }) {
  if (!metrics) return null;
  return (
    <div className="overflow-hidden rounded-xl border border-[#1e2535]">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-[#1a1f2e] text-slate-400 text-xs uppercase tracking-wider">
            <th className="text-left px-4 py-2">Métrique</th>
            <th className="text-right px-4 py-2">Valeur</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics).map(([key, val]) => (
            <tr
              key={key}
              className="border-t border-[#1e2535] hover:bg-white/[0.02] transition-colors"
            >
              <td className="px-4 py-2 text-slate-300">
                {METRIC_LABELS[key] || key}
              </td>
              <td className="px-4 py-2 text-right font-mono text-blue-300">
                {pct(val)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
