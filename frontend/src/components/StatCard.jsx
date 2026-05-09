export default function StatCard({ label, value, sub, color = "blue" }) {
  const colors = {
    blue:   "border-blue-500/30 bg-blue-500/5  text-blue-400",
    red:    "border-red-500/30  bg-red-500/5   text-red-400",
    green:  "border-green-500/30 bg-green-500/5 text-green-400",
    yellow: "border-yellow-500/30 bg-yellow-500/5 text-yellow-400",
    purple: "border-purple-500/30 bg-purple-500/5 text-purple-400",
  };
  return (
    <div className={`rounded-xl border p-4 ${colors[color]}`}>
      <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">{label}</p>
      <p className="text-2xl font-bold">{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
    </div>
  );
}
