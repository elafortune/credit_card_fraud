export default function ProgressBar({ progress, message, status }) {
  const color =
    status === "failed"    ? "bg-red-500"    :
    status === "completed" ? "bg-green-500"  :
                             "bg-blue-500";

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-xs text-slate-400">
        <span>{message || "En attente…"}</span>
        <span>{progress}%</span>
      </div>
      <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
        <div
          className={`h-2 rounded-full transition-all duration-500 ${color} ${
            status === "running" && progress < 100 ? "animate-pulse" : ""
          }`}
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}
