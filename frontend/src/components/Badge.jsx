const VARIANTS = {
  success: "bg-green-500/15 text-green-400 border border-green-500/30",
  danger:  "bg-red-500/15   text-red-400   border border-red-500/30",
  warning: "bg-yellow-500/15 text-yellow-400 border border-yellow-500/30",
  info:    "bg-blue-500/15  text-blue-400  border border-blue-500/30",
  neutral: "bg-slate-500/15 text-slate-400 border border-slate-500/30",
};

export default function Badge({ children, variant = "info" }) {
  return (
    <span className={`text-xs font-medium px-2.5 py-0.5 rounded-full ${VARIANTS[variant]}`}>
      {children}
    </span>
  );
}
