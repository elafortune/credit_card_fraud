import { NavLink } from "react-router-dom";
import {
  BarChart2,
  Brain,
  CheckSquare,
  RefreshCw,
  Shield,
} from "lucide-react";

const NAV_ITEMS = [
  { to: "/",            label: "EDA",         icon: BarChart2   },
  { to: "/training",   label: "Entraînement", icon: Brain       },
  { to: "/evaluation", label: "Évaluation",   icon: CheckSquare },
  { to: "/retraining", label: "Réapprentissage", icon: RefreshCw },
];

export default function Layout({ children }) {
  return (
    <div className="flex min-h-screen bg-[#0f1117]">
      {/* Sidebar */}
      <aside className="w-60 flex-shrink-0 bg-[#141920] border-r border-[#1e2535] flex flex-col">
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-[#1e2535]">
          <Shield className="text-blue-400" size={22} />
          <span className="font-semibold text-white text-sm leading-tight">
            Fraud<br />
            <span className="text-blue-400 font-bold">Detection</span>
          </span>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-4 space-y-1 px-3">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-blue-500/20 text-blue-400"
                    : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
                }`
              }
            >
              <Icon size={17} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="px-5 py-4 border-t border-[#1e2535]">
          <p className="text-xs text-slate-600">ML Pipeline v1.0</p>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  );
}
