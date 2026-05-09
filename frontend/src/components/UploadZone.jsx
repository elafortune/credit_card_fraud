import { useRef } from "react";
import { Upload } from "lucide-react";

export default function UploadZone({ onFile, label = "Glisser-déposer un CSV ici", accept = ".csv", disabled = false }) {
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  };

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer
        ${disabled ? "border-slate-700 opacity-50 cursor-not-allowed" : "border-slate-600 hover:border-blue-500 hover:bg-blue-500/5"}`}
    >
      <Upload className="mx-auto mb-3 text-slate-500" size={32} />
      <p className="text-sm text-slate-400">{label}</p>
      <p className="text-xs text-slate-600 mt-1">ou cliquez pour parcourir</p>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
      />
    </div>
  );
}
