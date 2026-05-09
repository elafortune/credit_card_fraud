export default function PlotImage({ b64, alt = "plot", className = "" }) {
  if (!b64) return null;
  return (
    <img
      src={`data:image/png;base64,${b64}`}
      alt={alt}
      className={`rounded-lg w-full object-contain ${className}`}
    />
  );
}
