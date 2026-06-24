type FRow = { day: number; p05?: number; p10: number; p25?: number; p50: number; p75?: number; p90: number; p95?: number; gt?: number };
type Factor = {
  label: string; unit: string; term_median: string; start_level: number;
  history: { day: number; v: number }[];
  forecast: FRow[];
  paths?: number[][];
};

const W = 220, H = 116, PX = 4, PY = 10;

export function MiniFan({ factor }: { factor: Factor }) {
  const fc = factor.forecast;
  const hasGt = fc.every((f) => typeof f.gt === "number");
  const hasNest = fc.every((f) => typeof f.p05 === "number" && typeof f.p95 === "number");
  const hy = factor.history.map((h) => h.v);
  const allv = [
    ...hy,
    ...fc.flatMap((f) => [f.p05 ?? f.p10, f.p95 ?? f.p90]),
    ...(hasGt ? fc.map((f) => f.gt as number) : []),
    ...(factor.paths ? factor.paths.flat() : []),
  ];
  const vmin = Math.min(...allv), vmax = Math.max(...allv), span = vmax - vmin || 1;
  const x = (d: number) => PX + ((d + 25) / 55) * (W - 2 * PX);
  const y = (v: number) => PY + ((vmax - v) / span) * (H - 2 * PY);
  const poly = (lo: keyof FRow, up: keyof FRow) =>
    [...fc.map((f) => `${x(f.day).toFixed(1)},${y(f[up] as number).toFixed(1)}`),
     ...fc.slice().reverse().map((f) => `${x(f.day).toFixed(1)},${y(f[lo] as number).toFixed(1)}`)].join(" ");
  const line = (vals: number[]) => fc.map((f, i) => `${x(f.day).toFixed(1)},${y(vals[i]).toFixed(1)}`).join(" ");
  const med = line(fc.map((f) => f.p50));
  const hist = factor.history.map((h) => `${x(h.day).toFixed(1)},${y(h.v).toFixed(1)}`).join(" ");
  const up = factor.term_median.trim().startsWith("+");

  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-[13px] font-medium">{factor.label}</span>
        <span className={"text-[12px] font-semibold tabular-nums " + (up ? "text-[var(--chart-1)]" : "text-destructive")}>
          {up ? "▲" : "▼"} {factor.term_median}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-1 w-full">
        <rect x={x(-25)} y={PY} width={x(0) - x(-25)} height={H - 2 * PY} fill="rgba(0,0,0,0.035)" />
        <line x1={x(0)} y1={PY} x2={x(0)} y2={H - PY} stroke="var(--border)" strokeWidth="1" />
        {hasNest ? (
          <>
            <polygon points={poly("p05", "p95")} fill="rgba(0,113,227,0.08)" />
            <polygon points={poly("p10", "p90")} fill="rgba(0,113,227,0.10)" />
            <polygon points={poly("p25", "p75")} fill="rgba(0,113,227,0.14)" />
          </>
        ) : (
          <polygon points={poly("p10", "p90")} fill="rgba(0,113,227,0.12)" />
        )}
        {factor.paths?.map((p, i) => (
          <polyline key={i} points={line(p)} fill="none" stroke="#0071e3" strokeWidth="0.6" strokeOpacity="0.28" />
        ))}
        <polyline points={hist} fill="none" stroke="#1d1d1f" strokeWidth="1.5" />
        <polyline points={med} fill="none" stroke="#0071e3" strokeWidth="1.8" />
        {hasGt && <polyline points={line(fc.map((f) => f.gt as number))} fill="none" stroke="#1d1d1f" strokeWidth="1.3" strokeDasharray="2 2" />}
      </svg>
    </div>
  );
}
