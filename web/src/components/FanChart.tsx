import { useEffect, useState, type ComponentType } from "react";

type FPt = { day: number } & Partial<Record<"p05" | "p10" | "p25" | "p50" | "p75" | "p90" | "p95" | "gt", number>>;
type Fan = { start_level: number; start_date: string; history: { day: number; spx?: number; v?: number }[]; forecast: FPt[]; paths?: number[][] };

const INK = "#1d1d1f", BLUE = "#0071e3", GRID = "#ececef", HAIR = "#d2d2d7", SUB = "#86868b";
const sl = { shape: "spline" as const, smoothing: 0.85 };

export function FanChart({ src = "/data/sample_fan.json", height = 360, baseline }: { src?: string; height?: number; baseline?: number[] }) {
  const [fan, setFan] = useState<Fan | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [Plot, setPlot] = useState<ComponentType<any> | null>(null);

  useEffect(() => {
    let ok = true;
    (async () => {
      const factory = (await import("react-plotly.js/factory")).default;
      // @ts-expect-error - plotly.js-dist-min ships no types
      const Plotly = (await import("plotly.js-dist-min")).default;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if (ok) setPlot(() => factory(Plotly as any) as ComponentType<any>);
    })();
    return () => { ok = false; };
  }, []);

  useEffect(() => { setFan(null); fetch(src).then((r) => r.json()).then(setFan); }, [src]);
  if (!fan || !Plot) return <div style={{ height }} />;

  const hx = fan.history.map((h) => h.day), hy = fan.history.map((h) => (h.spx ?? h.v) as number);
  const fx = fan.forecast.map((d) => d.day);
  const has = (k: keyof FPt) => fan.forecast.every((d) => typeof d[k] === "number");
  const col = (k: keyof FPt) => fan.forecast.map((d) => d[k] as number);
  const band = (lo: keyof FPt, up: keyof FPt, a: number) =>
    has(lo) && has(up) ? [
      { x: fx, y: col(up), mode: "lines", line: { width: 0, ...sl }, hoverinfo: "skip", showlegend: false },
      { x: fx, y: col(lo), mode: "lines", line: { width: 0, ...sl }, fill: "tonexty", fillcolor: `rgba(0,113,227,${a})`, hoverinfo: "skip", showlegend: false },
    ] : [];

  const hasGt = fan.forecast.every((d) => typeof d.gt === "number");
  const data = [
    ...band("p05", "p95", 0.07), ...band("p10", "p90", 0.12), ...band("p25", "p75", 0.16),
    ...(fan.paths ? fan.paths.map((p) => ({ x: fx, y: p, mode: "lines", line: { color: BLUE, width: 0.8, shape: "spline", smoothing: 0.5 }, opacity: 0.3, hoverinfo: "skip", showlegend: false })) : []),
    { x: fx, y: col("p50"), mode: "lines", line: { color: BLUE, width: 2.6, ...sl }, hovertemplate: "median %{y:.0f}<extra></extra>", showlegend: false },
    { x: hx, y: hy, mode: "lines", line: { color: INK, width: 2, shape: "spline", smoothing: 0.4 }, hovertemplate: "SPX %{y:.0f}<extra></extra>", showlegend: false },
    ...(hasGt ? [{ x: fx, y: col("gt"), mode: "lines", line: { color: INK, width: 2, dash: "dot", shape: "spline", smoothing: 0.4 }, hovertemplate: "realized %{y:.0f}<extra></extra>", showlegend: false }] : []),
    ...(baseline && baseline.length === fx.length ? [{ x: fx, y: baseline, mode: "lines", line: { color: SUB, width: 1.8, dash: "dash", ...sl }, hovertemplate: "no-narrative %{y:.0f}<extra></extra>", showlegend: false }] : []),
  ];
  const layout = {
    autosize: true, height, margin: { l: 48, r: 16, t: 8, b: 26 }, paper_bgcolor: "white", plot_bgcolor: "white",
    font: { family: "-apple-system, Inter, sans-serif", color: SUB, size: 12 },
    xaxis: { range: [hx[0], 30], tickvals: [-20, -10, 0, 10, 20, 30], ticktext: ["−20d", "−10d", "Today", "+10d", "+20d", "+30d"], showgrid: false, zeroline: false, linecolor: HAIR, fixedrange: true },
    yaxis: { showgrid: true, gridcolor: GRID, zeroline: false, tickformat: ",.0f", fixedrange: true },
    showlegend: false, shapes: [
      { type: "rect", x0: hx[0], x1: 0, y0: 0, y1: 1, yref: "paper", fillcolor: "rgba(0,0,0,0.035)", line: { width: 0 }, layer: "below" },
      { type: "line", x0: 0, x1: 0, y0: 0, y1: 1, yref: "paper", line: { color: HAIR, width: 1 } },
    ],
    annotations: [
      { x: hx[0], y: 1, yref: "paper", xanchor: "left", yanchor: "top", text: "conditioning context", showarrow: false, font: { color: SUB, size: 11 } },
      ...(hasGt ? [{ x: 30, y: 1, yref: "paper", xanchor: "right", yanchor: "top", text: "···· realized (held-out)", showarrow: false, font: { color: INK, size: 11 } }] : []),
      ...(baseline ? [{ x: 30, y: 1, yref: "paper", xanchor: "right", yanchor: "top", text: "—— narrative   – – no-narrative", showarrow: false, font: { color: SUB, size: 11 } }] : []),
    ],
    hovermode: "x unified",
  };
  return (
    <Plot data={data as never} layout={layout as never}
      config={{ displayModeBar: false, responsive: true } as never}
      style={{ width: "100%", height }} useResizeHandler />
  );
}
