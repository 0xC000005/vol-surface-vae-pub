import { useEffect, useMemo, useState } from "react";

type Pt = { x: number; y: number; c: number };
type Nbr = { i: number; cos: number; rank2d: number };
type Emb = {
  dims: number; n: number; pca_var2: number[];
  points: Pt[]; query: number; neighbors: Nbr[]; far_neighbor: number; far_rank2d: number;
};

const W = 460, H = 360, PAD = 26;
const CLUSTER = ["var(--chart-1)", "var(--chart-3)", "var(--muted-foreground)", "var(--chart-4)", "var(--chart-2)"];

export function EmbeddingWidget() {
  const [d, setD] = useState<Emb | null>(null);
  const [reveal, setReveal] = useState(true);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => { fetch("/data/embedding.json").then((r) => r.json()).then(setD); }, []);

  const scale = useMemo(() => {
    if (!d) return null;
    const xs = d.points.map((p) => p.x), ys = d.points.map((p) => p.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const sx = (x: number) => PAD + ((x - x0) / (x1 - x0)) * (W - 2 * PAD);
    const sy = (y: number) => PAD + ((y1 - y) / (y1 - y0)) * (H - 2 * PAD);
    return { sx, sy };
  }, [d]);

  if (!d || !scale) return <div style={{ height: H }} />;
  const { sx, sy } = scale;
  const q = d.points[d.query];
  const nbrIdx = new Set(d.neighbors.map((n) => n.i));

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: H }}>
        {reveal && d.neighbors.map((n) => {
          const p = d.points[n.i];
          const far = n.i === d.far_neighbor;
          return (
            <line key={"l" + n.i} x1={sx(q.x)} y1={sy(q.y)} x2={sx(p.x)} y2={sy(p.y)}
              stroke={far ? "var(--destructive)" : "var(--chart-1)"} strokeWidth={far ? 1.6 : 1.2}
              strokeOpacity={0.7} strokeDasharray={far ? "4 3" : undefined} />
          );
        })}
        {d.points.map((p, i) => {
          const isQ = i === d.query, isN = nbrIdx.has(i), isFar = i === d.far_neighbor;
          const r = isQ ? 6 : hover === i ? 5 : isN ? 4 : 2.6;
          return (
            <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={r}
              fill={isQ ? "var(--foreground)" : isFar ? "var(--destructive)" : CLUSTER[p.c % 5]}
              opacity={isQ || isN ? 0.95 : 0.5}
              stroke={isQ || isN ? "white" : "none"} strokeWidth={1}
              onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
          );
        })}
        {reveal && (
          <text x={sx(d.points[d.far_neighbor].x)} y={sy(d.points[d.far_neighbor].y) - 10}
            fontSize={11} fill="var(--destructive)" textAnchor="middle" fontWeight={600}>
            true neighbor — looks far here
          </text>
        )}
      </svg>
      <div className="mt-2 flex items-center justify-between px-1">
        <span className="text-xs text-muted-foreground">
          {d.n} real narrative embeddings · {d.dims}-D → PCA 2D ({Math.round(d.pca_var2[0] * 100 + d.pca_var2[1] * 100)}% var)
        </span>
        <button onClick={() => setReveal((v) => !v)}
          className="rounded-full border border-border px-3 py-1 text-xs font-medium hover:bg-secondary">
          {reveal ? "Hide retrieval" : "Show retrieval"}
        </button>
      </div>
    </div>
  );
}
