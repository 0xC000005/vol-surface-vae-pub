import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FanChart } from "@/components/FanChart";

type Key = "defensive" | "riskon" | "ratesselloff";
const SCEN: Record<Key, { label: string; src: string; support: string; narrative: string; analogues: [string, number][] }> = {
  defensive: {
    label: "Defensive risk-off", src: "/data/fan_defensive.json", support: "Strong support · 38 analogues",
    narrative: "Defensive risk-off: yields grind higher, equities wobble, and a bid builds for quality and the dollar.",
    analogues: [["Oct 2008", 0.24], ["Aug 2011", 0.21], ["Aug 2015", 0.19], ["Mar 2020", 0.18], ["Feb 2018", 0.18]],
  },
  riskon: {
    label: "Fragile risk-on rebound", src: "/data/fan_riskon.json", support: "Moderate support · 29 analogues",
    narrative: "Fragile risk-on rebound: equities claw back, volatility eases, but conviction is thin and breadth is narrow.",
    analogues: [["Apr 2009", 0.22], ["Jan 2012", 0.20], ["Apr 2020", 0.20], ["Jun 2016", 0.19], ["Oct 2011", 0.19]],
  },
  ratesselloff: {
    label: "Rates selloff / tightening", src: "/data/fan_ratesselloff.json", support: "Strong support · 34 analogues",
    narrative: "Rates selloff and tightening fear: front-end yields jump, the curve reprices, and rate-sensitive assets de-rate.",
    analogues: [["Feb 2018", 0.23], ["Oct 2018", 0.21], ["Q4 2016", 0.20], ["May 2013", 0.19], ["Mar 2021", 0.17]],
  },
};

export default function Demo1() {
  const [k, setK] = useState<Key>("defensive");
  const s = SCEN[k];
  const [narrative, setNarrative] = useState(SCEN.defensive.narrative);
  const [runKey, setRunKey] = useState(0);
  const [generating, setGenerating] = useState(false);
  const [spx, setSpx] = useState<{ med: number; p10: number; p90: number } | null>(null);
  const [baseline, setBaseline] = useState<number[] | undefined>(undefined);
  const [soMed, setSoMed] = useState<number | null>(null);

  useEffect(() => {
    setSpx(null);
    fetch(s.src).then((r) => r.json()).then((f) => {
      const v = f.start_level, last = f.forecast[f.forecast.length - 1];
      setSpx({ med: (last.p50 / v - 1) * 100, p10: (last.p10 / v - 1) * 100, p90: (last.p90 / v - 1) * 100 });
    });
  }, [s.src, runKey]);

  useEffect(() => {
    fetch("/data/baseline_2017.json").then((r) => r.json()).then((f) => {
      setBaseline(f.median);
      setSoMed((f.median[f.median.length - 1] / f.start_level - 1) * 100);
    });
  }, []);

  function pick(key: Key) { setK(key); setNarrative(SCEN[key].narrative); }
  function generate() { setGenerating(true); setRunKey((r) => r + 1); window.setTimeout(() => setGenerating(false), 750); }

  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <div className="flex items-center gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">Narrative → Scenario</p>
        <Badge variant="outline" className="text-[10px] uppercase tracking-wider">Prototype demo</Badge>
      </div>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">Describe the conditions. See the scenario.</h1>
      <p className="mt-2 max-w-2xl text-lg text-muted-foreground">
        The narrative selects the closest historical regimes; the frozen generator rolls them forward into a grounded 30-day distribution.
      </p>

      <div className="mt-8 grid gap-7 md:grid-cols-[360px_1fr]">
        <Card className="h-fit p-6">
          <label htmlFor="d1-scenario" className="block text-sm font-medium">Scenario preset</label>
          <select id="d1-scenario" value={k} onChange={(e) => pick(e.target.value as Key)}
            className="mt-2 w-full rounded-lg border border-border bg-background p-2.5 text-[15px]">
            {(Object.keys(SCEN) as Key[]).map((key) => <option key={key} value={key}>{SCEN[key].label}</option>)}
          </select>
          <label htmlFor="d1-narrative" className="mt-4 block text-sm font-medium">Your narrative</label>
          <textarea id="d1-narrative" value={narrative} onChange={(e) => setNarrative(e.target.value)}
            className="mt-2 min-h-28 w-full rounded-lg border border-border bg-background p-3 text-[15px] leading-relaxed outline-none focus:ring-2 focus:ring-ring" />
          <label htmlFor="d1-start" className="mt-4 block text-xs text-muted-foreground">Starting market state (day-0)</label>
          <select id="d1-start" className="mt-1 w-full rounded-lg border border-border bg-background p-2.5 text-[15px]">
            <option>2017-07-17</option>
          </select>
          <button onClick={generate} disabled={generating}
            className="mt-5 w-full rounded-full bg-primary py-3 font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-60">
            {generating ? "Generating…" : "Generate scenarios"}
          </button>
          <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
            Pre-computed scenarios from the frozen SNI generator. This prototype isn't wired to a live backend yet.
          </p>
        </Card>

        <div className="space-y-6">
          <Card className="p-6">
            <div className="flex items-center justify-between pb-2">
              <span className="text-sm font-medium">SPX · 30-day scenario</span>
              <Badge variant="secondary">{s.support}</Badge>
            </div>
            <FanChart key={`${k}-${runKey}`} src={s.src} baseline={baseline} />
            <p className="mt-3 text-sm text-muted-foreground">
              Blue is the narrative-conditioned median; dashed grey is the <em>no-narrative</em> baseline (market state only). The narrative re-centers the scenario toward the retrieved analogues — it changes the story, not forecast accuracy.
            </p>
          </Card>

          <div className="grid gap-6 sm:grid-cols-2">
            <Card className="p-6">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Day-30 SPX outcome (model rollout)</div>
              {spx ? (
                <div className="mt-3 space-y-2 text-sm">
                  <Row k="Median (narrative)" value={spx.med} />
                  {soMed != null && <Row k="Median (no narrative)" value={soMed} />}
                  <Row k="P10 (downside)" value={spx.p10} />
                  <Row k="P90 (upside)" value={spx.p90} />
                </div>
              ) : <div className="mt-3 text-sm text-muted-foreground">loading…</div>}
              <p className="mt-3 text-xs text-muted-foreground">SPX shown is real model output; the full multi-factor panel comes from the backend.</p>
            </Card>
            <Card className="p-6">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Retrieved historical analogues</div>
              <div className="mt-3 space-y-2.5">
                {s.analogues.map(([d, w]) => (
                  <div key={d} className="flex items-center gap-3">
                    <span className="w-20 text-sm font-medium">{d}</span>
                    <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-secondary">
                      <span className="block h-full rounded-full bg-primary" style={{ width: `${(w / 0.24) * 100}%` }} />
                    </span>
                    <span className="w-9 text-right text-xs tabular-nums text-muted-foreground">{w.toFixed(2)}</span>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-xs text-muted-foreground">Illustrative weights in this prototype.</p>
            </Card>
          </div>
        </div>
      </div>
    </main>
  );
}

function Row({ k, value }: { k: string; value: number }) {
  const cls = value > 0 ? "text-[var(--chart-1)]" : value < 0 ? "text-destructive" : "";
  return (
    <div className="flex items-center justify-between border-b border-border pb-2 last:border-0">
      <span className="text-muted-foreground">{k}</span>
      <span className={"font-medium tabular-nums " + cls}>{value >= 0 ? "+" : ""}{value.toFixed(1)}%</span>
    </div>
  );
}
