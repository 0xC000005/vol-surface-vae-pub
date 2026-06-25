import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScenarioFan } from "@/components/ScenarioFan";
import { NARRATIVE_FAMILIES, FACTORS, DEFAULT_NARRATIVE, type Length } from "@/data/narratives";
import { generateScenario, fetchStartDates, fetchContext, type LiveScenario, type Series } from "@/lib/backend";

const LENGTHS: Length[] = ["short", "medium", "full"];

// per-factor display formatting (matches the original terminal table)
function meta(m: string) {
  if (/^IV/.test(m)) return { s: 100, u: "%", d: 1 };
  if (["US2Y", "US10Y", "AAA_OAS", "BBB_OAS"].includes(m)) return { s: 1, u: "%", d: 2 };
  if (m === "SPX") return { s: 1, u: "", d: 1 };
  if (["GOLD", "CRUDE_OIL"].includes(m)) return { s: 1, u: "", d: 2 };
  return { s: 1, u: "", d: 2 }; // USDJPY / DXY / VIX
}
function fmtVal(m: string, v: number) {
  const f = meta(m);
  return (v * f.s).toLocaleString(undefined, { minimumFractionDigits: f.d, maximumFractionDigits: f.d }) + f.u;
}
function scale(s: Series, k: number): Series {
  const m = (a: number[]) => (a ?? []).map((v) => v * k);
  return { p10: m(s.p10), p50: m(s.p50), p90: m(s.p90), mean: m(s.mean), paths: (s.paths ?? []).map(m) };
}

export default function Demo1() {
  const [dates, setDates] = useState<{ date: string; index: number }[]>([]);
  const [idx, setIdx] = useState<number | null>(null);
  const [narrative, setNarrative] = useState(DEFAULT_NARRATIVE);
  const [famKey, setFamKey] = useState(NARRATIVE_FAMILIES[0].key);
  const [len, setLen] = useState<Length>("short");
  const [factor, setFactor] = useState("SPX");
  const [showBase, setShowBase] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LiveScenario | null>(null);
  const [context, setContext] = useState<Record<string, number[]> | null>(null);

  useEffect(() => {
    fetchStartDates().then((ds) => {
      setDates(ds);
      if (ds.length) setIdx((ds.find((d) => d.date >= "2008-10-24") ?? ds[Math.floor(ds.length / 2)]).index);
    }).catch(() => {});
  }, []);

  function pickExample(fk: string, l: Length) {
    setFamKey(fk); setLen(l);
    const fam = NARRATIVE_FAMILIES.find((f) => f.key === fk);
    if (fam) setNarrative(fam.variants[l]);
  }

  async function generate() {
    if (idx == null || !narrative.trim()) return;
    setGenerating(true); setError(null); setContext(null);
    try {
      const res = await generateScenario(narrative, idx);
      setResult(res);
      fetchContext(idx).then(setContext).catch(() => setContext(null));
    } catch (e) {
      console.error("[live] generateScenario failed:", e);
      setError(String((e as Error)?.message || e));
      setResult(null);
    } finally { setGenerating(false); }
  }

  const ff = result?.factors[factor];
  const k = meta(factor).s;
  const selDate = dates.find((d) => d.index === idx)?.date;
  const essColor = !result ? "" : result.ess >= 3 ? "text-emerald-600" : result.ess >= 1.5 ? "text-amber-600" : "text-destructive";

  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <div className="flex items-center gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">Narrative → Scenario</p>
        <Badge variant="outline" className="text-[10px] uppercase tracking-wider">Prototype demo</Badge>
      </div>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">Describe the conditions. See the scenario.</h1>
      <p className="mt-2 max-w-2xl text-lg text-muted-foreground">
        Write a market narrative — or load one of the examples to see what the model accepts — pick a day-0 market state, and the frozen generator builds a grounded 30-day distribution across every factor.
      </p>

      <div className="mt-8 grid gap-7 lg:grid-cols-[380px_1fr]">
        {/* ---- input rail ---- */}
        <Card className="h-fit p-6">
          <label htmlFor="d1-narr" className="block text-sm font-medium">Your narrative</label>
          <textarea id="d1-narr" value={narrative} onChange={(e) => setNarrative(e.target.value)}
            className="mt-2 min-h-32 w-full rounded-lg border border-border bg-background p-3 text-[15px] leading-relaxed outline-none focus:ring-2 focus:ring-ring" />

          <div className="mt-3 rounded-lg border border-border bg-secondary/40 p-3">
            <div className="text-xs font-medium text-muted-foreground">Or load an example regime</div>
            <select aria-label="Example regime" value={famKey} onChange={(e) => pickExample(e.target.value, len)}
              className="mt-2 w-full rounded-md border border-border bg-background p-2 text-sm">
              {NARRATIVE_FAMILIES.map((f) => <option key={f.key} value={f.key}>{f.label}</option>)}
            </select>
            <div className="mt-2 flex gap-1">
              {LENGTHS.map((l) => (
                <button key={l} onClick={() => pickExample(famKey, l)}
                  className={`flex-1 rounded-md border px-2 py-1 text-xs capitalize transition-colors ${len === l ? "border-primary bg-primary/10 font-medium text-foreground" : "border-border text-muted-foreground hover:bg-secondary"}`}>{l}</button>
              ))}
            </div>
          </div>

          <label htmlFor="d1-date" className="mt-4 block text-sm font-medium">Starting market state (day-0)</label>
          <select id="d1-date" value={idx ?? ""} onChange={(e) => setIdx(Number(e.target.value))}
            className="mt-2 w-full rounded-lg border border-border bg-background p-2.5 text-[15px]">
            {dates.length === 0 && <option>loading dates…</option>}
            {dates.map((d) => <option key={d.index} value={d.index}>{d.date}</option>)}
          </select>
          <p className="mt-1 text-xs text-muted-foreground">Type a year to search · spans 2000–2016 incl. the 2008 crisis.</p>

          <label htmlFor="d1-factor" className="mt-4 block text-sm font-medium">Scenario factor</label>
          <select id="d1-factor" value={factor} onChange={(e) => setFactor(e.target.value)}
            className="mt-2 w-full rounded-lg border border-border bg-background p-2.5 text-[15px]">
            {FACTORS.map(([label, key]) => <option key={key} value={key}>{label}</option>)}
          </select>

          <button onClick={generate} disabled={generating || idx == null}
            className="mt-5 w-full rounded-full bg-primary py-3 font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-60">
            {generating ? "Generating… (≈1 min)" : "Generate 30-day scenarios"}
          </button>
          {error && <p className="mt-2 text-xs text-destructive">Generation failed: {error}. Every scenario is generated live — the backend must be running (nothing is cached).</p>}
          <p className="mt-3 text-xs leading-relaxed text-muted-foreground">Runs the live frozen SNI generator with OpenAI grounding. One run produces all factors.</p>
        </Card>

        {/* ---- output ---- */}
        <div className="space-y-6">
          <Card className="p-6">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-2">
              <span className="text-sm font-medium">{FACTORS.find(([, k2]) => k2 === factor)?.[0]} · 30-day scenario{selDate ? ` · day-0 ${selDate}` : ""}</span>
              <div className="flex items-center gap-2">
                {result && <Badge variant="secondary">{result.support}</Badge>}
                <div className="flex rounded-full border border-border p-0.5 text-xs">
                  <button onClick={() => setShowBase(false)} className={`rounded-full px-2.5 py-0.5 ${!showBase ? "bg-primary text-primary-foreground" : "text-muted-foreground"}`}>Narrative</button>
                  <button onClick={() => setShowBase(true)} className={`rounded-full px-2.5 py-0.5 ${showBase ? "bg-primary text-primary-foreground" : "text-muted-foreground"}`}>vs start-only</button>
                </div>
              </div>
            </div>
            {ff ? (
              <ScenarioFan cond={scale(ff.cond, k)} base={scale(ff.base, k)} showBase={showBase}
                history={context?.[factor]?.map((v) => v * k)} unit={meta(factor).u} />
            ) : (
              <div className="flex h-[380px] items-center justify-center text-sm text-muted-foreground">
                {generating ? "Generating live scenario…" : "Generate to see the conditioned distribution."}
              </div>
            )}
            <p className="mt-3 text-sm text-muted-foreground">
              Blue is the narrative-conditioned distribution (median, mean, P10–P90 band, sample paths). Toggle <em>vs start-only</em> to overlay the no-narrative baseline — the gap is what the narrative's retrieved analogues add.
            </p>
          </Card>

          {result && (
            <div className="grid gap-6 sm:grid-cols-2">
              <Card className="p-6">
                <div className="flex items-center justify-between">
                  <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Retrieved historical analogues</div>
                  <span className={`text-xs font-medium ${essColor}`}>Effective: {result.ess.toFixed(1)}</span>
                </div>
                <div className="mt-3 space-y-2.5">
                  {result.analogues.map((a) => {
                    const maxW = Math.max(0.001, ...result.analogues.map((x) => x.weight));
                    return (
                      <div key={a.label} className="flex items-center gap-3">
                        <span className="w-20 text-sm font-medium">{a.label}</span>
                        <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-secondary"><span className="block h-full rounded-full bg-primary" style={{ width: `${(a.weight / maxW) * 100}%` }} /></span>
                        <span className="w-12 text-right text-xs tabular-nums text-muted-foreground">ρ {a.cosine.toFixed(2)}</span>
                      </div>
                    );
                  })}
                </div>
                <p className="mt-3 text-xs text-muted-foreground">Real held-out windows the narrative retrieved · bar = support weight, ρ = cosine. Effective-analogue count (higher = more diverse).</p>
              </Card>
              <Card className="p-6">
                <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Day-30 terminal levels (per factor)</div>
                <div className="mt-3 max-h-72 overflow-y-auto">
                  <table className="w-full text-xs tabular-nums">
                    <thead className="sticky top-0 bg-card text-muted-foreground"><tr className="text-left"><th className="pb-1 font-medium">Factor</th><th className="pb-1 text-right font-medium">Base P50</th><th className="pb-1 text-right font-medium">Narr P10</th><th className="pb-1 text-right font-medium">Narr P50</th><th className="pb-1 text-right font-medium">Narr P90</th></tr></thead>
                    <tbody>
                      {FACTORS.map(([label, m]) => {
                        const fx = result.factors[m]; if (!fx) return null;
                        const last = (a: number[]) => a[a.length - 1];
                        return (<tr key={m} className={m === factor ? "bg-secondary/50" : ""}><td className="py-0.5 pr-2">{label}</td><td className="py-0.5 text-right text-muted-foreground">{fmtVal(m, last(fx.base.p50))}</td><td className="py-0.5 text-right">{fmtVal(m, last(fx.cond.p10))}</td><td className="py-0.5 text-right font-medium">{fmtVal(m, last(fx.cond.p50))}</td><td className="py-0.5 text-right">{fmtVal(m, last(fx.cond.p90))}</td></tr>);
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">Baseline (start-only) vs narrative-conditioned terminal quantiles, all factors. Pick a factor to chart it.</p>
              </Card>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
