import { useEffect, useState, type ReactNode } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { generateNarrative, fetchHistoricalDates, fetchScenarioMovement, type NarrativePacket, type FactorRow } from "@/lib/backend";
import { Sparkline } from "@/components/Sparkline";

// minimal markdown render for the live narrative packet (## headings, **bold**)
function renderMd(md: string): ReactNode {
  const bold = (s: string) => s.split(/(\*\*[^*]+\*\*)/g).map((seg, j) => (seg.startsWith("**") ? <strong key={j}>{seg.slice(2, -2)}</strong> : seg));
  return md.split("\n").map((ln, i) => {
    if (ln.startsWith("### ")) return <h4 key={i} className="mt-3 text-sm font-semibold">{ln.slice(4)}</h4>;
    if (ln.startsWith("## ")) return <h3 key={i} className="mt-4 text-base font-semibold">{ln.slice(3)}</h3>;
    if (!ln.trim()) return null;
    return <p key={i} className="mt-1.5 text-[14px] leading-relaxed">{bold(ln)}</p>;
  });
}

const DIR_COLOR: Record<string, string> = { up: "text-emerald-600", down: "text-destructive", higher: "text-emerald-600", lower: "text-destructive" };

export default function Demo2() {
  const [dates, setDates] = useState<string[]>([]);
  const [date, setDate] = useState<string>("");
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scenario, setScenario] = useState<FactorRow[] | null>(null);
  const [movement, setMovement] = useState<Record<string, number[]> | null>(null);
  const [packet, setPacket] = useState<NarrativePacket | null>(null);

  useEffect(() => {
    fetchHistoricalDates().then((ds) => {
      setDates(ds);
      if (ds.length) setDate(ds.find((d) => d >= "2008-09-22") ?? ds[Math.floor(ds.length / 2)]);
    }).catch(() => {});
  }, []);

  async function generate() {
    if (!date) return;
    setGenerating(true); setError(null); setScenario(null); setPacket(null); setMovement(null);
    fetchScenarioMovement(date).then(setMovement).catch(() => setMovement(null));
    try { setPacket(await generateNarrative(date, setScenario)); }
    catch (e) { console.error("[live] generateNarrative failed:", e); setError(String((e as Error)?.message || e)); setScenario(null); setPacket(null); }
    finally { setGenerating(false); }
  }

  function download() {
    if (!packet) return;
    const blob = new Blob([packet.markdown], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = `narrative_${date}.md`; a.click();
    URL.revokeObjectURL(a.href);
  }

  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <div className="flex items-center gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">Scenario → Narrative</p>
        <Badge variant="outline" className="text-[10px] uppercase tracking-wider">Prototype demo</Badge>
      </div>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">Start from a scenario. Recover its story.</h1>
      <p className="mt-2 max-w-2xl text-lg text-muted-foreground">
        Pick any historical 30-day episode; the workbench reads its factor moves and an LLM author writes a grounded risk narrative — each paired with a deliberately-contradictory hard-negative so the read is confirmable. Never from realized future data.
      </p>

      <div className="mt-8 grid gap-7 lg:grid-cols-[380px_1fr]">
        {/* ---- input rail ---- */}
        <Card className="h-fit p-6">
          <label htmlFor="d2-date" className="block text-sm font-medium">Historical scenario — start date</label>
          <select id="d2-date" value={date} onChange={(e) => { setDate(e.target.value); }}
            className="mt-2 w-full rounded-lg border border-border bg-background p-2.5 text-[15px]">
            {dates.length === 0 && <option>loading dates…</option>}
            {dates.map((d) => <option key={d} value={d}>{d} → +30d</option>)}
          </select>
          <p className="mt-1 text-xs text-muted-foreground">Type a year to search · the workbench uses the following 30 observed market days.</p>

          <button onClick={generate} disabled={generating || !date}
            className="mt-5 w-full rounded-full bg-primary py-3 font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-60">
            {generating ? "Generating… (≈2 min)" : "Generate narrative"}
          </button>
          {error && <p className="mt-2 text-xs text-destructive">Generation failed: {error}. Every packet is generated live (retrieve supports → LLM author) — the backend must be running (nothing is cached).</p>}
          {packet && <button onClick={download} className="mt-3 w-full rounded-full border border-border py-2.5 text-sm font-medium transition-colors hover:bg-secondary">Download narrative packet (.md)</button>}
          <p className="mt-3 text-xs leading-relaxed text-muted-foreground">Step 1 retrieves the scenario's factor moves; Step 2 runs the live LLM author (Codex) over the retrieved supports.</p>
        </Card>

        {/* ---- output ---- */}
        <div className="space-y-6">
          <Card className="p-6">
            <div className="flex items-center justify-between pb-2">
              <span className="text-sm font-medium">Numerical scenario{date ? ` · ${date} → +30d` : ""}</span>
              {scenario && <Badge variant="secondary">{scenario.length} factors</Badge>}
            </div>
            {scenario ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm tabular-nums">
                  <thead className="text-muted-foreground"><tr className="border-b border-border text-left"><th className="py-1.5 pr-3 font-medium">Factor</th><th className="py-1.5 pr-3 text-right font-medium">Start</th><th className="py-1.5 pr-3 text-right font-medium">End</th><th className="py-1.5 pr-3 text-right font-medium">Δ</th><th className="py-1.5 pr-3 font-medium">Direction</th><th className="py-1.5 pr-3 font-medium">Magnitude</th><th className="py-1.5 font-medium">30-day movement</th></tr></thead>
                  <tbody>
                    {scenario.map((r) => (
                      <tr key={r.factor} className="border-b border-border/50"><td className="py-1.5 pr-3 font-medium">{r.factor}</td><td className="py-1.5 pr-3 text-right">{r.start}</td><td className="py-1.5 pr-3 text-right">{r.end}</td><td className="py-1.5 pr-3 text-right">{r.delta}</td><td className={`py-1.5 pr-3 ${DIR_COLOR[r.direction?.toLowerCase()] ?? ""}`}>{r.direction}</td><td className="py-1.5 pr-3 text-muted-foreground">{r.magnitude}</td><td className="py-1.5"><Sparkline values={movement?.[r.factor]} /></td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="flex h-48 items-center justify-center text-sm text-muted-foreground">{generating ? "Reading the scenario's factor moves…" : "Generate to read the scenario's factor moves."}</div>
            )}
          </Card>

          <Card className="p-6">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Generated narrative {packet ? "— live" : ""}</div>
            {packet ? (
              <div className="mt-3 max-h-[30rem] overflow-y-auto pr-1">{renderMd(packet.markdown)}</div>
            ) : (
              <div className="mt-3 flex h-40 items-center justify-center text-sm text-muted-foreground">{generating ? "LLM author writing the packet (≈2 min)…" : "The grounded narrative + hard-negative appear here."}</div>
            )}
            <p className="mt-4 text-xs leading-relaxed text-muted-foreground">Every claim is traceable to the scenario's factor moves; no realized future was used. The hard-negative is a deliberately-contradictory read used to confirm the scenario is distinguishable.</p>
          </Card>
        </div>
      </div>
    </main>
  );
}
