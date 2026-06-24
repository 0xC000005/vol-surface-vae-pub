import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FanChart } from "@/components/FanChart";

const SUPPORTS = [
  ["Oct 2008", 0.24], ["Aug 2011", 0.21], ["Aug 2015", 0.19], ["Feb 2018", 0.19],
] as const;

export default function Demo2() {
  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <div className="flex items-center gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">Scenario → Narrative</p>
        <Badge variant="outline" className="text-[10px] uppercase tracking-wider">Prototype demo</Badge>
      </div>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">Start from a scenario. Recover its story.</h1>
      <p className="mt-2 max-w-2xl text-lg text-muted-foreground">
        Pick a scenario; the workbench writes a grounded narrative from its retrieved historical supports — never from realized future data.
      </p>

      <div className="mt-8 grid gap-7 md:grid-cols-[360px_1fr]">
        <Card className="h-fit p-6">
          <div className="text-sm font-medium">Choose a scenario</div>
          <label htmlFor="d2-source" className="mt-3 block text-xs text-muted-foreground">Source</label>
          <select id="d2-source" className="mt-1 w-full rounded-lg border border-border bg-background p-2.5 text-[15px]">
            <option>Historical case</option><option>Uploaded scenario (paste factors)</option>
          </select>
          <label htmlFor="d2-period" className="mt-4 block text-xs text-muted-foreground">Starting period</label>
          <select id="d2-period" className="mt-1 w-full rounded-lg border border-border bg-background p-2.5 text-[15px]">
            <option>2017-01-06 → +30d</option><option>2011-08-04 → +30d</option><option>2020-02-21 → +30d</option>
          </select>
          <button className="mt-5 w-full rounded-full border border-border py-2.5 font-medium transition-colors hover:bg-secondary">
            Step 1 · Visualize scenario
          </button>
          <button className="mt-3 w-full rounded-full bg-primary py-3 font-medium text-primary-foreground transition-opacity hover:opacity-90">
            Step 2 · Generate narrative
          </button>
          <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
            In the full system the narrative author (Codex) writes from the retrieved supports. This prototype shows a pre-generated sample, not wired to a live backend yet.
          </p>
        </Card>

        <div className="space-y-6">
          <Card className="p-6">
            <div className="flex items-center justify-between pb-2">
              <span className="text-sm font-medium">Selected scenario · SPX</span>
              <Badge variant="secondary">Strong support · 42 analogues</Badge>
            </div>
            <FanChart />
          </Card>

          <Card className="p-6">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Generated narrative — sample output</div>
            <div className="mt-3 space-y-3 text-[15px] leading-relaxed">
              <p>
                Conditions resemble a <strong>defensive risk-off regime</strong>. Equities drift lower with a widening downside, implied volatility lifts and stays elevated, and rates show a mild flight-to-quality bid as the dollar firms — a coherent move toward quality rather than a single-asset shock.
              </p>
              <p>
                Dispersion is wide but the centre is only moderately negative: conditions like these have historically resolved more through <em>uncertainty</em> than a committed directional collapse. The downside tail is real, but so is a meaningful chance of stabilisation.
              </p>
            </div>
            <div className="mt-5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Grounded in</div>
            <div className="mt-3 space-y-2.5">
              {SUPPORTS.map(([d, w]) => (
                <div key={d} className="flex items-center gap-3">
                  <span className="w-20 text-sm font-medium">{d}</span>
                  <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-secondary">
                    <span className="block h-full rounded-full bg-primary" style={{ width: `${(w / 0.24) * 100}%` }} />
                  </span>
                  <span className="w-9 text-right text-xs tabular-nums text-muted-foreground">{w.toFixed(2)}</span>
                </div>
              ))}
            </div>
            <p className="mt-4 text-xs leading-relaxed text-muted-foreground">
              Every claim is traceable to the weighted analogues; no realized future was used. When a scenario sits outside historical precedent, the narrative says so plainly.
            </p>
          </Card>
        </div>
      </div>
    </main>
  );
}
