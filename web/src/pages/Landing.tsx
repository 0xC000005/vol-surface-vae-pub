import { Link } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { FanChart } from "@/components/FanChart";
import { FactorGrid } from "@/components/FactorGrid";

function WorkflowVideo({ src }: { src: string }) {
  return <video src={src} autoPlay loop muted playsInline className="w-full rounded-lg border border-border bg-white" />;
}

export default function Landing() {
  return (
    <main>
      {/* HERO — foundation, video analogy */}
      <section className="mx-auto max-w-6xl px-6 pb-20 pt-28 text-center">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">A foundational generative model for markets</p>
        <h1 className="mx-auto mt-4 max-w-3xl text-5xl font-semibold leading-[1.06] tracking-tight">
          Realistic, multi-factor, multi-day market scenarios.
        </h1>
        <p className="mx-auto mt-5 max-w-2xl text-xl leading-relaxed text-muted-foreground">
          Built on the same generative methodology as Sora, GPT-Image and Nano Banana — transforming
          Gaussian noise into data — fine-tuned on financial markets. One foundational model generates
          realistic market scenario distributions across factors and horizons, conditioned on market
          history. Not a traditional quant model with imposed dynamics, and not a general-purpose generator.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link to="/demo1" className="flex items-center gap-2 rounded-full border border-[color-mix(in_oklab,var(--primary)_40%,var(--border))] bg-[color-mix(in_oklab,var(--primary)_8%,white)] px-5 py-3 text-[17px] font-medium text-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--primary)_15%,white)]">
            <span className="rounded-full bg-[color-mix(in_oklab,var(--primary)_12%,white)] px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wider text-primary">Demo</span>
            Narrative → Scenario
          </Link>
          <Link to="/demo2" className="flex items-center gap-2 rounded-full border border-[color-mix(in_oklab,var(--primary)_40%,var(--border))] bg-[color-mix(in_oklab,var(--primary)_8%,white)] px-5 py-3 text-[17px] font-medium text-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--primary)_15%,white)]">
            <span className="rounded-full bg-[color-mix(in_oklab,var(--primary)_12%,white)] px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wider text-primary">Demo</span>
            Scenario → Narrative
          </Link>
        </div>
      </section>

      {/* FOUNDATION — generating scenarios */}
      <section className="border-y border-border bg-secondary/40 py-20">
        <div className="mx-auto max-w-6xl px-6">
          <h2 className="text-center text-3xl font-semibold tracking-tight">Generating market scenarios</h2>
          <p className="mx-auto mt-3 max-w-2xl text-center text-lg text-muted-foreground">
            The same generative recipe behind frontier image and video models — transport Gaussian noise into data — applied to markets.
          </p>
          <div className="mt-10 grid gap-6 md:grid-cols-2">
            <Card className="flex items-center justify-center p-6"><WorkflowVideo src="/anim/image_gen.mp4" /></Card>
            <Card className="flex items-center justify-center p-6"><WorkflowVideo src="/anim/market_gen.mp4" /></Card>
          </div>
          <p className="mx-auto mt-3 max-w-2xl text-center text-[13px] text-muted-foreground">
            Pixels for an image; market factors for a scenario — the same noise-to-data process.
          </p>
          <Card className="mt-6 p-6">
            <div className="pb-2 text-sm font-medium">The learned forecast vs. what happened</div>
            <FanChart src="/data/fan_startonly.json" height={320} />
            <p className="mt-2 text-xs text-muted-foreground">The generator's SPX forecast from market state alone (no narrative), at a held-out 2016 window. The shaded band is the conditioning context; the dotted line is the realized path it never saw — the fan brackets it.</p>
          </Card>
          <div className="mt-8 grid gap-6 md:grid-cols-3">
            {([
              ["Individually realistic", "Each sampled path is a plausible market trajectory, drawn from the generator rather than averaged."],
              ["Learned from data", "Volatility, drift and co-movement are learned from history — not imposed by hand."],
              ["Cross-factor coherent", "Equities, rates, volatility and FX move together as they historically do."],
            ] as const).map(([t, d]) => (
              <div key={t}>
                <div className="text-base font-semibold">{t}</div>
                <p className="mt-1.5 text-[15px] leading-relaxed text-muted-foreground">{d}</p>
              </div>
            ))}
          </div>

          <div className="mt-12">
            <div className="text-center text-base font-semibold">One law, every factor — and what actually happened</div>
            <p className="mx-auto mt-1.5 mb-6 max-w-2xl text-center text-[15px] leading-relaxed text-muted-foreground">
              The generator forecasts all factors jointly from market state. The dotted line is each factor's realized 30-day path, held out from the model — it lands inside the forecast for most of them.
            </p>
            <Card className="p-6"><FactorGrid src="/data/panel_startonly.json" /></Card>
          </div>
        </div>
      </section>

      {/* APPLICATIONS — what you can do with it */}
      <section className="mx-auto max-w-6xl px-6 py-20">
        <p className="text-center text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Built on the foundation</p>
        <h2 className="mt-3 text-center text-3xl font-semibold tracking-tight">What you can do with it</h2>
        <p className="mx-auto mt-3 max-w-2xl text-center text-lg text-muted-foreground">
          Once the scenario generator is learned, you can condition it — and invert it.
        </p>

        <div className="mt-10 grid gap-6 md:grid-cols-2">
          <Card className="flex flex-col p-6">
            <div className="text-lg font-semibold">Narrative → Scenario</div>
            <p className="mt-1.5 text-[15px] leading-relaxed text-muted-foreground">
              Condition the generator on a natural-language narrative: embed it, retrieve the nearest historical regimes, and roll them forward.
            </p>
            <div className="my-5"><WorkflowVideo src="/anim/NarrativeToScenario.mp4" /></div>
            <Link to="/demo1" className="mt-auto text-[15px] font-medium text-primary">Open the demo ›</Link>
          </Card>
          <Card className="flex flex-col p-6">
            <div className="text-lg font-semibold">Scenario → Narrative</div>
            <p className="mt-1.5 text-[15px] leading-relaxed text-muted-foreground">
              Invert it: take any scenario, match the historical supports behind it, and write a grounded explanation in plain language.
            </p>
            <div className="my-5"><WorkflowVideo src="/anim/ScenarioToNarrative.mp4" /></div>
            <Link to="/demo2" className="mt-auto text-[15px] font-medium text-primary">Open the demo ›</Link>
          </Card>
        </div>
      </section>

      <footer className="border-t border-border py-10">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 text-sm text-muted-foreground">
          <span>Conditional market scenario generation · research prototype</span>
          <span className="flex gap-5">
            <a href="#" title="placeholder" className="hover:text-foreground">Paper</a>
            <a href="#" title="placeholder" className="hover:text-foreground">GitHub</a>
          </span>
        </div>
      </footer>
    </main>
  );
}
