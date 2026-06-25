import { useState, type ReactNode } from "react";

type Section = { title: string; positive: string; negative: string };

// The Codex packet is one markdown doc: an intro, then N "### <n>. <title>" sections, each with a
// **Positive** (faithful) read and a **Hard negative** (deliberately-contradictory) foil. Parse it
// into structured sections so the UI can present a format selector + side-by-side read/foil instead
// of one long scrolling blob.
function parsePacket(md: string): { intro: string; sections: Section[] } {
  const bySection = md.split(/\n###\s+/);
  const intro = bySection[0].replace(/^#{1,3}\s+.*$/m, "").trim();
  const sections = bySection.slice(1).map((block) => {
    const nl = block.indexOf("\n");
    const title = (nl >= 0 ? block.slice(0, nl) : block).trim();
    const body = nl >= 0 ? block.slice(nl) : "";
    const parts = body.split(/\*\*Hard[- ]negative\*\*/i);
    const positive = parts[0].replace(/\*\*Positive\*\*/i, "").trim();
    const negative = (parts[1] ?? "").trim();
    return { title, positive, negative };
  }).filter((s) => s.positive || s.negative);
  return { intro, sections };
}

function inline(text: string): ReactNode {
  return text.split("\n").filter((l) => l.trim()).map((ln, i) => (
    <p key={i} className="mt-2 first:mt-0">
      {ln.split(/(\*\*[^*]+\*\*)/g).map((seg, j) => (seg.startsWith("**") ? <strong key={j}>{seg.slice(2, -2)}</strong> : seg))}
    </p>
  ));
}

export function NarrativePacketView({ md }: { md: string }) {
  const { intro, sections } = parsePacket(md);
  const [sel, setSel] = useState(0);
  if (!sections.length) {
    // fallback: render raw if the structure is unexpected
    return <div className="mt-3 max-h-[28rem] overflow-y-auto whitespace-pre-wrap text-[14px] leading-relaxed">{md}</div>;
  }
  const s = sections[Math.min(sel, sections.length - 1)];
  return (
    <div className="mt-3">
      {intro && <p className="text-xs leading-relaxed text-muted-foreground">{intro}</p>}
      <div className="mt-3">
        <label htmlFor="np-fmt" className="text-xs font-medium text-muted-foreground">Narrative format</label>
        <select id="np-fmt" value={sel} onChange={(e) => setSel(Number(e.target.value))}
          className="mt-1 w-full rounded-lg border border-border bg-background p-2 text-sm">
          {sections.map((sec, i) => <option key={i} value={i}>{sec.title}</option>)}
        </select>
      </div>
      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        <div className="rounded-lg border border-border p-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-emerald-700">Positive · faithful read</div>
          <div className="mt-2 text-[14px] leading-relaxed text-foreground">{inline(s.positive)}</div>
        </div>
        <div className="rounded-lg border border-destructive/30 bg-destructive/[0.03] p-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-destructive/80">Hard-negative · deliberately wrong</div>
          <div className="mt-2 text-[13px] leading-relaxed text-muted-foreground">{s.negative ? inline(s.negative) : <span className="italic">No hard-negative for this format.</span>}</div>
        </div>
      </div>
      <p className="mt-3 text-xs text-muted-foreground">{sections.length} narrative formats generated — switch above. The hard-negative is a near-miss the read must be distinguishable from; it is <em>not</em> a valid reading of this scenario.</p>
    </div>
  );
}
