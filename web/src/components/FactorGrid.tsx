import { useEffect, useState } from "react";
import { MiniFan } from "./MiniFan";

type Factor = Parameters<typeof MiniFan>[0]["factor"];
type Panel = { scenario: string; factors: Factor[] };

export function FactorGrid({ src = "/data/panel_defensive.json" }: { src?: string }) {
  const [p, setP] = useState<Panel | null>(null);
  useEffect(() => { fetch(src).then((r) => r.json()).then(setP); }, [src]);
  if (!p) return <div style={{ height: 240 }} />;
  return (
    <div className="grid grid-cols-2 gap-x-8 gap-y-5 sm:grid-cols-3">
      {p.factors.map((f) => <MiniFan key={f.label} factor={f} />)}
    </div>
  );
}
