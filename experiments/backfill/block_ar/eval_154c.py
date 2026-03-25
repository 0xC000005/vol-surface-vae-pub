#!/usr/bin/env python
"""Quick test eval for 154c conditional residual FM."""
import numpy as np, torch, sys
from scipy.stats import ks_2samp, kurtosis
from statsmodels.tsa.stattools import coint
sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    ConditionalFactoredVelocityTransformer, load_encoder, normalize_iv
)

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

bc = torch.load('models/backfill/flow_153a/final_model.pt', weights_only=False, map_location=device)
base = ConditionalFactoredVelocityTransformer(n_frames=30,n_cells=25,d_model=128,n_heads=4,n_layers=4,cond_dim=128)
base.load_state_dict(bc['model_state_dict']); base.to(device).eval()
enc, cd = load_encoder('models/backfill/block_ar_vol_scaled_30ep/best_model.pt', device)
bm = torch.from_numpy(bc['train_mean']).float().to(device)
bs = torch.from_numpy(bc['train_std']).float().to(device)

rc = torch.load('models/backfill/flow_154c/final_model.pt', weights_only=False, map_location=device)
rcfg = rc['config']
res = ConditionalFactoredVelocityTransformer(n_frames=rcfg['n_frames'],n_cells=rcfg['n_cells'],
    d_model=rcfg['d_model'],n_heads=rcfg['n_heads'],n_layers=rcfg['n_layers'],cond_dim=rcfg['cond_dim'])
res.load_state_dict(rc['model_state_dict']); res.to(device).eval()
rm = torch.from_numpy(rc['res_mean']).float().to(device)
rs = torch.from_numpy(rc['res_std']).float().to(device)

data = np.load('data/vol_surface_with_ret.npz'); surfaces = data['surface']; rets = data['ret']
H,T,DIM = 30,30,750; ts = 4540; ns = 50; nst = 8; dt = 1.0/nst

tw = [(surfaces[i:i+H], surfaces[i+H:i+H+T]) for i in range(ts, min(ts+160, len(surfaces)-H-T+1))]
N = len(tw); print(f'Eval 154c: {N} win, {ns} samp')

aS = []; aG = []
with torch.no_grad():
    for w in range(N):
        h = torch.from_numpy(tw[w][0][None].astype(np.float32)).to(device)
        c = enc(normalize_iv(h))
        bp = []
        for _ in range(3):
            x = torch.randn(1, DIM, device=device)
            for s in range(nst):
                t = torch.full((1,), s*dt, device=device); x = x + base(x, t, cond=c) * dt
            bp.append((x * bs + bm).clamp(0, 1))
        bp = torch.stack(bp).mean(0)
        samps = []
        for _ in range(ns):
            x = torch.randn(1, DIM, device=device)
            for s in range(nst):
                t = torch.full((1,), s*dt, device=device); x = x + res(x, t, cond=c) * dt
            combined = (bp + x * rs + rm).clamp(0, 1).cpu().numpy().reshape(T, 5, 5)
            samps.append(combined)
        aS.append(np.array(samps)); aG.append(tw[w][1])
        if (w+1) % 40 == 0: print(f'  {w+1}/{N}')

cs = np.array(aS); gt = np.array(aG)

rv = np.array([np.std(rets[i:i+H]) for i in range(ts, ts+N)])
tm = rv > np.percentile(rv, 80); cm = rv < np.percentile(rv, 20)
tc = cs[tm].std(axis=1).mean() / (cs[cm].std(axis=1).mean() + 1e-8)
wc = 1.0; cih = 0
for h2 in range(T):
    lo = np.percentile(cs[:,:,h2], 5, axis=1); hi = np.percentile(cs[:,:,h2], 95, axis=1)
    if ((gt[:,h2] >= lo) & (gt[:,h2] <= hi)).mean() >= 0.85: cih += 1
for r in range(5):
    for c2 in range(5):
        lo = np.percentile(cs[:,:,:,r,c2], 5, axis=1); hi = np.percentile(cs[:,:,:,r,c2], 95, axis=1)
        wc = min(wc, ((gt[:,:,r,c2] >= lo) & (gt[:,:,r,c2] <= hi)).mean())
gch = np.diff(cs[:,0], axis=1).reshape(-1, 25); gtch = np.diff(gt, axis=1).reshape(-1, 25)
kr = kurtosis(gch.flatten()) / (kurtosis(gtch.flatten()) + 1e-6)
spr = [cs[:,:,h3].std(axis=1).mean() for h3 in range(T)]
mo = sum(1 for i in range(len(spr)-1) if spr[i+1] >= spr[i] * 0.97)
gc = np.corrcoef(gch.T); gtc = np.corrcoef(gtch.T)
def efr(corr):
    ev = np.linalg.eigvalsh(corr)[::-1]; ev = np.maximum(ev, 0)
    p = ev / (ev.sum() + 1e-10); p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))
erg = efr(gc); ergt = efr(gtc); rr = erg/ergt
crr = np.abs(gc).mean() / (np.abs(gtc).mean() + 1e-6)
ksd = sum(1 for c3 in range(25) if ks_2samp(gch[:,c3], gtch[:,c3])[0] < 0.15)
gl = cs[:,0,-1].reshape(-1, 25); gtl = gt[:,-1].reshape(-1, 25)
ksl = sum(1 for c3 in range(25) if ks_2samp(gl[:,c3], gtl[:,c3])[0] < 0.15)
cp = 0; ct = 0
for i in range(min(N, 30)):
    sp = cs[i,0].reshape(T, 25)
    for c1 in range(0, 25, 5):
        for c2 in range(c1+1, min(c1+5, 25)):
            try: _, pv, _ = coint(sp[:,c1], sp[:,c2]); cp += int(pv < 0.05); ct += 1
            except: ct += 1
co = cp / max(ct, 1)

print(f'CI worst_cell: {wc:.3f}')
print(f'CI horizon_pass: {cih}/30')
print(f'Turb/calm: {tc:.3f}')
print(f'Kurt: {kr:.3f}')
print(f'KS daily: {ksd}/25, levels: {ksl}/25')
print(f'eff_rank: {erg:.2f}(GT={ergt:.2f}) ratio={rr:.3f}')
print(f'Corr ratio: {crr:.3f}')
print(f'Coint: {co:.3f}')
print(f'Mono: {mo}/29')
print(f'Spread h1={spr[0]:.5f} h30={spr[-1]:.5f}')
