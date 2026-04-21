# 263a Ideation Memo — Clean Next Family After 262a/262b

## Context
The clean restart line is now at:

- `260e`: deterministic co-champion at `4/11`
- `262a`: joint probabilistic latent-factor FM, `3/11`
- `262b`: same joint family + bounded history-mean EC baseline, `2/11`

The `262` family taught something useful:

- `262a` fixed the old `261d` residual-target misalignment
- `262b` proved deterministic MR can be recovered inside the joint line

But the family also exposed a clean unresolved tradeoff:

- `262a`: calibration / coverage / structure are acceptable, MR collapses
- `262b`: MR improves sharply, but dispersion and cross-cell structure collapse

In **both** models, the bounded deterministic idio mean path effectively died.

That is the decisive design signal.

## What Not To Do
Do **not** keep patching `262` with more local heads.

Examples of bad next moves:

- extra jump head
- extra regime router
- panel-space residual fallback
- more suite-specific loss terms
- more optional flags in the same model file

That would turn a still-clean family into knob soup.

## Two Credible Options

### Option A — `262c`: one last local patch
Idea:
- keep `262` exactly as-is
- add one structured deterministic mean-flexibility mechanism
- no extra stochastic branch

The only acceptable local patch would be something like:
- a tiny diagonal mean-correction path with explicit budget and direct supervision

Why this is weak:
- the deterministic idio path already died twice (`262a`, `262b`)
- another local correction starts to look like forcing life into a dead component
- high risk of accumulating knobs without fixing the real cause

Verdict:
- **not recommended** as the primary next move

### Option B — `263a`: clean joint latent state-space factor FM
Idea:
- keep the probabilistic factor thesis
- change the backbone from full-horizon latent-path prediction to an explicit **latent state-space mean evolution**

Minimal decomposition:

1. history encoder -> initial latent state `h0`
2. deterministic latent mean state `m_t` evolves recurrently
3. latent positive scale `s_t` evolves recurrently
4. standardized latent innovation `eps_t` is modeled by vanilla conditional FM
5. latent factor state: `z_t = m_t + s_t * eps_t`
6. panel change path decoded by explicit low-rank loadings

Why this is cleaner:
- it addresses the actual missing ingredient from `262`: **dynamic deterministic mean evolution**
- it avoids forcing mean flexibility through a dying panel-idio head
- it keeps the stochastic core vanilla
- it stays elegant and publishable:
  - one joint probabilistic model
  - one low-rank readout
  - no residual scenario layer
  - no router / motif / token hierarchy

Why it is better identified than `262c`:
- `262a/262b` already showed the problem is not “need one more local head”
- the real missing piece is a better deterministic mean backbone inside the joint model

Verdict:
- **recommended**

## Recommendation
Choose **`263a-v0`** next.

This is a **clean paradigm shift**, not an incremental patch:

- retire `262` as an active local-patch family after two informative endpoints
- preserve the broad principle:
  - joint probabilistic model
  - low-rank latent factors
  - vanilla FM stochastic core
- replace only the part that the evidence says is wrong:
  - the deterministic mean-path parameterization

## 263a-v0 Spec Sketch

### Core
- history GRU encoder
- recurrent latent mean state update
- recurrent latent scale state update
- FM over standardized latent innovations
- explicit low-rank loadings

### Keep
- `asinh_local_scale` change coordinate
- no frozen deterministic core
- no panel residual scenario layer
- no extra jump branch

### Kill Criteria
- if MR remains below `0.50`, the deterministic state backbone is still too weak
- if coverage falls below `0.70`, the scale path is over-collapsing again
- if `corr_ratio` falls below `0.50`, the readout / latent dynamics are losing common structure

## Decision
Next iteration should be:

- `paradigm_shift` to `263a`
- then `experiment`: implement `263a-v0`
