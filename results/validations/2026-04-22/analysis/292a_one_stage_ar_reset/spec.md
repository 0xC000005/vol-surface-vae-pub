## 292a One-Stage Probabilistic AR Reset

### Context
The explicit two-level Stage A / Stage B program is being retired as the active search tree.

Why:
- the current deterministic Stage A line (`289e -> 290 -> 291`) became an output-representation search
- the repo already showed that "strong H=1 model + AR rollout" is not enough by itself (`212 -> 221/222/223/226/227/233`)
- the more fundamental conceptual issue is that **realized next-day changes may be largely irreducible**, while the **conditional law** remains predictable

That means the active target should not be:
- deterministic next-day realized move
- plus residual scenario layer later

It should be:
- a single model that learns the **conditional next-step law**
- and rolls forward autoregressively

### Decision
Next step: `292a-v0`

### Family
One-stage probabilistic autoregressive latent state-space model.

This is:
- AR
- not one-shot
- not retrieval
- not explicit Stage A / Stage B decomposition

### Core idea
At each future step:
1. maintain a persistent hidden state summarizing the episode
2. infer or sample a latent innovation variable
3. decode a **distribution** over the next normalized change panel
4. sample the next change
5. update the recurrent state using the generated observation

This lets the model decide:
- what is predictable in the conditional state
- what is irreducible innovation

without forcing a deterministic center-path target first.

### Minimal 292a-v0 architecture
History encoder:
- GRU over recent normalized levels and changes
- initializes recurrent hidden state

State evolution:
- recurrent hidden state updated every generated day
- update uses previous observation embedding and sampled latent innovation

Latent law:
- prior network `p(z_t | h_t, o_t)`
- posterior network `q(z_t | h_t, o_t, x_{t+1})` during training

Emission:
- decoder outputs mean and log-scale for the next normalized-change panel
- stochasticity comes from both shared latent `z_t` and diagonal emission noise

Rollout:
- autoregressive for 30 days
- generated changes update generated levels
- generated observations feed back into state

### Training objective
Multi-step sequential variational objective:
- negative log-likelihood of realized next normalized changes
- KL between posterior and prior at each step

No:
- deterministic Stage A target
- retrieval bank
- hard low-rank output head
- bounded side paths
- hand-coded error correction

### Why this is the most principled reset
It directly addresses the conceptual failure in the current line:
- if realized one-day change is largely unpredictable, deterministic Stage A is misspecified

`292a` instead learns:
- conditional latent state
- conditional innovation law
- and full AR scenario rollout

in one coherent probabilistic model.

### Baseline scope
`292a-v0` is intentionally minimal:
- diagonal emission noise
- shared latent innovation
- recurrent hidden state
- no extra slow/fast path split yet

If even this baseline cannot beat the current deterministic local-law ceiling, that is valuable evidence.

### Kill criteria
`292a` is only alive if it improves at least one category the deterministic world-model line could not:
- nonzero and meaningful coverage
- better conditionality/regime differentiation
- while keeping at least usable shared structure (`corr_ratio > 0.5`)

If it fails purely by reverting to the old AR dampening pathologies, the one-stage AR reset is falsified quickly.
