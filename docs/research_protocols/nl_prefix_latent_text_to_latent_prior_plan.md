# Narrative Text-To-Latent Prior Plan

Last updated: 2026-05-11

This note records the next research direction for the narrative-conditioned
scenario generator after the generator-calibrated support rerank failed seed
stability. It is a protocol note, not a promoted production claim.

## Local Evidence

The current promoted bridge remains:

```text
mlp_mse_contrastive + multi_caption_with_negatives
```

on the representative OpenAI-labeled manifest:

- 182 usable narrative windows;
- 2,293 text examples after anchors, positive captions, and hard negatives;
- OpenAI `text-embedding-3-small` embeddings;
- frozen joint39 SNI generator-memory targets of dimension 128;
- held-out target cosine around 0.856 across seeds;
- hard-negative gap around 0.936 across seeds.

The useful mechanism is not a pure CLIP replacement. Local evidence says:

- multi-caption training materially improves generator-memory alignment versus
  anchor-only captions;
- hard negatives are the strongest directional mechanism and fix the earlier
  "rates up" versus "rates down" semantic-nearness issue;
- the tested CLIP/InfoNCE hybrid improves recall@1 in some settings but loses
  target-memory cosine and directional hard-negative separation;
- recent support-policy reranks can look better under one rollout seed but do
  not survive seed stability.

The next move should therefore improve the text-to-latent training target, not
add another support-ranking knob.

## Primary-Source Literature Read

- CLIP shows that the important idea is paired multimodal alignment: predict
  which caption matches which image among alternatives, rather than merely use
  an off-the-shelf text embedding. Source:
  https://openai.com/index/clip/ and
  https://icml.cc/virtual/2021/oral/9194.
- DALL-E 2 is closer to our target than plain CLIP: it uses a prior from text
  to a learned image latent, then a decoder conditioned on that latent. Source:
  https://openai.com/index/hierarchical-text-conditional-image-generation-with-clip-latents/.
- Sora reinforces two constraints: generate in compressed latent space and
  improve text fidelity through highly descriptive captions / re-captioning.
  Source:
  https://openai.com/index/video-generation-models-as-world-simulators/.
- BRIDGE frames text-controlled time-series generation as a data-scarce
  problem and uses LLM-generated text-time-series pairs plus semantic
  prototypes and diffusion modeling. Source:
  https://proceedings.mlr.press/v267/li25ah.html.
- T2S aligns textual representations with time-series latent embeddings using
  VAE-style latent compression, flow matching, and a diffusion transformer.
  Source: https://www.ijcai.org/proceedings/2025/580.
- VerbalTS argues that unstructured text can express temporal nuance that
  structured labels miss, and uses multi-focal alignment/generation rather
  than reducing the text to a few labels. Source:
  https://proceedings.mlr.press/v267/gu25a.html.

## Transfer To This Project

What transfers:

- Use richer captions and multiple views of the same window.
- Keep hard negatives to force directional separation.
- Learn a project-specific mapping into the frozen generator's condition-memory
  or prefix latent space.
- Treat historical support as a manifold/audit object, not as the entire model.
- Evaluate semantic alignment and downstream scenario quality separately.

What does not transfer directly:

- We cannot copy text-to-image/video architectures because the frozen SNI
  generator recomputes prefix memory during autoregressive rollout.
- A single 128-dimensional memory vector is not a complete future rollout
  condition; it is only the final hidden summary of a recent prefix.
- Generating a full 30-day hidden prefix from text alone remains too ambitious
  without stronger oracle evidence.
- Broad CLIP-weight or support-temperature sweeps would be knob tuning unless
  tied to a precise failure mechanism.

## Recommended Direction

The next principled model family is:

```text
rich narrative captions
+ hard-negative directional captions
+ fixed start state
+ analogue support prior
-> text/start-aware latent prior over generator condition memory
-> frozen SNI rollout
```

This is DALL-E-2-like in structure, but adapted to the frozen financial
generator:

```text
text + start -> condition-memory latent prior -> frozen scenario generator
```

The first version should stay deterministic or low-variance. A distributional
prior is only justified after the deterministic target audit passes.

## Next TestFlight

Before any larger OpenAI labeling run or architecture redesign, run a local
TestFlight on the existing 182-window representative OpenAI set:

1. Build an audit table of the text examples:
   - split counts by manifest train/validation/test;
   - role counts for anchor, positive, and negative captions;
   - caption kinds;
   - number of positive captions per window;
   - hard-negative availability per window.
2. Add a text/start target diagnostic:
   - compare text-only versus text-plus-start training inputs;
   - keep the incumbent MLP memory regression and hard-negative loss;
   - use exactly the existing train/test split and cached OpenAI embeddings;
   - report target cosine, hard-negative gap/margin, and retrieval diagnostics.
3. Decide whether start conditioning improves the text-to-memory target:
   - expected improvement: same or better target cosine with no hard-negative
     degradation;
   - acceptable trade-off: target cosine can improve modestly while retrieval
     stays weak, because exact historical retrieval is not the objective;
   - falsifier: target cosine falls by more than 0.01 or hard-negative gap
     falls materially versus the incumbent.

If the TestFlight passes, then train a small text/start latent-prior variant
with the same hard-negative contract. If it fails, improve caption quality or
training coverage before adding a new architecture.

## Deliberately Out Of Scope

- No hidden model-chosen starting level.
- No future-outcome language as conditioning target.
- No all-training OpenAI relabeling until the local target diagnostic is
  positive.
- No broad hyperparameter sweep.
- No generator decoder fine-tuning until prefix-latent oracle gates show the
  decoded prefix preserves generator-relevant information.

## Decision

Iteration 121 should be a research-ideation checkpoint. The next executable
iteration should be a bounded experiment that adds the text/start target
diagnostic to the bridge bakeoff, using cached embeddings and existing
representative artifacts only.
