# Model-tiered agents & workflow routing convention

Token-saving policy for this repo (from the 2026-06-11 Claude Code cost research). The main
session / Workflow orchestrator stays on **Opus 4.8**; high-volume sub-work routes to cheaper
models via subagents. A subagent beats a mid-loop `/model` swap because the prompt cache is
model-specific — swapping mid-conversation forces a full cache miss, whereas a subagent runs in
its own context and returns only a summary, keeping the Opus prefix warm.

## Tiers

| Tier | Model | $/1M (in/out) | Use for |
|------|-------|---------------|---------|
| Orchestrator | Opus 4.8 | 5 / 25 | planning, cross-subsystem synthesis, final judgement (main session — do not pin) |
| `analyst` | Sonnet 4.6 | 3 / 15 | per-subsystem review/verification, codegen, claim↔evidence checks |
| `reader` | Haiku 4.5 | 1 / 5 | file/result inventory, grep/extract, pull numbers from JSON/logs |

Do NOT orchestrate on Fable 5 (10/50 = 2× Opus) unless planning quality is genuinely the bottleneck.

## Two mechanisms

1. **Agent tool / `subagent_type`:** these `.claude/agents/*.md` files pin `model:` in frontmatter
   (`reader`→haiku, `analyst`→sonnet). Dispatch with the Agent tool using `subagent_type: reader|analyst`.
2. **Workflow `agent()` calls:** pin per stage with the `model` opt —
   `agent(prompt, {model:'haiku'})` for search/extract/verify stages,
   `agent(prompt, {model:'sonnet'})` for analysis/review/codegen,
   omit `model` (inherit Opus) only for the final cross-subsystem synthesis stage.
   Optionally combine with `agentType:'reader'|'analyst'` to also get the system prompt.

Canonical workflow shape: **haiku inventory/extraction → sonnet per-subsystem verification → opus synthesis.**
This cuts an all-Opus fan-out (e.g. the 34-agent/~2M-token research run) ~3–5× at near-zero quality loss.
