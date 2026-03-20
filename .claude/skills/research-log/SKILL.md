---
name: research-log
description: Manages the project research log — appending new entries, searching past research, and keeping the vector index current. Use this skill whenever you need to record research findings, document decisions, look up past research, or the user asks to add something to the research log. Also use when you've just completed a research task (e.g., agent-based exploration, web searches, paper reviews) and need to persist the findings. Triggers on phrases like "add to research log", "document this", "what did we find about X", "search research", "log this finding", or any time research results should be saved for future sessions.
---

# Research Log Skill

The research log lives at `RESEARCH_LOG.md` (repo root) — a chronological, append-only file indexed by QMD for hybrid search (BM25 + vector + LLM reranking). QMD indexes all markdown files in the project as the "research" collection.

## Appending New Entries

**Never read the full file just to append.** Use Bash to append directly:

```bash
cat >> /home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md << RESEARCH_EOF

## $(date +%Y-%m-%d): Short Title Describing the Topic

### Context
Why this research was done.

### Key Findings
What was discovered. Use tables, bullet points, code blocks as needed.

### Decision
What was decided based on the findings. What's next.

---
RESEARCH_EOF
```

Note: The heredoc uses `RESEARCH_EOF` (without quotes) so that `$(date +%Y-%m-%d)` is evaluated by the shell at append time. This ensures the correct date is always inserted automatically.

Key rules:
- The date is automatically inserted via `$(date +%Y-%m-%d)` — do NOT hardcode it
- End every entry with `---` on its own line (separates entries visually)
- Keep entries concise — summaries and key takeaways, not full dumps
- Use markdown tables for comparisons, bullet points for lists
- Use the project's existing header format: `## YYYY-MM-DD: Title` (colon, not pipe)

## Re-indexing After Changes

After appending, update the QMD index so the new content is searchable:

```bash
Bash command="qmd update --collection research && qmd embed" timeout=120000
```

QMD's update is incremental — it detects which files changed and only re-embeds those, skipping unchanged content. Much faster than a full re-index.

## Retrieving Past Research: Two Modes

Choose the retrieval method based on what you need. Getting this right avoids either wasting tokens (reading everything when you need one fact) or missing context (getting fragments when you need the full picture).

### Quick Lookup (use QMD semantic search)

**When:** You need a specific fact, a single decision, a parameter value, or a quick answer.

```
mcp__qmd__query({
  searches: [
    { type: "lex", query: "your exact keywords here" },
    { type: "vec", query: "your natural language question here" }
  ],
  intent: "what you're trying to find",
  collection: "research",
  limit: 5
})
```

**QMD search tips:**
- Use `lex` for exact terms (experiment IDs, metric names, specific phrases)
- Use `vec` for conceptual/semantic queries (understanding why something happened)
- Combine both for best results — lex finds exact matches, vec finds related content
- The `intent` field helps disambiguate (e.g., query="performance", intent="model training speed")
- Avoid hyphens in `vec` queries — they get parsed as negation. Write "cross cell" not "cross-cell"
- Use `lex` with quoted phrases for exact matches: `"Exp 120b"`, `"rank collapse"`

**Works well for:**
- "What CI coverage did experiment 99k achieve?" → `lex: "99k" CI coverage`
- "What is the kurtosis ratio target?" → `vec: what is the target range for kurtosis ratio`
- "Why did we switch from ratio loss to full ES?" → `vec: why switch from ratio loss to energy score`
- Single facts, parameter values, specific decisions

**QMD returns richer results than raw RAG:** each result includes the file path, line number, score (0-1, higher is better), and a context snippet with surrounding lines. This often gives enough information without needing a follow-up Read.

**Project-specific note:** Experiments use IDs like `99k`, `99j_v3`, `97a`, `100a`. For known experiment IDs, skip QMD and use the Experiment ID Shortcut below — Grep is faster and more precise for exact ID lookups.

### Experiment ID Shortcut (skip QMD, use Grep directly)

**When:** You know the experiment ID (e.g., 99k, 99j_v3, 97a, 100a).

Most experiments are `###` subsections, not `##` top-level entries. Go straight to Grep:
```
Grep pattern="### Exp 99k" path="/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md" output_mode="content"
```
Then Read at that offset. This is faster than QMD search for known experiment IDs.

### Targeted Section Read (QMD search as index → Read for content)

**When:** You need comprehensive understanding of a topic — comparing experiment results in a table, understanding full reasoning behind an architecture decision, or any task where QMD search snippets are too fragmented.

**Step 1:** QMD search to find the location:
```
mcp__qmd__query({
  searches: [
    { type: "lex", query: "experiment 99k" },
    { type: "vec", query: "experiment 99k results and findings" }
  ],
  intent: "find the full results section for experiment 99k",
  collection: "research",
  limit: 3
})
```
This returns results with file paths and line numbers — note the line number from the snippet.

**Step 2:** Use Grep to find the exact line number of the section header. Note: most experiments are `###` subsections, not `##` top-level entries:
```
Grep pattern="### Exp 99k" path="/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md" output_mode="content"
```
If unsure of the heading level, search both: `Grep pattern="#{2,3} .*99k"`

**Step 3:** Read that section with surrounding context:
```
Read file_path="/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md" offset=<line> limit=80
```

This gives you the full section — tables intact, all rows visible, context + findings + decision together — without loading the entire 25,000+ line file.

### Multi-Section Read (for comprehensive retrieval across multiple entries)

**When:** You need to synthesize across multiple research entries — e.g., comparing results across the full 99-series experiments.

**Step 1:** QMD search with higher limit to identify all relevant sections:
```
mcp__qmd__query({
  searches: [
    { type: "lex", query: "your keywords" },
    { type: "vec", query: "your broad topic description" }
  ],
  intent: "find all entries related to this topic",
  collection: "research",
  limit: 15
})
```

**Step 2:** Note the distinct section headers and line numbers in the results.

**Step 3:** Grep for section headers to get line numbers (table of contents). Use `##` for dated entries, `###` for individual experiments:
```
Grep pattern="^## 2026-" path="/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md" output_mode="content"
```
For experiment-level TOC:
```
Grep pattern="^### Exp 99" path="/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md" output_mode="content"
```

**Step 4:** Read each relevant section individually with offset/limit. Typical section is 40-150 lines.

### Retrieving from Other Project Files

QMD indexes all markdown files in the project, not just RESEARCH_LOG.md. For content in investigation reports, kurtosis analysis, or other markdown files, use `mcp__qmd__get` to retrieve by file path:

```
mcp__qmd__get({ file: "research/results/investigations/120b-deep/synthesis.md" })
```

You can also slice by line: `mcp__qmd__get({ file: "path/to/file.md", fromLine: 50, maxLines: 100 })`

### Decision Flowchart

```
Need research log content?
    |
    v
Is it a single fact or quick answer?
    |           |
   YES          NO
    |           |
    v           v
QMD search    Is it within a single section (one dated entry)?
(limit: 5)      |           |
               YES          NO (spans multiple entries)
                |           |
                v           v
        Targeted Section   Multi-Section Read
        Read (QMD→Grep     (QMD search limit:15 →
         →Read offset)      Grep headers → Read
                            each section)
```

### Token Budget

| Action | Tool | Token cost | When to use |
|--------|------|-----------|-------------|
| Append new entry | Bash (cat >>) | ~0 | Always — never read to append |
| Quick fact lookup | mcp__qmd__query (limit 5) | ~300-500 tokens | Single facts, decisions, values |
| Broad search | mcp__qmd__query (limit 15-20) | ~1,500-2,500 tokens | Multiple related facts |
| Targeted section | QMD search → Grep → Read offset/limit | ~2,000-4,000 tokens | One section with full context |
| Multi-section | QMD search → Grep headers → multiple Reads | ~4,000-8,000 tokens | Synthesizing across entries |
| Edit existing entry | Read + Edit | Section only via offset/limit | Correcting past entries (rare) |
| Re-index after append | Bash: qmd update && qmd embed | ~0 (runs in shell) | Always after appending |
| Retrieve other file | mcp__qmd__get | ~500-2,000 tokens | Investigation reports, analyses |

### Scaling Notes

This approach works at any file size because you never read the entire file:
- QMD search finds WHAT is relevant (hybrid BM25 + semantic match on content)
- Grep finds WHERE it is (exact line number)
- Read with offset/limit gets the FULL SECTION (tables, context, reasoning intact)
- QMD also indexes all other markdown files in the project (investigations, analyses, etc.)
- No splitting into multiple files needed — one chronological log, any length
