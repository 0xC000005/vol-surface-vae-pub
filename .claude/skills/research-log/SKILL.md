---
name: research-log
description: Manages the project research log — appending new entries, searching past research, and keeping the vector index current. Use this skill whenever you need to record research findings, document decisions, look up past research, or the user asks to add something to the research log. Also use when you've just completed a research task (e.g., agent-based exploration, web searches, paper reviews) and need to persist the findings. Triggers on phrases like "add to research log", "document this", "what did we find about X", "search research", "log this finding", or any time research results should be saved for future sessions.
---

# Research Log Skill

The research log lives at `RESEARCH_LOG.md` (repo root) — a chronological, append-only file indexed by `mcp-local-rag` for semantic search.

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

## Re-ingesting After Changes

After appending, always re-ingest so the new content is searchable:

```
mcp__local-rag__ingest_file({ filePath: "/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md" })
```

Re-ingesting updates existing chunks and adds new ones. It's idempotent and fast.

## Retrieving Past Research: Two Modes

Choose the retrieval method based on what you need. Getting this right avoids either wasting tokens (reading everything when you need one fact) or missing context (getting fragments when you need the full picture).

### Quick Lookup (use MCP semantic search)

**When:** You need a specific fact, a single decision, a parameter value, or a quick answer.

```
mcp__local-rag__query_documents({ query: "your question here", limit: 5 })
```

**Works well for:**
- "What CI coverage did experiment 99k achieve?" → returns the exact value
- "What is the kurtosis ratio target?" → returns the threshold
- "Why did we switch from ratio loss to full ES?" → returns the decision
- Single facts, parameter values, specific decisions

**Limitations to be aware of:**
- Markdown tables get fragmented into individual rows — you'll get row 3 but miss rows 1, 2, 4, 5
- Results are 1-2 sentence snippets with no surrounding context (no section headers, no motivation)
- Semantic search can miss tabular data unless you include exact keywords from the table cells
- If you need the full picture of a topic, quick lookup will give you disconnected fragments

**Tip:** Include specific keywords from the content, not just abstract descriptions. "99k full ES corr decorrelation" works better than "which experiment fixed correlation."

### Targeted Section Read (MCP search as index → Read for content)

**When:** You need comprehensive understanding of a topic — comparing experiment results in a table, understanding full reasoning behind an architecture decision, or any task where MCP search snippets are too fragmented.

**Step 1:** MCP search to find the location:
```
mcp__local-rag__query_documents({ query: "experiment 99k results", limit: 3 })
```
This returns chunks — note which part of the file the results point to.

**Step 2:** Use Grep to find the exact line number of the section header:
```
Grep pattern="## 2026-03-09: Exp 99k" path="/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md"
```

**Step 3:** Read that section with surrounding context:
```
Read file_path="/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md" offset=<line> limit=80
```

This gives you the full section — tables intact, all rows visible, context + findings + decision together — without loading the entire 25,000+ line file.

### Multi-Section Read (for comprehensive retrieval across multiple entries)

**When:** You need to synthesize across multiple research entries — e.g., comparing results across the full 99-series experiments.

**Step 1:** MCP search with higher limit to identify all relevant sections:
```
mcp__local-rag__query_documents({ query: "your broad topic", limit: 15 })
```

**Step 2:** Note the distinct section headers in the results.

**Step 3:** Grep for all section headers to get line numbers (table of contents):
```
Grep pattern="^## 2026-" path="/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md" output_mode="content"
```

**Step 4:** Read each relevant section individually with offset/limit. Typical section is 40-150 lines.

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
MCP search    Is it within a single section (one dated entry)?
(limit: 5)      |           |
               YES          NO (spans multiple entries)
                |           |
                v           v
        Targeted Section   Multi-Section Read
        Read (MCP→Grep     (MCP search limit:15 →
         →Read offset)      Grep headers → Read
                            each section)
```

### Token Budget

| Action | Tool | Token cost | When to use |
|--------|------|-----------|-------------|
| Append new entry | Bash (cat >>) | ~0 | Always — never read to append |
| Quick fact lookup | query_documents (limit 5) | ~300-500 tokens | Single facts, decisions, values |
| Broad search | query_documents (limit 15-20) | ~1,500-2,500 tokens | Multiple related facts |
| Targeted section | MCP search → Grep → Read offset/limit | ~2,000-4,000 tokens | One section with full context |
| Multi-section | MCP search → Grep headers → multiple Reads | ~4,000-8,000 tokens | Synthesizing across entries |
| Edit existing entry | Read + Edit | Section only via offset/limit | Correcting past entries (rare) |
| Re-index after append | ingest_file | ~0 | Always after appending |

### Scaling Notes

This approach works at any file size because you never read the entire file:
- MCP search finds WHAT is relevant (semantic match on content)
- Grep finds WHERE it is (exact line number)
- Read with offset/limit gets the FULL SECTION (tables, context, reasoning intact)
- No splitting into multiple files needed — one chronological log, any length
