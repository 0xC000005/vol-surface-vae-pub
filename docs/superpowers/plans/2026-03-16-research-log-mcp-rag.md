# Research Log MCP + RAG Setup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up `mcp-local-rag` MCP server with a `research-log` skill so Claude Code can semantically search the 25,466-line RESEARCH_LOG.md instead of reading the full file.

**Architecture:** Add `mcp-local-rag` as a project-scoped MCP server backed by LanceDB (file-based, no server process). Create a `research-log` skill that codifies the retrieval strategy: MCP semantic search for quick lookups, MCP-as-index → Grep → Read for full-section retrieval when snippets aren't enough. Update CLAUDE.md to reference the skill. Auto-allow MCP tools in settings.

**Tech Stack:** `mcp-local-rag` (npm), LanceDB, `Xenova/all-MiniLM-L6-v2` embeddings

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `.mcp.json` | Create | MCP server configuration pointing `local-rag` at repo root (RESEARCH_LOG.md lives at root) |
| `.claude/settings.local.json` | Modify | Add `mcp__local-rag__*` tool permissions + enable MCP server |
| `.claude/skills/research-log/SKILL.md` | Create | Skill definition: append, search, targeted read workflows |
| `CLAUDE.md` | Modify | Add reference to `research-log` skill for research log access |
| `.gitignore` | Modify | Add `lancedb/` directory (vector DB artifacts shouldn't be committed) |

---

## Chunk 1: Infrastructure Setup

### Task 1: Create `.mcp.json` with `local-rag` server config

**Files:**
- Create: `.mcp.json`

- [ ] **Step 1: Create `.mcp.json`**

```json
{
  "mcpServers": {
    "local-rag": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "mcp-local-rag"],
      "env": {
        "BASE_DIR": "/home/max/Documents/vol-surface-vae-pub"
      }
    }
  }
}
```

Note: `BASE_DIR` points to repo root because `RESEARCH_LOG.md` is at the root level (not in a `research/` subdirectory like causal-sentiment).

- [ ] **Step 2: Verify file was created**

Run: `cat .mcp.json`
Expected: The JSON above, valid JSON.

- [ ] **Step 3: Commit**

```bash
git add .mcp.json
git commit -m "feat: add mcp-local-rag server config for research log semantic search"
```

---

### Task 2: Update `.claude/settings.local.json` with MCP permissions

**Files:**
- Modify: `.claude/settings.local.json`

- [ ] **Step 1: Add MCP tool permissions and server enablement**

Add the following entries to the existing `settings.local.json`:

**Insertion point 1:** Append these 4 items to the END of the `permissions.allow` array (before the closing `]`):
```
"mcp__local-rag__query_documents",
"mcp__local-rag__ingest_file",
"mcp__local-rag__status",
"mcp__local-rag__list_files"
```

**Insertion point 2:** Add these top-level keys AFTER the `"outputStyle"` key (at the end of the JSON object):
```json
"enableAllProjectMcpServers": true,
"enabledMcpjsonServers": ["local-rag"]
```

The existing `permissions`, `hooks`, and `outputStyle` fields remain unchanged.

- [ ] **Step 2: Validate JSON**

Run: `python -c "import json; json.load(open('.claude/settings.local.json'))"`
Expected: No output (valid JSON)

- [ ] **Step 3: Commit**

```bash
git add .claude/settings.local.json
git commit -m "feat: auto-allow local-rag MCP tools in settings"
```

---

### Task 3: Add `lancedb/` to `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Check current .gitignore**

Run: `cat .gitignore` (or create if missing)

- [ ] **Step 2: Append lancedb exclusion**

Add to `.gitignore`:
```
# Vector DB artifacts (mcp-local-rag)
lancedb/
# Note: models/ (embedding model cache) already covered by existing models/ gitignore entry
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: exclude lancedb vector DB from git"
```

---

### Task 4: First ingest of RESEARCH_LOG.md

This step is manual/interactive — requires the MCP server to be running.

- [ ] **Step 1: Restart Claude Code session** (so it picks up the new `.mcp.json`)

- [ ] **Step 2: Verify MCP server is running**

Call: `mcp__local-rag__status()`
Expected: Response showing the server is connected and LanceDB is initialized.

Note: First run downloads `Xenova/all-MiniLM-L6-v2` (~90MB). May take 30-60s.

- [ ] **Step 3: Ingest the research log**

Call: `mcp__local-rag__ingest_file({ filePath: "/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md" })`
Expected: Success message with chunk count. A 25,466-line file should produce ~500+ chunks.

- [ ] **Step 4: Test a semantic query**

Call: `mcp__local-rag__query_documents({ query: "what is the cross-cell correlation root cause", limit: 5 })`
Expected: Returns chunks about decoder weight problem, rank-1 noise pathway, PC1=92%.

---

## Chunk 2: Research Log Skill

### Task 5: Create the `research-log` skill

**Files:**
- Create: `.claude/skills/research-log/SKILL.md`

- [ ] **Step 1: Create skill directory**

Run: `mkdir -p .claude/skills/research-log`

- [ ] **Step 2: Write the skill file**

Create `.claude/skills/research-log/SKILL.md` with the following content:

```markdown
---
name: research-log
description: Manages the project research log — appending new entries, searching past research, and keeping the vector index current. Use this skill whenever you need to record research findings, document decisions, look up past research, or the user asks to add something to the research log. Also use when you've just completed a research task (e.g., agent-based exploration, web searches, paper reviews) and need to persist the findings. Triggers on phrases like "add to research log", "document this", "what did we find about X", "search research", "log this finding", or any time research results should be saved for future sessions.
---

# Research Log Skill

The research log lives at `RESEARCH_LOG.md` (repo root) — a chronological, append-only file indexed by `mcp-local-rag` for semantic search.

## Appending New Entries

**Never read the full file just to append.** Use Bash to append directly:

` ``bash
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
` ``

Note: The heredoc uses `RESEARCH_EOF` (without quotes) so that `$(date +%Y-%m-%d)` is evaluated by the shell at append time. This ensures the correct date is always inserted automatically.

Key rules:
- The date is automatically inserted via `$(date +%Y-%m-%d)` — do NOT hardcode it
- End every entry with `---` on its own line (separates entries visually)
- Keep entries concise — summaries and key takeaways, not full dumps
- Use markdown tables for comparisons, bullet points for lists
- Use the project's existing header format: `## YYYY-MM-DD: Title` (colon, not pipe)

## Re-ingesting After Changes

After appending, always re-ingest so the new content is searchable:

` ``
mcp__local-rag__ingest_file({ filePath: "/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md" })
` ``

Re-ingesting updates existing chunks and adds new ones. It's idempotent and fast.

## Retrieving Past Research: Two Modes

Choose the retrieval method based on what you need. Getting this right avoids either wasting tokens (reading everything when you need one fact) or missing context (getting fragments when you need the full picture).

### Quick Lookup (use MCP semantic search)

**When:** You need a specific fact, a single decision, a parameter value, or a quick answer.

` ``
mcp__local-rag__query_documents({ query: "your question here", limit: 5 })
` ``

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
` ``
mcp__local-rag__query_documents({ query: "experiment 99k results", limit: 3 })
` ``
This returns chunks — note which part of the file the results point to.

**Step 2:** Use Grep to find the exact line number of the section header:
` ``
Grep pattern="## 2026-03-09: Exp 99k" path="/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md"
` ``

**Step 3:** Read that section with surrounding context:
` ``
Read file_path="/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md" offset=<line> limit=80
` ``

This gives you the full section — tables intact, all rows visible, context + findings + decision together — without loading the entire 25,000+ line file.

### Multi-Section Read (for comprehensive retrieval across multiple entries)

**When:** You need to synthesize across multiple research entries — e.g., comparing results across the full 99-series experiments.

**Step 1:** MCP search with higher limit to identify all relevant sections:
` ``
mcp__local-rag__query_documents({ query: "your broad topic", limit: 15 })
` ``

**Step 2:** Note the distinct section headers in the results.

**Step 3:** Grep for all section headers to get line numbers (table of contents):
` ``
Grep pattern="^## 2026-" path="/home/max/Documents/vol-surface-vae-pub/RESEARCH_LOG.md" output_mode="content"
` ``

**Step 4:** Read each relevant section individually with offset/limit. Typical section is 40-150 lines.

### Decision Flowchart

` ``
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
` ``

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
```

Note: The triple backticks inside the skill file need to be real backticks (the escaped versions above are just to avoid breaking this plan's markdown). When writing the actual file, use proper triple backticks for all code blocks.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/research-log/SKILL.md
git commit -m "feat: add research-log skill for MCP semantic search + targeted retrieval"
```

---

### Task 6: Update CLAUDE.md to reference the skill

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read the current CLAUDE.md**

Identify where to add the reference. The existing `## Development Environment` section or a new section after the project overview.

- [ ] **Step 2: Add skill reference**

Find the line `- **Research log**: RESEARCH_LOG.md` or similar reference. Add/modify to include:

```markdown
## Research Log

The research log is in `RESEARCH_LOG.md` (25,000+ lines). **Never read the full file.** Use the `research-log` skill (MCP semantic search or targeted Read) to retrieve past findings. The skill is invoked automatically when you need to search or append to the log.
```

This should go right after the `## Project Overview` section or replace any existing research log reference.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add research-log skill reference to CLAUDE.md"
```

---

## Verification

After all tasks are complete, verify the full setup works end-to-end:

1. **MCP server responds:** `mcp__local-rag__status()` returns connected
2. **Semantic search works:** `mcp__local-rag__query_documents({ query: "butterfly arbitrage rate", limit: 3 })` returns relevant chunks about the 24% butterfly arbitrage issue
3. **Targeted read works:** Use MCP result to find a section header → Grep for line number → Read with offset/limit → get full table intact
4. **Append + re-ingest works:** Append a test entry, re-ingest, query for it, then remove the test entry
5. **Skill is discoverable:** In a new Claude Code session, the `research-log` skill should appear in the available skills list
