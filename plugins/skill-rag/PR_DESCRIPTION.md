# skill-rag: Semantic Skill Retrieval Plugin

## Overview

Add a new plugin that automatically recommends relevant skills from the user's `~/.hermes/skills/` library based on conversation context. The plugin uses vector embeddings (OpenAI-compatible API) with BM25/FTS5 fallback, injecting up to 5 skill recommendations into each LLM call via the `pre_llm_call` hook.

## Motivation

Currently, the agent must manually call `skills_list` and `skill_view` to discover relevant skills, adding latency and token overhead. This plugin proactively surfaces relevant skills before each LLM call, reducing the number of round-trips needed to load the right skill for a task.

## Key Features

- **Semantic vector search** — Embeds skill metadata (name, description, triggers, tags) into 1024-dim vectors and performs cosine similarity search against the query
- **BM25/FTS5 fallback** — When embedding API is unavailable or results are stale, falls back to SQLite FTS5 full-text search
- **Hermes-native visibility filtering** — Uses the same filtering logic as the system prompt builder (`get_disabled_skill_names()`, `skill_matches_*()`, `_skill_should_show()`) to index only visible skills
- **Dependency injection** — Uses `Protocol`-based abstractions (`Embedder`, `SkillIndex`) for testability and swappable backends
- **Thread safety** — Double-checked locking for plugin init, `threading.Lock` for SQLite connections, thread-local `requests.Session` for HTTP pooling
- **Incremental indexing** — Content-hash tracking (`SHA-256`) detects changed skills and re-embeds only what's needed; model-hash invalidation triggers full re-embedding on model change
- **Stale detection** — Checks freshness of retrieved skills before injection; re-indexes stale skills and retries search
- **Context-aware deduplication** — Scans conversation history for skills already loaded via `skill_view` or present in `<available_skills>` blocks, excluding them from recommendations
- **Multi-provider support** — Works with LM Studio, Ollama, vLLM, llama.cpp (OpenAI-compatible API) or offline via `sentence-transformers`

## Architecture

```
skill-rag/
├── plugin.yaml      # Plugin manifest: hooks (pre_llm_call, on_skill_lifecycle), metadata
├── config.py        # All configuration: paths, embedding params, retrieval thresholds
├── indexer.py       # SQLite + FTS5 index, embedding (local/API), scanning, upsert
├── retrieval.py     # Query building, vector search, BM25 fallback, context assembly
├── __init__.py      # Plugin entry point, DI container, hook implementations
└── requirements.txt # Runtime dependencies
```

### Data Flow

1. **Hook fires** → `on_pre_llm_call` receives `user_message` and `conversation_history`
2. **Query building** → `build_query()` extracts intent from user message, strips injected blocks (memory, attachments, tool outputs), includes final assistant text
3. **Visibility filtering** → `scan_skills()` uses `_is_skill_visible()` to filter skills by disabled/platform/toolset conditions
4. **Vector search** → `retrieve()` embeds query with `query: ` prefix, computes cosine similarity against indexed skills, filters by threshold (0.3)
5. **Stale check** → `check_fresh()` compares content hashes; stale skills trigger `reindex()` then retry
6. **BM25 fallback** → If vector results still stale after reindex, `retrieve_bm25()` uses FTS5 with word-level matching and 2-word minimum
7. **Context injection** → `build_context()` formats results in `<available_skills>` block matching Hermes system prompt format

### Configuration

All settings are configurable via environment variables or `~/.hermes/config.yaml`:

| Variable | Default | Description |
|---|---|---|
| `SKILL_RAG_PROVIDER` | `openai_compatible` | Embedding provider (`openai_compatible` or `local`) |
| `SKILL_RAG_API_BASE` | `http://localhost:11434/v1` | Embedding API endpoint |
| `SKILL_RAG_API_KEY` | `local-not-needed` | API key for embedding endpoint |
| `SKILL_RAG_API_MODEL` | `nomic-embed-text` | Model name for embeddings |

Additional tunables: `TOP_K` (5), `THRESHOLD` (0.3), `HISTORY_WINDOW` (4), `ASSISTANT_TRUNCATE` (500), `FIELD_MAX_LEN` (300), `EMBEDDING_DIM` (1024).

### System Prompt Opt-Out (Generic)

To avoid duplicating skills (system prompt + plugin injection), set in `~/.hermes/config.yaml`:

```yaml
skills:
  prompt_index: false
```

This makes the system prompt skip the static `<available_skills>` block. **This flag is generic** — it works regardless of which plugin manages skill discovery. It follows the project's plugin policy (`plugins/AGENTS.md`: never hardcode plugin-specific logic in core).

## Test Coverage

79 unit tests across 5 test classes, covering:

| Module | Tests | Coverage |
|---|---|---|
| `retrieval.py` (context extraction, query building, vector/BM25 search) | 24 | Message parsing, query construction, dedup, vector search, BM25 fallback, context assembly |
| `indexer.py` (schema, scanning, upsert, embedding) | 18 | Frontmatter parsing, path IDs, content hashes, compose text, schema init, skill scanning, upsert with/without vectors |
| `__init__.py` (plugin hooks, registration) | 6 | Hook registration, empty input handling, lifecycle reindexing, plugin.yaml validation |
| `config.py` (defaults, constants) | 2 | Default provider, constant existence |
| Legacy injector tests (compatibility) | 29 | Marker formatting, context extraction, context building |

```bash
# Run all tests (no API required)
cd ~/.hermes/plugins/skill-rag
python -m pytest tests/test_skill_rag.py -v
```

## Security Measures

| ID | Severity | Mitigation |
|---|---|---|
| **H1** | High | FTS5 query words stripped of `"` characters to prevent FTS5 query injection |
| **H3** | High | All SQL queries use parameterized statements (no f-string interpolation of user data) |
| **M1** | Medium | File size validation (`_MAX_SKILL_FILE_SIZE = 1MB`) before reading SKILL.md files |
| **M3** | Medium | Oversized skill files are skipped to prevent memory exhaustion during scanning |
| **M6** | Medium | FTS5 table name hardcoded in SQL — never interpolated from user input |

Additional security measures:
- SQLite WAL journal mode for concurrent read safety
- `check_same_thread=False` with explicit `threading.Lock` for connection management
- Graceful degradation: embedding API failures return empty results (no crash, no partial state)
- `atexit` handler ensures database connection cleanup on process shutdown

## Performance

- **Query latency**: ~0.038s/query (vector search against 50+ indexed skills)
- **Startup**: One-time cost for scanning and embedding; subsequent calls use cached index
- **Incremental indexing**: Only changed skills are re-embedded (content-hash comparison)
- **Batch embedding**: Skills embedded in batches of 16 to reduce API round-trips
- **Thread-local HTTP**: `requests.Session` with connection pooling (4 connections, 8 max) per thread

## Breaking Changes

None. This is a new plugin that hooks into the `pre_llm_call` lifecycle. Existing behavior is unchanged when the plugin is not installed.

## Dependencies

### Required
- `requests` — HTTP client for OpenAI-compatible embedding API
- `numpy` — Vector operations for cosine similarity

### Optional
- `sentence-transformers` — For offline embedding via `SKILL_RAG_PROVIDER=local`
- `pyyaml` — For config.yaml parsing (already present in Hermes)

## Installation

```bash
cp -r skill-rag/ ~/.hermes/plugins/
pip install requests numpy
```

The plugin activates automatically on the next Hermes session. No manual configuration required — defaults work with Ollama (`nomic-embed-text` model).

## Checklist

- [x] Plugin follows Hermes plugin conventions (plugin.yaml, register(), hooks)
- [x] No `entry_point` in plugin.yaml (uses hook-based registration)
- [x] All SQL is parameterized (no injection vectors)
- [x] Thread-safe initialization and connection management
- [x] Graceful degradation when embedding API is unavailable
- [x] Tests pass without external API dependencies
- [x] README with installation, configuration, troubleshooting
- [x] No breaking changes to existing Hermes behavior
- [x] Home resolution via `hermes_constants.get_hermes_home()` (canonical Hermes pattern)
- [x] System prompt opt-out via generic `skills.prompt_index: false` config flag (no plugin-specific logic in core)
- [x] Complies with `plugins/AGENTS.md` policy (plugins never touch core)
