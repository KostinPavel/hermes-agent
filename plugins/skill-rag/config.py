"""skill-rag plugin configuration.

All settings in one place. Environment variables override defaults.
"""
import os
from pathlib import Path

from hermes_constants import get_hermes_home, get_skills_dir

# --- Paths ---
# Hermes home (context override → HERMES_HOME env → platform default)
HERMES_HOME = get_hermes_home()
# Skills root directory (canonical get_skills_dir)
SKILLS_ROOT = get_skills_dir()
# Fallback folder for the index DB (if skills/ is not writable)
FALLBACK_INDEX_DIR = HERMES_HOME / "skills_index"
DB_FILENAME = ".skill_rag.db"  # SQLite index filename
FTS_TABLE = "skills_fts"  # FTS5 virtual table name (hardcoded — do not change)
SKILL_FILE = "SKILL.md"  # Skill file name for scanning

# --- Embedding provider ---
# "openai_compatible" — API (LM Studio, Ollama, vLLM, llama.cpp)
# "local" — offline via sentence-transformers (~2GB)
EMBEDDING_PROVIDER = os.environ.get("SKILL_RAG_PROVIDER", "openai_compatible")

# --- OpenAI-compatible provider ---
API_BASE = os.environ.get("SKILL_RAG_API_BASE", "http://localhost:1234/v1")  # Embedding API URL
API_KEY = os.environ.get("SKILL_RAG_API_KEY", "local-not-needed")  # API key (usually not needed)
API_MODEL = os.environ.get("SKILL_RAG_API_MODEL", "text-embedding-bge-m3")  # Embedding model name
API_BATCH_SIZE = 16  # Batch size for batch-embedding
API_TIMEOUT_CONNECT = 2.0  # Connection timeout (seconds)
API_TIMEOUT_READ = 10.0  # Read timeout (seconds)

# --- Local provider (optional) ---
LOCAL_MODEL = "intfloat/multilingual-e5-small"  # Model for offline embedding
LOCAL_REVISION = None  # Model revision (None = latest)

# --- Embedding parameters ---
EMBEDDING_DIM = 1024  # Vector dimension (384 for nomic/e5, 1024 for bge-m3)
PREFIX_QUERY = "query: "  # Prefix for queries (required by e5 models)
PREFIX_PASSAGE = "passage: "  # Prefix for passages (required by e5 models)

# --- Frontmatter fields for embedding ---
EMBED_FIELDS = (
    "name",
    "description",
    "when_to_use",
    "triggers",
    "tags",
    "category",
)
FIELD_MAX_LEN = 300  # Max field length in composed text (truncated)

# --- Retrieval ---
TOP_K = 5  # Number of recommendations
THRESHOLD = 0.3  # Min cosine similarity for vector search (0.75 too high for bge-m3)
HISTORY_WINDOW = 4  # Number of recent messages for query
ASSISTANT_TRUNCATE = 500  # Assistant response truncation in query

# --- Skill tools ---
SKILL_TOOLS = {"skill_view", "skill_manage", "skills_list"}  # Excluded from tool signal

# --- Logging ---
LOG_PREFIX = "[skill-rag]"  # Plugin log prefix
