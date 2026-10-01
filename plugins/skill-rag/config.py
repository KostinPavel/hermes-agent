"""Конфигурация плагина skill-rag.

Все настройки в одном месте. Переменные окружения переопределяют значения по умолчанию.
"""
import os
from pathlib import Path

from hermes_constants import get_hermes_home, get_skills_dir

# --- Пути ---
# Hermes home (context override → HERMES_HOME env → platform default)
HERMES_HOME = get_hermes_home()
# Корневая директория скиллов Hermes (канонический get_skills_dir)
SKILLS_ROOT = get_skills_dir()
# Резервная папка для БД (если skills/ недоступна для записи)
FALLBACK_INDEX_DIR = HERMES_HOME / "skills_index"
DB_FILENAME = ".skill_rag.db"  # Имя SQLite-файла индекса
FTS_TABLE = "skills_fts"  # Имя FTS5 виртуальной таблицы (хардкод — не менять)
SKILL_FILE = "SKILL.md"  # Имя файла скилла для сканирования

# --- Провайдер эмбеддингов ---
# "openai_compatible" — API (LM Studio, Ollama, vLLM, llama.cpp)
# "local" — offline через sentence-transformers (~2GB)
EMBEDDING_PROVIDER = os.environ.get("SKILL_RAG_PROVIDER", "openai_compatible")

# --- OpenAI-совместимый провайдер ---
API_BASE = os.environ.get("SKILL_RAG_API_BASE", "http://localhost:1234/v1")  # URL embedding API
API_KEY = os.environ.get("SKILL_RAG_API_KEY", "local-not-needed")  # Ключ API (обычно не нужен)
API_MODEL = os.environ.get("SKILL_RAG_API_MODEL", "text-embedding-bge-m3")  # Имя модели для эмбеддинга
API_BATCH_SIZE = 16  # Размер батча для batch-embedding
API_TIMEOUT_CONNECT = 2.0  # Таймаут подключения (сек)
API_TIMEOUT_READ = 10.0  # Таймаут чтения (сек)

# --- Локальный провайдер (опционально) ---
LOCAL_MODEL = "intfloat/multilingual-e5-small"  # Модель для offline-эмбеддинга
LOCAL_REVISION = None  # Ревизия модели (None = latest)

# --- Параметры эмбеддинга ---
EMBEDDING_DIM = 1024  # Размерность вектора (384 для nomic/e5, 1024 для bge-m3)
PREFIX_QUERY = "query: "  # Префикс для запросов (e5-модели требуют)
PREFIX_PASSAGE = "passage: "  # Префикс для passages (e5-модели требуют)

# --- Поля frontmatter для эмбеддинга ---
EMBED_FIELDS = (
    "name",
    "description",
    "when_to_use",
    "triggers",
    "tags",
    "category",
)
FIELD_MAX_LEN = 300  # Макс. длина поля в compose-тексте (обрезка)

# --- Retrieval ---
TOP_K = 5  # Количество рекомендаций
THRESHOLD = 0.3  # Мин. cosine similarity для vector search (0.75 слишком высок для bge-m3)
HISTORY_WINDOW = 4  # Количество последних сообщений для query
ASSISTANT_TRUNCATE = 500  # Обрезка ответов ассистента в query

# --- Инструменты скиллов ---
SKILL_TOOLS = {"skill_view", "skill_manage", "skills_list"}  # Исключаются из tool-сигнала

# --- Логи ---
LOG_PREFIX = "[skill-rag]"  # Префикс для логов плагина
