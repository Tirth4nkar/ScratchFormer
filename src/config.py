"""
config.py — Project-wide paths and settings.
All modules should import from here rather than hard-coding paths.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parents[2]
DATA_DIR       = ROOT_DIR / os.getenv("DATA_DIR", "data")
RAW_DIR        = DATA_DIR / "raw"
INTERIM_DIR    = DATA_DIR / "interim"
PROCESSED_DIR  = DATA_DIR / "processed"
EXTERNAL_DIR   = DATA_DIR / "external"
MODELS_DIR     = ROOT_DIR / os.getenv("MODELS_DIR", "models")
REPORTS_DIR    = ROOT_DIR / os.getenv("REPORTS_DIR", "reports")
FIGURES_DIR    = REPORTS_DIR / "figures"

# ── LLM API keys ──────────────────────────────────────────────
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── LangSmith tracing ─────────────────────────────────────────
LANGCHAIN_API_KEY      = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT      = os.getenv("LANGCHAIN_PROJECT", "my_project")
LANGCHAIN_TRACING_V2   = os.getenv("LANGCHAIN_TRACING_V2", "false")
