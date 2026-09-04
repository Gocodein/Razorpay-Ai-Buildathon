"""
Central configuration — all settings loaded from .env once.
Import `settings` anywhere in the project; never use os.getenv directly.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # ── Razorpay Credentials ─────────────────────────────────────────────────
    razorpay_key_id: str = field(
        default_factory=lambda: os.getenv("RAZORPAY_KEY_ID", "rzp_test_demo")
    )
    razorpay_key_secret: str = field(
        default_factory=lambda: os.getenv("RAZORPAY_KEY_SECRET", "demo_secret")
    )

    # ── Merchant Identity & UPI Rails ─────────────────────────────────────────
    merchant_id: str = field(
        default_factory=lambda: os.getenv("MERCHANT_ID", "demo_merchant_001")
    )
    merchant_name: str = field(
        default_factory=lambda: os.getenv("MERCHANT_NAME", "GreenLeaf Organics")
    )
    merchant_upi_vpa: str = field(
        default_factory=lambda: os.getenv("MERCHANT_UPI_VPA", "rzp.greenleaf@hdfcbank")
    )

    # ── Financial Governance (NPCI UAP spending cap per session in ₹) ────────
    agent_spending_limit_inr: int = field(
        default_factory=lambda: int(os.getenv("AGENT_SPENDING_LIMIT_INR", "2000"))
    )

    # ── LLM Provider: "gemini" | "groq" | "claude" | "openai" | "offline" ─────
    llm_provider: str = field(
        default_factory=lambda: os.getenv(
            "LLM_PROVIDER",
            "gemini" if os.getenv("GEMINI_API_KEY")
            else ("groq" if os.getenv("GROQ_API_KEY")
            else ("claude" if os.getenv("ANTHROPIC_API_KEY")
            else "offline"))
        )
    )

    # ── API Keys ─────────────────────────────────────────────────────────────
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )

    # ── Model Identifiers ────────────────────────────────────────────────────
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    )
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    claude_model: str = field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    # ── Local Storage Paths ──────────────────────────────────────────────────
    chroma_db_path: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_DB_PATH", str(BASE_DIR / "data" / "catalog_db")
        )
    )
    audit_log_path: str = field(
        default_factory=lambda: os.getenv(
            "AUDIT_LOG_PATH", str(BASE_DIR / "data" / "audit.db")
        )
    )

    # ── Vector DB & Embeddings ───────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    catalog_collection: str = "merchant_catalog"


settings = Settings()


def reload_settings() -> Settings:
    """Reload .env file and update the global settings instance."""
    global settings
    load_dotenv(override=True)
    settings = Settings()
    return settings


def save_env_key(key: str, value: str) -> None:
    """Safely update or add an environment variable in the local .env file."""
    env_path = BASE_DIR / ".env"
    lines = []
    found = False
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}=") or line.strip().startswith(f"{key} ="):
            new_lines.append(f"{key}={value.strip()}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        new_lines.append(f"{key}={value.strip()}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    os.environ[key] = value.strip()
    reload_settings()
