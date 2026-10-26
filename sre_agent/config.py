from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SRE_", extra="ignore")

    llm_model: str = "ollama:qwen3.6"
    llm_temperature: float = 0.0
    # Ollama only: its default 4k context silently truncates the larger prompts,
    # and "thinking" mode is slow without helping structured answers much.
    ollama_num_ctx: int = 16384
    ollama_reasoning: bool = False

    persistence: Literal["sqlite", "postgres", "memory"] = "sqlite"
    sqlite_path: Path = PROJECT_ROOT / ".data" / "sre_agent.db"
    postgres_url: str = "postgresql://sre:sre@localhost:5442/sre_agent"

    scenarios_dir: Path = PROJECT_ROOT / "scenarios"

    confidence_threshold: float = 0.75
    max_investigation_rounds: int = 3
    max_tool_steps: int = 4
    max_remediation_attempts: int = 2
    tool_call_budget: int = 40


def get_settings() -> Settings:
    return Settings()
