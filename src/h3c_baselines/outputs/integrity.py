"""Scan new MPC evidence without loading an LLM runtime contract."""

from __future__ import annotations

import os
from pathlib import Path


def secret_occurrences(output_dir: Path) -> int:
    """Count known secret values; never return or print the values themselves."""
    secrets = {
        value
        for name in ("DEEPSEEK_API_KEY", "BASETEN_API_KEY", "OPENAI_API_KEY", "BOPTEST_API_KEY")
        if (value := os.environ.get(name, ""))
    }
    return sum(
        path.read_text(encoding="utf-8", errors="ignore").count(secret)
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".txt"}
        for secret in secrets
    )
