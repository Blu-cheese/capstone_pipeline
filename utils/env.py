"""
Minimal .env loader.

The pipeline reads credentials from os.environ. Rather than requiring every
shell to export them by hand (the footgun that made OPENAI_BASE_URL unset in
practice), load a gitignored .env at the repo root on import.

Deliberately dependency-free - no python-dotenv - to keep the "no pip
packages required for mock mode" property from the README.

Real environment variables always win, so `OPENAI_API_KEY=... python
pipeline.py` still overrides the file.
"""

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path = ENV_PATH, override: bool = False) -> dict:
    """
    Load KEY=VALUE pairs from a .env file into os.environ.

    Returns the keys that were applied. Missing file is not an error - the
    variables may legitimately come from the shell instead.
    """
    if not path.exists():
        return {}

    applied = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue

        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value

    return applied


def mask(secret: str, keep: int = 4) -> str:
    """Render a credential safe to print in logs."""
    if not secret:
        return "(unset)"
    if len(secret) <= keep * 2:
        return "*" * len(secret)
    return f"{secret[:keep]}...{secret[-keep:]} ({len(secret)} chars)"
