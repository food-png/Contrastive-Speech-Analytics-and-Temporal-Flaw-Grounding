"""Load project-root ``.env`` into ``os.environ`` (safe to call multiple times)."""

from __future__ import annotations

from pathlib import Path

_loaded = False


def load_project_env() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True

    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.is_file():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(env_path, override=False)
