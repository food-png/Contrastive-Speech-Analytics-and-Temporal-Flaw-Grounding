#!/usr/bin/env python3
"""Boot the Phase 6 dashboard: uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

console = Console()
console.print(
    Panel.fit(
        "[bold]Track C | Phase 6[/bold]\n"
        "Interactive speech analytics dashboard\n\n"
        "[cyan]http://localhost:8000[/cyan]",
        border_style="cyan",
        title="uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload",
    )
)

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(ROOT),
    )
