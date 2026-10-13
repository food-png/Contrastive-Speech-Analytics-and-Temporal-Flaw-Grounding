#!/usr/bin/env python3
"""Install pytest if needed, run the Track C suite, and print a rich summary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "tests" / "test_automated_suite.py"


def _ensure_pytest() -> None:
    try:
        import pytest  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest>=8.0.0"])


class _ResultCollector:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, float]] = []

    def pytest_runtest_logreport(self, report) -> None:  # noqa: ANN001
        if report.when == "call":
            self.rows.append((report.nodeid, report.outcome, float(report.duration)))
        elif report.when == "setup" and report.failed:
            self.rows.append((report.nodeid, "failed", 0.0))


def main() -> None:
    _ensure_pytest()
    import pytest
    from rich.console import Console
    from rich.table import Table

    console = Console()
    collector = _ResultCollector()
    exit_code = pytest.main(
        ["-v", "--tb=short", str(SUITE)],
        plugins=[collector],
    )

    table = Table(title="Track C automated QA summary")
    table.add_column("Test")
    table.add_column("Result")
    table.add_column("Seconds", justify="right")
    passed = failed = skipped = 0
    for nodeid, outcome, duration in collector.rows:
        name = nodeid.split("::")[-1]
        style = {"passed": "green", "failed": "red", "skipped": "yellow"}.get(outcome, "white")
        table.add_row(name, f"[{style}]{outcome}[/{style}]", f"{duration:.2f}")
        if outcome == "passed":
            passed += 1
        elif outcome == "failed":
            failed += 1
        else:
            skipped += 1
    console.print(table)
    console.print(
        f"[bold]{passed} passed[/bold], {failed} failed, {skipped} skipped "
        f"across {len(collector.rows)} tests."
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
