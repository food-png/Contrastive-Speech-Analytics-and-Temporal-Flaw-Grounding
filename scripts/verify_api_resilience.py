#!/usr/bin/env python3
"""Audit LLM key rotation, request pacing, and deterministic fallback. No live API calls."""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.app import EVAL_SLOTS, MAX_AUDIO_SEC, MAX_UPLOAD_BYTES, wav_duration_seconds
from utils.api_guard import APIGuardManager, GuardSettings, ProviderHTTPError
from utils.causal_explainer import CausalExplainer, compact_flaw_prompt
from utils.api_guard import set_api_guard

console = Console()


def _settings() -> GuardSettings:
    return GuardSettings(grok_model="grok-2", gemini_model="gemini-2.0-flash")


def _pace_transport(provider: str, key: str, prompt: str, system_instruction: str, settings: GuardSettings) -> str:
    return f"{provider}:{key[-1]}:{prompt[:12]}"


def _http_error(status: int) -> ProviderHTTPError:
    return ProviderHTTPError(status, f"simulated {status}")


def scenario_pacing() -> dict:
    guard = APIGuardManager(
        grok_keys=["grok-key-1"],
        gemini_keys=[],
        transport=_pace_transport,
        min_interval_sec=0.5,
        cooldown_sec=60,
        settings=_settings(),
    )
    started = time.perf_counter()
    texts = [guard.call_llm(f"flaw {index}", "coach", fallback_text="template") for index in range(3)]
    elapsed = time.perf_counter() - started
    waits = [row["wait_sec"] for row in guard.attempts if row["event"] == "ok"]
    ok = elapsed >= 0.9 and sum(waits) >= 0.95 and all(text.startswith("grok:") for text in texts)
    return {"name": "Rapid-fire pacing", "ok": ok, "elapsed": elapsed, "waits": waits, "guard": guard, "texts": texts}


def scenario_rotate() -> dict:
    def transport(provider: str, key: str, prompt: str, system_instruction: str, settings: GuardSettings) -> str:
        if key.endswith("1"):
            raise _http_error(429)
        return "coaching from secondary key"

    guard = APIGuardManager(
        grok_keys=["grok-key-1", "grok-key-2"],
        gemini_keys=["gemini-key-a"],
        transport=transport,
        min_interval_sec=0.0,
        cooldown_sec=60,
        settings=_settings(),
    )
    text = guard.call_llm("pause delta 2.4s", "coach", fallback_text="TEMPLATE")
    statuses = {row["key"]: row for row in guard.status_rows()}
    primary = next(row for row in guard.status_rows() if row["key"].endswith("y-1") or "key-1" in row["key"] or row["key"].startswith("grok"))
    # masked form is grok...-2 style: "grok...-2" from grok-key-2 -> first4 ... last2 = "grok...-2"
    cooled = [row for row in guard.status_rows() if row["provider"] == "grok" and row["status"] == "rate_limited"]
    used_secondary = any(row["event"] == "ok" and row["provider"] == "grok" for row in guard.attempts)
    ok = text == "coaching from secondary key" and len(cooled) == 1 and used_secondary
    return {"name": "429 key rotation", "ok": ok, "text": text, "guard": guard, "statuses": statuses, "primary": primary}


def scenario_provider_failover() -> dict:
    def transport(provider: str, key: str, prompt: str, system_instruction: str, settings: GuardSettings) -> str:
        if provider == "grok":
            raise _http_error(429)
        return "gemini failover text"

    guard = APIGuardManager(
        grok_keys=["grok-only"],
        gemini_keys=["gemini-backup"],
        transport=transport,
        min_interval_sec=0.0,
        cooldown_sec=60,
        settings=_settings(),
    )
    text = guard.call_llm("rush 1.6x", "coach", fallback_text="TEMPLATE")
    ok = text == "gemini failover text" and any(row["provider"] == "gemini" and row["event"] == "ok" for row in guard.attempts)
    return {"name": "Grok to Gemini failover", "ok": ok, "text": text, "guard": guard}


def scenario_total_failure() -> dict:
    def transport(provider: str, key: str, prompt: str, system_instruction: str, settings: GuardSettings) -> str:
        raise _http_error(429 if provider == "grok" else 403)

    guard = APIGuardManager(
        grok_keys=["grok-a", "grok-b"],
        gemini_keys=["gem-a", "gem-b"],
        transport=transport,
        min_interval_sec=0.0,
        cooldown_sec=60,
        settings=_settings(),
    )
    raised = False
    try:
        text = guard.call_llm("spike +12 dB", "coach", fallback_text="LOCAL TEMPLATE")
    except Exception as exc:  # noqa: BLE001
        raised = True
        text = f"RAISED {exc}"
    set_api_guard(guard)
    try:
        diagnosis = CausalExplainer().enrich_with_llm(
            {
                "category": "pause",
                "sub_type": "awkward_pause",
                "start_sec": 1.0,
                "end_sec": 3.2,
                "target_phrase": "looking forward",
                "math_evidence": "Unscheduled silence of 2.20s.",
                "causal_rationale": "Prolonged hesitation between clauses.",
                "impact": "Breaks syntactic momentum.",
                "mathematical_evidence_raw": {"delta_pause_sec": 2.2},
            }
        )
    finally:
        set_api_guard(None)
    huge = compact_flaw_prompt({"target_phrase": "word " * 800, "math_evidence": "x" * 2000, "mathematical_evidence_raw": {}})
    ok = (not raised) and text == "LOCAL TEMPLATE" and "hesitation" in diagnosis and len(huge) <= 900
    cooled = sum(1 for row in guard.status_rows() if row["status"] == "rate_limited")
    return {
        "name": "All keys down, local template",
        "ok": ok,
        "text": text,
        "diagnosis": diagnosis,
        "prompt_chars": len(huge),
        "cooled": cooled,
        "guard": guard,
    }


def scenario_upload_guards() -> dict:
    header_frames = int(200 * 16_000)
    channels = 1
    rate = 16_000
    bits = 16
    block_align = channels * bits // 8
    payload = b"\x00\x00" * 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(payload),
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        rate,
        rate * block_align,
        block_align,
        bits,
        b"data",
        header_frames * block_align,
    )
    blob = header + payload
    duration = wav_duration_seconds(blob)
    ok = duration > MAX_AUDIO_SEC and EVAL_SLOTS._value == 2 and MAX_UPLOAD_BYTES == 25 * 1024 * 1024
    return {"name": "Upload duration and concurrency caps", "ok": ok, "duration": duration}


def _print_attempts(guard: APIGuardManager) -> None:
    table = Table(title="Attempt log")
    table.add_column("Provider")
    table.add_column("Key")
    table.add_column("Event")
    table.add_column("Wait s", justify="right")
    for row in guard.attempts:
        table.add_row(row["provider"], row["key"] or "-", row["event"], f"{row['wait_sec']:.3f}")
    console.print(table)


def main() -> None:
    console.print(Panel.fit("[bold]Track C API resilience audit[/bold]\nSimulated 429s, pacing, and local fallback", border_style="cyan"))
    results = [
        scenario_pacing(),
        scenario_rotate(),
        scenario_provider_failover(),
        scenario_total_failure(),
        scenario_upload_guards(),
    ]

    summary = Table(title="Resilience summary")
    summary.add_column("Check")
    summary.add_column("Result")
    summary.add_column("Detail")
    for row in results:
        detail = ""
        if row["name"].startswith("Rapid"):
            detail = f"elapsed {row['elapsed']:.2f}s waits {row['waits']}"
        elif "rotation" in row["name"] or "failover" in row["name"]:
            detail = str(row.get("text", ""))[:60]
        elif row["name"].startswith("All"):
            detail = f"fallback={row['text']} cooled_keys={row['cooled']} prompt_chars={row['prompt_chars']}"
        else:
            detail = f"header duration {row['duration']:.1f}s > {MAX_AUDIO_SEC:.0f}s, slots=2"
        style = "green" if row["ok"] else "red"
        summary.add_row(row["name"], f"[{style}]{'PASS' if row['ok'] else 'FAIL'}[/{style}]", detail)
    console.print(summary)

    console.print("\n[bold]Pacing delays[/bold]")
    _print_attempts(results[0]["guard"])
    console.print("[bold]Primary key cooled, secondary served the call[/bold]")
    _print_attempts(results[1]["guard"])
    status = Table(title="Key status after 429")
    status.add_column("Provider")
    status.add_column("Key")
    status.add_column("Status")
    status.add_column("Cooldown s", justify="right")
    status.add_column("Errors", justify="right")
    for key_row in results[1]["guard"].status_rows():
        status.add_row(
            key_row["provider"],
            key_row["key"],
            key_row["status"],
            f"{key_row['cooldown_remaining_sec']:.1f}",
            str(key_row["error_count"]),
        )
    console.print(status)
    console.print("[bold]Provider failover log[/bold]")
    _print_attempts(results[2]["guard"])
    console.print("[bold]Total outage log[/bold]")
    _print_attempts(results[3]["guard"])

    if not all(row["ok"] for row in results):
        console.print("[red]Resilience audit failed.[/red]")
        raise SystemExit(1)
    console.print("[bold green]Resilience audit passed.[/bold green]")


if __name__ == "__main__":
    main()
