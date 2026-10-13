"""Thread-safe LLM gateway: key rotation, token bucket, and local fallback.

Failover order is Grok, then Gemini, then the caller-supplied deterministic text.
Outbound failures are recorded and never raised to the speech pipeline.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from utils.env_loader import load_project_env

GROK_URL_PATH = "/chat/completions"
DEFAULT_GROK_BASE = "https://api.x.ai/v1"
TRANSIENT_STATUS = {408, 500, 502, 503, 504}
THROTTLE_STATUS = {429, 403}
MAX_PROMPT_CHARS = 1_500
MAX_OUTPUT_TOKENS = 160


class ProviderHTTPError(Exception):
    """HTTP outcome the guard can classify as throttle, transient, or fatal."""

    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status}")
        self.status = int(status)


@dataclass
class KeyState:
    provider: str
    key: str
    error_count: int = 0
    rate_limited_until: float = 0.0

    @property
    def status(self) -> str:
        if self.rate_limited_until > time.monotonic():
            return "rate_limited"
        return "healthy"

    def masked(self) -> str:
        raw = self.key
        tail = raw[-2:] if len(raw) >= 2 else "**"
        return f"{self.provider}:{tail}"


class GuardSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    grok_api_keys: str = ""
    grok_api_key: str = ""
    gemini_api_keys: str = ""
    gemini_api_key: str = ""
    google_api_key: str = ""
    llm_api_key: str = ""
    grok_model: str = "grok-2"
    grok_base_url: str = DEFAULT_GROK_BASE
    gemini_model: str = "gemini-2.0-flash"
    llm_min_interval_sec: float = 0.75
    llm_key_cooldown_sec: float = 60.0


Transport = Callable[[str, str, str, str, GuardSettings], str]


@dataclass
class APIGuardManager:
    """Rotate provider keys and pace outbound LLM calls."""

    grok_keys: list[str] | None = None
    gemini_keys: list[str] | None = None
    transport: Transport | None = None
    min_interval_sec: float | None = None
    cooldown_sec: float | None = None
    settings: GuardSettings | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _states: dict[str, list[KeyState]] = field(default_factory=dict, init=False, repr=False)
    _cursor: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _last_outbound: float = field(default=0.0, init=False, repr=False)
    delays: list[float] = field(default_factory=list, init=False, repr=False)
    attempts: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        load_project_env()
        if self.settings is None:
            self.settings = GuardSettings()
        if self.min_interval_sec is None:
            self.min_interval_sec = max(0.0, float(self.settings.llm_min_interval_sec))
        if self.cooldown_sec is None:
            self.cooldown_sec = max(1.0, float(self.settings.llm_key_cooldown_sec))
        explicit = self.grok_keys is not None or self.gemini_keys is not None
        grok = list(self.grok_keys or [])
        gemini = list(self.gemini_keys or [])
        if not explicit:
            grok = _split_keys(self.settings.grok_api_keys, self.settings.grok_api_key)
            gemini = _split_keys(
                self.settings.gemini_api_keys,
                self.settings.gemini_api_key,
                self.settings.google_api_key,
                self.settings.llm_api_key,
            )
        self._states = {
            "grok": [KeyState("grok", key) for key in _dedupe(grok)],
            "gemini": [KeyState("gemini", key) for key in _dedupe(gemini)],
        }
        self._cursor = {"grok": 0, "gemini": 0}
        if self.transport is None:
            self.transport = default_transport

    def call_llm(
        self,
        prompt: str,
        system_instruction: str = "",
        fallback_text: str = "",
        extra_gemini_key: str | None = None,
    ) -> str:
        """Return model text, or ``fallback_text`` when every provider is unavailable."""
        prompt = cap_text(prompt)
        system_instruction = cap_text(system_instruction, limit=400)
        try:
            text = self._dispatch(prompt, system_instruction, extra_gemini_key)
        except Exception as exc:  # noqa: BLE001 — pipeline must survive guard bugs
            self._record("local", "template", "guard_error", 0.0, str(exc))
            return fallback_text
        if text and text.strip():
            return text.strip()
        self._record("local", "template", "deterministic_fallback", 0.0, "")
        return fallback_text

    def status_rows(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        rows: list[dict[str, Any]] = []
        with self._lock:
            for provider in ("grok", "gemini"):
                for state in self._states.get(provider, []):
                    remaining = max(0.0, state.rate_limited_until - now)
                    rows.append(
                        {
                            "provider": provider,
                            "key": state.masked(),
                            "status": "rate_limited" if remaining > 0 else "healthy",
                            "cooldown_remaining_sec": round(remaining, 1),
                            "error_count": state.error_count,
                        }
                    )
        return rows

    def _dispatch(self, prompt: str, system_instruction: str, extra_gemini_key: str | None) -> str | None:
        if extra_gemini_key:
            self._prepend_key("gemini", extra_gemini_key.strip())
        for provider in ("grok", "gemini"):
            text = self._try_provider(provider, prompt, system_instruction)
            if text:
                return text
        return None

    def _try_provider(self, provider: str, prompt: str, system_instruction: str) -> str | None:
        pool = self._states.get(provider) or []
        if not pool:
            return None
        for _ in range(len(pool)):
            state = self._next_healthy(provider)
            if state is None:
                self._record(provider, "", "pool_exhausted", 0.0, "all keys cooling down")
                return None
            waited = self._acquire_slot()
            try:
                text = self._request_with_retry(provider, state.key, prompt, system_instruction)
            except ProviderHTTPError as exc:
                self._on_http_error(state, exc)
                self._record(provider, state.masked(), f"http_{exc.status}", waited, str(exc))
                continue
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    state.error_count += 1
                self._record(provider, state.masked(), "error", waited, type(exc).__name__)
                continue
            if text and text.strip():
                self._record(provider, state.masked(), "ok", waited, "")
                return text.strip()
            self._record(provider, state.masked(), "empty", waited, "")
        return None

    def _request_with_retry(self, provider: str, key: str, prompt: str, system_instruction: str) -> str:
        assert self.transport is not None
        assert self.settings is not None

        def _invoke() -> str:
            return self.transport(provider, key, prompt, system_instruction, self.settings)

        retrying = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=1, exp_base=2, max=4, jitter=0.5),
            retry=retry_if_exception(_is_transient),
            reraise=True,
        )
        return retrying(_invoke)

    def _on_http_error(self, state: KeyState, exc: ProviderHTTPError) -> None:
        with self._lock:
            state.error_count += 1
            if exc.status in THROTTLE_STATUS:
                state.rate_limited_until = time.monotonic() + float(self.cooldown_sec or 60.0)

    def _next_healthy(self, provider: str) -> KeyState | None:
        with self._lock:
            pool = self._states.get(provider) or []
            if not pool:
                return None
            now = time.monotonic()
            for _ in range(len(pool)):
                index = self._cursor[provider] % len(pool)
                self._cursor[provider] += 1
                state = pool[index]
                if state.rate_limited_until <= now:
                    return state
        return None

    def _prepend_key(self, provider: str, key: str) -> None:
        if not key:
            return
        with self._lock:
            pool = self._states.setdefault(provider, [])
            if any(state.key == key for state in pool):
                return
            pool.insert(0, KeyState(provider, key))

    def _acquire_slot(self) -> float:
        interval = float(self.min_interval_sec or 0.0)
        with self._lock:
            now = time.monotonic()
            wait = interval - (now - self._last_outbound)
            if wait < 0:
                wait = 0.0
            self._last_outbound = now + wait
            self.delays.append(wait)
        if wait > 0:
            time.sleep(wait)
        return wait

    def _record(self, provider: str, key: str, event: str, waited: float, detail: str) -> None:
        row = {
            "provider": provider,
            "key": key,
            "event": event,
            "wait_sec": round(waited, 3),
            "detail": detail[:180],
        }
        with self._lock:
            self.attempts.append(row)


def cap_text(text: str, limit: int = MAX_PROMPT_CHARS) -> str:
    """Keep prompts inside a fixed character budget so one flaw cannot burn the quota."""
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def default_transport(
    provider: str,
    key: str,
    prompt: str,
    system_instruction: str,
    settings: GuardSettings,
) -> str:
    if provider == "grok":
        return _post_grok(key, prompt, system_instruction, settings)
    if provider == "gemini":
        return _post_gemini(key, prompt, system_instruction, settings)
    raise ProviderHTTPError(400, f"unknown provider {provider}")


def _post_grok(key: str, prompt: str, system_instruction: str, settings: GuardSettings) -> str:
    url = settings.grok_base_url.rstrip("/") + GROK_URL_PATH
    body = {
        "model": settings.grok_model or "grok-2",
        "temperature": 0.2,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": [
            {"role": "system", "content": system_instruction or "You are a concise oratory coach."},
            {"role": "user", "content": prompt},
        ],
    }
    with httpx.Client(timeout=12.0) as client:
        response = client.post(
            url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
    _raise_for_status(response)
    payload = response.json()
    try:
        return str(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderHTTPError(502, "grok response missing message content") from exc


def _post_gemini(key: str, prompt: str, system_instruction: str, settings: GuardSettings) -> str:
    model = settings.gemini_model or "gemini-2.0-flash"
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    combined = f"{system_instruction}\n\n{prompt}".strip()
    body = {
        "contents": [{"parts": [{"text": combined}]}],
        "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": 0.2},
    }
    with httpx.Client(timeout=12.0) as client:
        response = client.post(url, params={"key": key}, json=body)
    _raise_for_status(response)
    payload = response.json()
    try:
        return str(payload["candidates"][0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderHTTPError(502, "gemini response missing text") from exc


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code in THROTTLE_STATUS or response.status_code in TRANSIENT_STATUS:
        raise ProviderHTTPError(response.status_code, response.text[:200])
    if response.status_code >= 400:
        raise ProviderHTTPError(response.status_code, response.text[:200])


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, ProviderHTTPError) and exc.status in TRANSIENT_STATUS


def _split_keys(*chunks: str) -> list[str]:
    found: list[str] = []
    for chunk in chunks:
        for part in str(chunk or "").split(","):
            key = part.strip()
            if key:
                found.append(key)
    return found


def _dedupe(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


_GUARD: APIGuardManager | None = None
_GUARD_LOCK = threading.Lock()


def get_api_guard() -> APIGuardManager:
    global _GUARD
    with _GUARD_LOCK:
        if _GUARD is None:
            _GUARD = APIGuardManager()
        return _GUARD


def set_api_guard(guard: APIGuardManager | None) -> None:
    global _GUARD
    with _GUARD_LOCK:
        _GUARD = guard
