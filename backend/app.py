"""Track C Phase 6 — FastAPI dashboard."""

from __future__ import annotations

import asyncio
import io
import sys
import wave
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response as StarletteResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.catalog import DATASET, list_samples, sample_payload  # noqa: E402
from backend.live_eval import evaluate_upload  # noqa: E402

app = FastAPI(title="Track C — Contrastive Speech Analytics", version="1.0.0")

class FreshStaticFiles(StaticFiles):
    """Avoid a stale app.js leaving the preset dropdown on its placeholder."""

    async def get_response(self, path: str, scope):  # noqa: ANN001
        response = await super().get_response(path, scope)
        if isinstance(response, StarletteResponse) and path.endswith((".js", ".css", ".html")):
            response.headers["Cache-Control"] = "no-cache"
        return response


templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", FreshStaticFiles(directory=str(ROOT / "static")), name="static")
# Starlette FileResponse honors Range requests (HTTP 206) so WaveSurfer can seek.
app.mount("/dataset", StaticFiles(directory=str(DATASET)), name="dataset")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_AUDIO_SEC = 180.0
# Heavy alignment + feature extraction stays at two jobs so uploads cannot exhaust RAM.
EVAL_SLOTS = asyncio.Semaphore(2)

FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#020617"/>
  <path d="M6 34h6l4-12 6 22 6-28 6 30 5-16 4 8h11" fill="none" stroke="#22d3ee" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M6 34h6l4-12 6 22 6-28 6 30 5-16 4 8h11" fill="none" stroke="#67e8f9" stroke-width="1.5" stroke-linecap="round" opacity="0.85"/>
</svg>"""


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/json/version", include_in_schema=False)
def devtools_version_probe() -> JSONResponse:
    """Answer Chrome/extension DevTools probes so they do not log a 404."""
    return JSONResponse(
        {
            "Browser": "speech_hack1",
            "Protocol-Version": "1.3",
            "User-Agent": "Track C dashboard",
            "V8-Version": "0",
            "WebKit-Version": "0",
        }
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.get("/api/samples")
def api_samples() -> dict:
    return {"samples": list_samples()}


@app.get("/api/sample/{sample_id}")
def api_sample(sample_id: str) -> dict:
    try:
        return sample_payload(sample_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def wav_duration_seconds(raw: bytes) -> float:
    """Read duration from the WAV header without decoding the whole pipeline."""
    try:
        with wave.open(io.BytesIO(raw), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
    except wave.Error as exc:
        raise ValueError("Could not read WAV header.") from exc
    if rate <= 0 or frames < 0:
        raise ValueError("WAV header has no duration.")
    return frames / float(rate)


@app.post("/api/evaluate")
async def api_evaluate(
    audio: UploadFile = File(...),
    baseline_id: str = Form(...),
    transcript: str = Form(""),
    sample_id: str = Form(""),
) -> dict:
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds 25 MB.")
    name = (audio.filename or "").lower()
    if name and not name.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Upload a WAV file.")
    try:
        duration = wav_duration_seconds(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if duration > MAX_AUDIO_SEC:
        raise HTTPException(
            status_code=413,
            detail=f"Audio is {duration:.1f}s; the limit is {MAX_AUDIO_SEC:.0f}s.",
        )
    sid = sample_id.strip() or None
    try:
        async with EVAL_SLOTS:
            return await asyncio.to_thread(
                evaluate_upload,
                raw,
                baseline_id.strip(),
                transcript,
                sid,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface pipeline failures to the UI
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc
