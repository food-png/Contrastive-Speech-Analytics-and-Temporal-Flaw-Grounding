"""
Phase 2 — Speaker-agnostic forced alignment (PyTorch / torchaudio).

Primary backend: ``torchaudio.pipelines.MMS_FA``
Fallback acoustic model: ``WAV2VEC2_ASR_BASE_960H`` + CTC trellis
Last-resort: energy-aware duration packing (unaligned / OOM / empty path)

Frame times use ``time_sec = frame_index * (duration_sec / n_frames)``.
Pauses are explicit spans where the inter-word (or edge) gap is >= 0.15 s.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field, model_validator

PAUSE_MIN_SEC = 0.15
ALIGN_SR = 16_000

# MMS_FA character inventory (plus blank '-' and optional '*')
_MMS_CHARS = set("abcdefghijklmnopqrstuvwxyz'")

CONTRACTIONS: dict[str, str] = {
    "don't": "dont",
    "doesn't": "doesnt",
    "didn't": "didnt",
    "can't": "cant",
    "won't": "wont",
    "isn't": "isnt",
    "aren't": "arent",
    "wasn't": "wasnt",
    "weren't": "werent",
    "couldn't": "couldnt",
    "shouldn't": "shouldnt",
    "wouldn't": "wouldnt",
    "i'm": "im",
    "you're": "youre",
    "we're": "were",
    "they're": "theyre",
    "it's": "its",
    "that's": "thats",
    "there's": "theres",
    "here's": "heres",
    "who's": "whos",
    "what's": "whats",
    "let's": "lets",
    "i've": "ive",
    "we've": "weve",
    "you've": "youve",
    "they've": "theyve",
    "i'll": "ill",
    "you'll": "youll",
    "we'll": "well",
    "they'll": "theyll",
    "i'd": "id",
    "you'd": "youd",
    "we'd": "wed",
    "they'd": "theyd",
}


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class WordSegment(BaseModel):
    word: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self) -> "WordSegment":
        if self.end_sec < self.start_sec:
            raise ValueError(f"word end {self.end_sec} < start {self.start_sec}")
        return self


class PauseSegment(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    duration_sec: float = Field(ge=0)

    @model_validator(mode="after")
    def consistent(self) -> "PauseSegment":
        expected = round(self.end_sec - self.start_sec, 6)
        if self.end_sec < self.start_sec:
            raise ValueError("pause end precedes start")
        if abs(self.duration_sec - (self.end_sec - self.start_sec)) > 0.02:
            self.duration_sec = round(self.end_sec - self.start_sec, 6)
        _ = expected
        return self


class AlignmentResult(BaseModel):
    sample_id: str
    audio_path: str
    words: list[WordSegment]
    pauses: list[PauseSegment]
    transcript_path: str | None = None
    duration_sec: float | None = None
    backend: str | None = None
    device: str | None = None


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedWord:
    surface: str
    normalized: str
    index: int


def _strip_punct(token: str) -> str:
    token = unicodedata.normalize("NFKC", token)
    token = token.replace("\u2019", "'").replace("`", "'").replace("\u2018", "'")
    token = token.replace("\u2014", " ").replace("\u2013", " ").replace("—", " ").replace("–", " ")
    token = re.sub(r"[^a-zA-Z0-9']+", "", token)
    return token


def normalize_transcript(text: str) -> list[NormalizedWord]:
    """
    Lower-case, expand/standardize contractions, drop punctuation.

    Returns aligned (surface-form, normalized) pairs. Empty normalized
    tokens (punctuation-only) are dropped but surface is retained when
    the token still contains letters after cleaning.
    """
    raw_tokens = re.findall(r"\S+", text.strip())
    out: list[NormalizedWord] = []
    for i, surface in enumerate(raw_tokens):
        cleaned = _strip_punct(surface).lower()
        if not cleaned:
            continue
        cleaned = CONTRACTIONS.get(cleaned, cleaned)
        cleaned = cleaned.replace("'", "")
        cleaned = "".join(ch for ch in cleaned if ch in _MMS_CHARS or ch.isdigit())
        # digits: romanize simply by spelling nothing — keep if alphabetic remains
        cleaned = re.sub(r"[0-9]+", "", cleaned)
        if not cleaned:
            continue
        out.append(NormalizedWord(surface=surface, normalized=cleaned, index=i))
    return out


# ---------------------------------------------------------------------------
# Device / audio helpers
# ---------------------------------------------------------------------------

def select_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_mono_16k(path: str | Path) -> tuple[np.ndarray, int, float]:
    y, sr = sf.read(str(path), always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    if sr != ALIGN_SR:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=ALIGN_SR).astype(np.float32)
        sr = ALIGN_SR
    duration = float(len(y) / sr) if sr else 0.0
    return y, sr, duration


def _frame_to_sec(frame: float, n_frames: int, duration_sec: float) -> float:
    if n_frames <= 0 or duration_sec <= 0:
        return 0.0
    return float(frame) * (duration_sec / float(n_frames))


def _extract_pauses(
    words: list[WordSegment],
    duration_sec: float,
    min_sec: float = PAUSE_MIN_SEC,
) -> list[PauseSegment]:
    pauses: list[PauseSegment] = []
    if duration_sec <= 0:
        return pauses

    cursor = 0.0
    for w in words:
        gap = w.start_sec - cursor
        if gap >= min_sec:
            pauses.append(
                PauseSegment(
                    start_sec=round(cursor, 4),
                    end_sec=round(w.start_sec, 4),
                    duration_sec=round(gap, 4),
                )
            )
        cursor = max(cursor, w.end_sec)
    tail = duration_sec - cursor
    if tail >= min_sec:
        pauses.append(
            PauseSegment(
                start_sec=round(cursor, 4),
                end_sec=round(duration_sec, 4),
                duration_sec=round(tail, 4),
            )
        )
    return pauses


def _enforce_monotonic(
    words: list[WordSegment],
    duration_sec: float,
) -> list[WordSegment]:
    """Clip to [0, duration] and make start/end non-decreasing across words."""
    if not words:
        return words
    cleaned: list[WordSegment] = []
    prev_end = 0.0
    n = len(words)
    for i, w in enumerate(words):
        start = max(0.0, min(w.start_sec, duration_sec))
        end = max(0.0, min(w.end_sec, duration_sec))
        start = max(start, prev_end)
        if end < start:
            end = start
        # leave a 1-sample epsilon so later words can progress
        if i < n - 1 and end >= duration_sec:
            end = min(duration_sec, start + 1e-3)
        cleaned.append(
            WordSegment(
                word=w.word,
                start_sec=round(start, 4),
                end_sec=round(end, 4),
                confidence=float(np.clip(w.confidence, 0.0, 1.0)),
            )
        )
        prev_end = cleaned[-1].end_sec
    return cleaned


# ---------------------------------------------------------------------------
# Energy-aware fallback (unaligned tokens / failed CTC / synthetic audio)
# ---------------------------------------------------------------------------

def _rms_envelope(y: np.ndarray, sr: int, hop_ms: float = 10.0, win_ms: float = 30.0) -> np.ndarray:
    hop = max(1, int(sr * hop_ms / 1000.0))
    win = max(hop, int(sr * win_ms / 1000.0))
    if len(y) < win:
        return np.array([float(np.sqrt(np.mean(y * y) + 1e-12))], dtype=np.float32)
    n_frames = 1 + (len(y) - win) // hop
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        sl = y[i * hop : i * hop + win]
        rms[i] = float(np.sqrt(np.mean(sl * sl) + 1e-12))
    return rms


def _voiced_mask(rms: np.ndarray) -> np.ndarray:
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    p80 = float(np.percentile(rms, 80))
    p20 = float(np.percentile(rms, 20))
    thr = max(1e-4, 0.25 * p80 + 0.05 * p20)
    voiced = rms >= thr
    # close holes shorter than ~80 ms (8 frames at 10 ms)
    k = 4
    if voiced.size > 2:
        padded = np.pad(voiced.astype(np.int32), (k, k), constant_values=0)
        closed = np.zeros_like(voiced, dtype=bool)
        for i in range(len(voiced)):
            closed[i] = np.max(padded[i : i + 2 * k + 1]) > 0 and (
                np.mean(padded[i : i + 2 * k + 1]) > 0.25 or voiced[i]
            )
        # reopen only long silence: if a run of False in original is long, keep it
        out = voiced.copy()
        i = 0
        while i < len(voiced):
            if not voiced[i]:
                j = i
                while j < len(voiced) and not voiced[j]:
                    j += 1
                if (j - i) <= 8:
                    out[i:j] = True
                i = j
            else:
                i += 1
        return out
    return voiced


def _voiced_to_real(tv: float, regions: list[tuple[float, float]]) -> float:
    """Map a time on the concatenated voiced axis back onto the full timeline."""
    remaining = max(0.0, float(tv))
    last_end = regions[-1][1] if regions else 0.0
    for start, end in regions:
        dur = end - start
        if remaining <= dur:
            return start + remaining
        remaining -= dur
    return last_end


def energy_pack_words(
    y: np.ndarray,
    sr: int,
    tokens: list[NormalizedWord],
    duration_sec: float,
) -> list[WordSegment]:
    """Pack transcript words into voiced regions, weighted by character length."""
    if not tokens or duration_sec <= 0:
        return []
    rms = _rms_envelope(y, sr)
    voiced = _voiced_mask(rms)
    n_frames = max(len(rms), 1)
    sec_per = duration_sec / float(n_frames)

    regions: list[tuple[float, float]] = []
    i = 0
    while i < n_frames:
        if i < len(voiced) and bool(voiced[i]):
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            regions.append((i * sec_per, min(duration_sec, j * sec_per)))
            i = j
        else:
            i += 1
    if not regions:
        regions = [(0.05 * duration_sec, 0.95 * duration_sec)]

    weights = np.array([max(1, len(t.normalized)) for t in tokens], dtype=np.float64)
    weight_sum = float(weights.sum()) or 1.0
    weights = weights / weight_sum
    total_voiced = sum(e - s for s, e in regions)
    if total_voiced <= 1e-6:
        total_voiced = duration_sec
        regions = [(0.0, duration_sec)]

    words: list[WordSegment] = []
    voiced_cursor = 0.0
    for tok, w in zip(tokens, weights):
        span = max(1e-3, float(w) * total_voiced)
        start = _voiced_to_real(voiced_cursor, regions)
        end = _voiced_to_real(voiced_cursor + span, regions)
        for r_s, r_e in regions:
            if r_s - 1e-4 <= start <= r_e + 1e-4:
                end = min(end, r_e)
                break
        voiced_cursor += span
        words.append(
            WordSegment(
                word=tok.surface,
                start_sec=round(start, 4),
                end_sec=round(max(start + 1e-3, end), 4),
                confidence=0.35,
            )
        )
    return _enforce_monotonic(words, duration_sec)


# ---------------------------------------------------------------------------
# CTC backends
# ---------------------------------------------------------------------------

class ForcedAligner:
    """
    Lazy-loaded aligner. Prefers MMS_FA, then wav2vec2-base-960h CTC,
    then energy packing. OOM automatically retries on CPU.
    """

    def __init__(self, prefer: Literal["mms", "wav2vec2", "energy"] = "mms"):
        self.prefer = prefer
        self.device = select_device()
        self._mms = None
        self._w2v = None
        self._torch = None

    def _import_torch(self):
        if self._torch is None:
            import torch
            import torchaudio

            self._torch = (torch, torchaudio)
        return self._torch

    def _load_mms(self):
        if self._mms is not None:
            return self._mms
        torch, torchaudio = self._import_torch()
        bundle = torchaudio.pipelines.MMS_FA
        try:
            model = bundle.get_model()
        except TypeError:
            model = bundle.get_model(with_star=True)
        try:
            model = model.to(self.device)
        except Exception:
            self.device = torch.device("cpu")
            model = model.to(self.device)
        model.eval()
        self._mms = {
            "bundle": bundle,
            "model": model,
            "tokenizer": bundle.get_tokenizer(),
            "aligner": bundle.get_aligner(),
            "sample_rate": int(getattr(bundle, "sample_rate", ALIGN_SR)),
        }
        return self._mms

    def _load_w2v(self):
        if self._w2v is not None:
            return self._w2v
        torch, torchaudio = self._import_torch()
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model()
        try:
            model = model.to(self.device)
        except Exception:
            self.device = torch.device("cpu")
            model = model.to(self.device)
        model.eval()
        labels = bundle.get_labels()
        label_to_id = {c: i for i, c in enumerate(labels)}
        blank = 0
        for i, lab in enumerate(labels):
            if lab in ("<s>", "<blank>", "-", "<pad>"):
                blank = i
                break
        self._w2v = {
            "bundle": bundle,
            "model": model,
            "labels": labels,
            "label_to_id": label_to_id,
            "blank": blank,
            "sample_rate": int(getattr(bundle, "sample_rate", ALIGN_SR)),
        }
        return self._w2v

    def _to_tensor(self, y: np.ndarray):
        torch, _ = self._import_torch()
        wav = torch.from_numpy(np.ascontiguousarray(y)).float().unsqueeze(0)
        return wav

    def _align_mms(
        self,
        y: np.ndarray,
        sr: int,
        tokens: list[NormalizedWord],
        duration_sec: float,
    ) -> tuple[list[WordSegment], float]:
        torch, torchaudio = self._import_torch()
        pack = self._load_mms()
        wav = self._to_tensor(y)
        if sr != pack["sample_rate"]:
            wav = torchaudio.functional.resample(wav, sr, pack["sample_rate"])
        words_norm = [t.normalized for t in tokens]
        if not words_norm:
            return [], 0.0

        def _run(dev) -> tuple[list, int]:
            model = pack["model"].to(dev)
            pack["model"] = model
            with torch.inference_mode():
                emission, _ = model(wav.to(dev))
                token_spans = pack["aligner"](emission[0], pack["tokenizer"](words_norm))
            return token_spans, int(emission.size(1))

        try:
            token_spans, n_frames = _run(self.device)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "oom" in msg:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                self.device = torch.device("cpu")
                token_spans, n_frames = _run(self.device)
            else:
                raise

        words: list[WordSegment] = []
        scores: list[float] = []
        for tok, spans in zip(tokens, token_spans):
            if not spans:
                continue
            start_f = float(spans[0].start)
            end_f = float(spans[-1].end)
            # TokenSpan.end is exclusive frame index in torchaudio
            denom = 0.0
            num = 0.0
            for sp in spans:
                length = max(1, int(sp.end) - int(sp.start))
                sc = float(getattr(sp, "score", 0.0))
                num += sc * length
                denom += length
            conf = float(num / denom) if denom else 0.0
            conf = float(np.clip(conf, 0.0, 1.0))
            words.append(
                WordSegment(
                    word=tok.surface,
                    start_sec=_frame_to_sec(start_f, n_frames, duration_sec),
                    end_sec=_frame_to_sec(end_f, n_frames, duration_sec),
                    confidence=conf,
                )
            )
            scores.append(conf)
        mean_conf = float(np.mean(scores)) if scores else 0.0
        return _enforce_monotonic(words, duration_sec), mean_conf

    def _align_wav2vec2(
        self,
        y: np.ndarray,
        sr: int,
        tokens: list[NormalizedWord],
        duration_sec: float,
    ) -> tuple[list[WordSegment], float]:
        torch, torchaudio = self._import_torch()
        pack = self._load_w2v()
        wav = self._to_tensor(y)
        if sr != pack["sample_rate"]:
            wav = torchaudio.functional.resample(wav, sr, pack["sample_rate"])

        ids: list[int] = []
        word_char_counts: list[int] = []
        unk = pack["label_to_id"].get("<unk>", pack["label_to_id"].get("[UNK]", 1))
        pipe = pack["label_to_id"].get("|")
        if pipe is None:
            raise RuntimeError("WAV2VEC2 labels missing word delimiter '|'")
        for tok in tokens:
            n = 0
            for ch in tok.normalized.upper():
                tid = pack["label_to_id"].get(ch, unk)
                ids.append(tid)
                n += 1
            if n == 0:
                continue
            ids.append(pipe)
            word_char_counts.append(n)

        if not ids:
            return [], 0.0
        if ids[-1] == pipe:
            ids = ids[:-1]

        def _run(dev):
            model = pack["model"].to(dev)
            pack["model"] = model
            with torch.inference_mode():
                emission, _ = model(wav.to(dev))
                logp = torch.log_softmax(emission, dim=-1)
                targets = torch.tensor(ids, device=dev, dtype=torch.int64).unsqueeze(0)
                aligned, scores = torchaudio.functional.forced_align(
                    logp, targets, blank=int(pack["blank"])
                )
                spans = torchaudio.functional.merge_tokens(aligned[0], scores[0], blank=int(pack["blank"]))
            return spans, int(emission.size(1))

        try:
            spans, n_frames = _run(self.device)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "oom" in msg:
                self.device = torch.device("cpu")
                spans, n_frames = _run(self.device)
            else:
                raise

        # Group character spans into words using '|' delimiters
        words: list[WordSegment] = []
        scores: list[float] = []
        buf: list = []
        token_iter = iter(tokens)
        pipe_id = pipe
        labels = pack["labels"]

        def _flush(buffer: list) -> None:
            if not buffer:
                return
            tok = next(token_iter, None)
            if tok is None:
                return
            start_f = float(buffer[0].start)
            end_f = float(buffer[-1].end)
            denom = 0.0
            num = 0.0
            for sp in buffer:
                length = max(1, int(sp.end) - int(sp.start))
                sc = float(getattr(sp, "score", 0.0))
                num += sc * length
                denom += length
            conf = float(np.clip(num / denom if denom else 0.0, 0.0, 1.0))
            words.append(
                WordSegment(
                    word=tok.surface,
                    start_sec=_frame_to_sec(start_f, n_frames, duration_sec),
                    end_sec=_frame_to_sec(end_f, n_frames, duration_sec),
                    confidence=conf,
                )
            )
            scores.append(conf)

        for sp in spans:
            lab = labels[int(sp.token)] if 0 <= int(sp.token) < len(labels) else ""
            if lab == "|" or int(sp.token) == pipe_id:
                _flush(buf)
                buf = []
            else:
                buf.append(sp)
        _flush(buf)
        mean_conf = float(np.mean(scores)) if scores else 0.0
        return _enforce_monotonic(words, duration_sec), mean_conf

    def align_file(
        self,
        audio_path: str | Path,
        transcript: str,
        sample_id: str,
        rel_audio: str | None = None,
        transcript_path: str | None = None,
        min_confidence: float = 0.08,
    ) -> AlignmentResult:
        audio_path = Path(audio_path)
        y, sr, duration = _load_mono_16k(audio_path)
        tokens = normalize_transcript(transcript)
        rel = rel_audio or str(audio_path).replace("\\", "/")

        words: list[WordSegment] = []
        backend = "energy_pack"
        mean_conf = 0.0
        errors: list[str] = []

        if self.prefer == "energy":
            backends = ["energy"]
        elif self.prefer == "wav2vec2":
            backends = ["wav2vec2", "energy"]
        else:
            backends = ["mms", "energy"]

        for name in backends:
            if name == "energy":
                words = energy_pack_words(y, sr, tokens, duration)
                note = "; ".join(errors[:2])
                backend = "energy_pack" + (f" ({note})" if note else "")
                break
            try:
                if name == "mms":
                    words, mean_conf = self._align_mms(y, sr, tokens, duration)
                else:
                    words, mean_conf = self._align_wav2vec2(y, sr, tokens, duration)
            except Exception as exc:  # noqa: BLE001 — backend isolation
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                words = []
                continue
            if len(words) < max(1, int(0.5 * len(tokens))):
                errors.append(f"{name}: aligned {len(words)}/{len(tokens)} tokens")
                continue
            if mean_conf < min_confidence:
                errors.append(f"{name} mean_confidence={mean_conf:.3f} < {min_confidence}")
                continue
            backend = f"{name} (mean_conf={mean_conf:.3f})"
            break

        words = _enforce_monotonic(words, duration)
        pauses = _extract_pauses(words, duration, PAUSE_MIN_SEC)
        return AlignmentResult(
            sample_id=sample_id,
            audio_path=rel.replace("\\", "/"),
            words=words,
            pauses=pauses,
            transcript_path=transcript_path,
            duration_sec=round(duration, 4),
            backend=backend,
            device=str(self.device),
        )


def align_audio(
    audio_path: str | Path,
    transcript: str,
    sample_id: str,
    **kwargs,
) -> AlignmentResult:
    """Module-level convenience wrapper around a process-global aligner."""
    global _ALIGNER
    if _ALIGNER is None:
        _ALIGNER = ForcedAligner()
    return _ALIGNER.align_file(audio_path, transcript, sample_id, **kwargs)


_ALIGNER: ForcedAligner | None = None
