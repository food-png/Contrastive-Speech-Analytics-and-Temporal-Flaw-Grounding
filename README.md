# Track C — Contrastive Speech Analytics & Temporal Flaw Grounding

Phase 1: **Custom Contrastive Dataset Engineering & Stress Testing**
Phase 2: **Forced Alignment & Temporal Token-Level Synchronization**
Phase 3: **Advanced Feature Extraction & Speaker-Agnostic Acoustic Footprint**
Phase 4: **Contrastive Temporal Grounding & Anomaly Detection**
Phase 5: **Causal Explainability & Rubric Scoring Engine**
Phase 6: **Interactive Frontend & Dashboard Visualization**

Reproducible pipeline that synthesizes pristine baseline oratory, then injects a
controlled gradient of delivery flaws with exact temporal ground-truth labels.

## Layout

```
speech_hack1/
├── requirements.txt
├── README.md
├── dataset/
│   ├── raw_baseline/                # Reference WAVs + transcripts + meta
│   ├── processed_baseline/          # 16 kHz mono PCM16, peak −1 dBFS
│   ├── flawed_spectrum/             # Level 0–3 contrastive variations
│   ├── annotations/
│   │   └── ground_truth_flaws.json  # Pydantic-validated flaw regions
│   ├── alignments/                  # Word/pause timestamps per sample
│   └── features/
│       ├── npz/                     # Full-rate feature matrices
│       ├── json/                    # 20 Hz curves for plotting
│       └── qc_plots/                # Contrast PNGs
│   ├── grounding/                   # Detected intervals, eval, residuals, plots
│   └── evaluations/                 # Causal reports + rubric scorecards
├── scripts/
│   ├── setup_baselines.py           # Pristine speech synthesis
│   ├── augment_flaws.py             # Deterministic flaw injection (seed=42)
│   ├── verify_dataset.py            # Schema + audio QA
│   ├── run_forced_alignment.py      # MMS_FA / CTC word alignment
│   ├── verify_alignments.py         # Monotonic timestamp QA
│   ├── run_feature_extraction.py    # Speaker-agnostic feature curves
│   ├── verify_features.py           # Finite-array QA + contrast plot
│   ├── run_temporal_grounding.py    # Residual anomaly detection + IoU eval
│   ├── plot_temporal_grounding.py   # Timeline GT vs detection PNGs
│   ├── run_causal_evaluation.py     # Causal diagnoses + 1–100 rubric
│   └── verify_evaluations.py        # Monotone score / field QA
└── utils/
    ├── __init__.py
    ├── audio_ops.py                 # DSP: cross-fade, stretch, pitch, energy
    ├── aligner.py                   # Forced alignment + pause extraction
    ├── feature_extractor.py         # F0, RMS, WPM, flux, MFCCs
    ├── temporal_grounding.py        # Contrastive residual grounding
    ├── metrics.py                   # Temporal IoU / P / R / F1
    ├── causal_explainer.py          # Deterministic (+ optional LLM) diagnoses
    └── rubric_scorer.py             # 4-dimension delivery rubric
```

## Audio contract

| Property     | Value              |
|--------------|--------------------|
| Sample rate  | 16 000 Hz          |
| Channels     | 1 (mono)           |
| Bit depth    | 16-bit PCM WAV     |
| Peak level   | −1.0 dBFS          |
| Cross-fades  | 8–10 ms cosine     |
| RNG seed     | 42                 |

## Quick start

```bash
# 1. Create & activate virtual environment (already present as .venv)
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

# 2. Install dependencies into the venv
python -m pip install -r requirements.txt

# 3. Synthesize 3 pristine baselines (A/B/C, ~45–60 s each)
python scripts/setup_baselines.py

# Optional: prefer Edge-TTS when installed + networked
# python scripts/setup_baselines.py --edge-tts

# 4. Inject Level 0–3 flaw spectrum + write JSON annotations
python scripts/augment_flaws.py

# 5. Validate schema, paths, format, and temporal bounds
python scripts/verify_dataset.py

# 6. Forced-align every baseline + flawed clip (writes dataset/alignments/)
python scripts/run_forced_alignment.py

# 7. Validate word/pause timestamps
python scripts/verify_alignments.py

# 8. Extract speaker-agnostic feature curves
python scripts/run_feature_extraction.py

# 9. Validate feature packs + write the speech_A contrast PNG
python scripts/verify_features.py

# 10. Contrastive temporal grounding + IoU benchmark vs Phase-1 GT
python scripts/run_temporal_grounding.py

# 11. Timeline plots (GT green vs detected red boxes)
python scripts/plot_temporal_grounding.py

# 12. Causal diagnoses + 4-dimension rubric scorecards
python scripts/run_causal_evaluation.py

# Optional LLM enrichment: copy `.env.example` to `.env` and set GEMINI_API_KEY
# python scripts/run_causal_evaluation.py --llm

# 13. Verify monotone level scores + non-empty diagnosis fields
python scripts/verify_evaluations.py

# 14. Interactive dashboard (http://localhost:8000)
python scripts/run_dashboard.py
```

## Phase 2 — Forced alignment

`utils/aligner.py` maps each transcript word to `[start_sec, end_sec]` using
`torchaudio.pipelines.MMS_FA` (fallback: `WAV2VEC2_ASR_BASE_960H` +
`torchaudio.functional.forced_align`). Gaps ≥ 0.15 s become explicit `[PAUSE]`
spans (including leading/trailing silence).

If the acoustic model cannot produce a confident path (OOM, empty tokens, or
very low CTC confidence — expected for purely algorithmic formant audio), the
aligner packs words into voiced-energy regions so every sample still emits
valid, monotonic timestamps.

## Phase 3 — Acoustic footprint

`utils/feature_extractor.py` builds a synchronized matrix at 16 kHz / `n_fft=1024` /
`hop=512` (~31.25 Hz):

- **Pitch**: `librosa.pyin` (`fmin=65`, `fmax=400`), short-gap interpolation (<200 ms),
  voiced z-score \(F_{0,\mathrm{norm}}\).
- **Energy**: RMS → dB relative to the 95th percentile, clipped to \([-60, 0]\) dB.
- **Pacing**: 3 s rolling WPM and pause ratio fused from Phase 2 alignments.
- **Timbre**: spectral centroid, spectral flux, 13 MFCCs (per-file z-score).

Binary dumps live in `dataset/features/npz/`; 20 Hz JSON curves in
`dataset/features/json/` for lightweight plotting.

## Phase 4 — Temporal grounding

`utils/temporal_grounding.py` warps each candidate onto its baseline via
word-anchored interpolation (+ optional FastDTW energy refine), then flags:

- **Pitch**: rolling \(\sigma_{F_0}\) collapse (monotone)
- **Pacing**: WPM ratio ≥ 1.40 (rush) or ≤ 0.65 (drag)
- **Energy**: under-projection (≤ −14 dB) and spikes (≥ +12 dB)
- **Pause**: candidate pauses > 1.5 s that are > 3× baseline (or mid-speech)

Detections are scored against `ground_truth_flaws.json` with temporal IoU ≥ 0.35
(or center offset ≤ 0.75 s). Outputs: `dataset/grounding/*_detected.json`,
`evaluation_summary.json`, and `plots/*_grounding.png`.

## Phase 5 — Causal explainability & rubric scoring

`utils/causal_explainer.py` maps each detection to math / cause / impact / fix
templates (optional Gemini enrichment via `GEMINI_API_KEY`, with guaranteed
deterministic fallback). `utils/rubric_scorer.py` emits a 4×25-point scorecard:

| Dimension | Max | Driven by |
|-----------|-----|-----------|
| Pacing | 25 | Rush / drag residuals |
| Dynamic energy | 25 | Volume spikes / under-projection |
| Tonal variety | 25 | Monotone F0 collapse |
| Pause discipline | 25 | Awkward hesitations > 1.5 s |

Expected contrastive bands: L0 92–98, L1 80–89, L2 65–79, L3 < 55.
Reports land in `dataset/evaluations/<sample_id>_report.json`.

## Phase 6 — Dashboard

`python scripts/run_dashboard.py` serves a FastAPI app on port 8000:

- `GET /` — single-page HUD (WaveSurfer, Chart.js, Three.js)
- `GET /api/samples` — preset scorecards
- `GET /api/sample/{sample_id}` — audio URLs, feature curves, causal flaws
- `POST /api/evaluate` — upload a WAV against a baseline id and run align → features → grounding → rubric

WAV files are mounted at `/dataset` and support HTTP range requests for seeking.

## Baseline styles

| ID | Style                         | Prosody signature                                      |
|----|-------------------------------|--------------------------------------------------------|
| A  | Persuasive / Oratory          | High dynamic contrast, intentional pausing             |
| B  | Competitive Parliamentary     | Fast assertive cadence, structured emphasis            |
| C  | Interpretive / Expressive     | Wide pitch variance, emotional arcs                    |

Baselines are generated offline via algorithmic formant synthesis so the
pipeline runs with zero microphone capture. Pass `--edge-tts` to prefer
Microsoft Edge neural TTS when the optional `edge-tts` package is available.

## Flaw gradient

| Level | Name     | Injected behaviours                                                                 |
|-------|----------|-------------------------------------------------------------------------------------|
| 0     | Baseline | Unmodified reference copy                                                           |
| 1     | Subtle   | Pause compression **or** mild F0 flattening on one clause                           |
| 2     | Moderate | 1.55× rush + ≥70% F0 suppression + 2.0–2.5 s awkward silence                        |
| 3     | Botched  | +12 dB volume spikes + multiple >2 s hesitations + >1.8× hyperspeed pitch collapse |

Silence insertions and time-stretches update a `TimestampTracker` so every
subsequent flaw bound is shifted to the post-edit timeline.

## Annotation schema

`dataset/annotations/ground_truth_flaws.json` validates with Pydantic models in
`scripts/verify_dataset.py`:

```json
{
  "dataset_version": "1.0.0",
  "generated_at": "ISO-8601 Timestamp",
  "samples": [
    {
      "sample_id": "speech_A_level2_pacing_monotone",
      "baseline_id": "speech_A_baseline",
      "audio_path": "dataset/flawed_spectrum/speech_A_level2_pacing_monotone.wav",
      "transcript_path": "dataset/raw_baseline/speech_A.txt",
      "duration_sec": 52.4,
      "severity_level": 2,
      "injected_flaws": [
        {
          "flaw_id": "flaw_01",
          "category": "pacing",
          "sub_type": "rushed_delivery",
          "start_sec": 14.25,
          "end_sec": 18.60,
          "target_phrase": "to follow your heart and intuition",
          "parameters_applied": { "time_stretch_factor": 1.55 },
          "expected_acoustic_signature": "Sudden drop in syllable duration..."
        }
      ]
    }
  ]
}
```

## Reproducibility

- Global seed `42` in `augment_flaws.py`
- Per-level / per-sample RNG streams derived from that seed
- Fixed cross-fade length (`DEFAULT_CROSSFADE_MS = 9.0`)
- Deterministic algorithmic synthesizer (seeded NumPy generator)

## AI assistance and transparency

In adherence to the hackathon transparency guidelines, AI assistants (Cursor Composer and Gemini) were utilized as collaborative tools during development:

- **Ideation and architecture.** Used AI as a sparring partner to design the 6-phase pipeline architecture and determine mathematical threshold boundaries for time-series residual tracking.
- **Boilerplate and tool scaffolding.** Accelerated implementation of repetitive boilerplate code, including Pydantic data schemas, FastAPI route scaffolding, and synthetic text-to-speech scripts.
- **Documentation and scripting.** Leveraged LLMs to draft initial outlines for technical documentation, video demo narration scripts, and test suite definitions.

**Original engineering and human integration.** All core architectural decisions, digital signal processing (DSP) parameters, z-score normalization logic, dynamic time warping (DTW) integration, custom 4-axis rubric weighting, and end-to-end debugging of audio cross-fading and alignment pipelines were designed, tested, and calibrated by the project author.

## License note

Synthetic audio and paraphrase-style transcripts are generated for research
benchmarking. Replace baselines with licensed recordings when moving beyond
offline stress tests.
