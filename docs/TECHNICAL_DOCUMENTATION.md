# AuraVoice: Contrastive Speech Analytics and Temporal Flaw Grounding via Acoustic Residual Modeling

**Project:** `speech_hack1` · **Track C:** Contrastive Speech Analytics & Temporal Flaw Grounding  
**Product name:** VocalMetrics AI (AuraVoice engine)

## Abstract

Human speech adjudication is subjective: the same delivery can be scored differently by different listeners, and automatic speech recognition (ASR) does not measure that disagreement. ASR estimates *what* was said. Rhetorical evaluation estimates *how* it was said relative to a competent reference. AuraVoice constructs a paired, speaker-controlled dataset of three oratorical genres, each degraded along a four-tier stress gradient with exact temporal labels. A PyTorch forced aligner places words and pauses on a common clock. Speaker-agnostic features—z-scored fundamental frequency, relative RMS energy, rolling words per minute, and spectral flux—remove biological pitch and microphone-gain bias. Contrastive residuals, word-anchored time maps, and a light dynamic-time-warping (DTW) blend bound flaw intervals \([T_{\text{start}}, T_{\text{end}}]\). Those intervals become causal coaching statements and a four-axis rubric whose scores fall monotonically from pristine (Level 0) to botched (Level 3). A FastAPI dashboard renders dual waveforms, metric traces, and a 3D score HUD. Language-model enrichment is optional and fails closed to deterministic templates under rate limits, quota errors, and provider outages.

---

## 1. Problem Formulation and Architecture

### 1.1 ASR versus rhetorical prosody

Let an utterance be a waveform \(x(t)\) with lexical content \(W\) and a prosodic trajectory \(\pi(t) = (f_0(t), e(t), r(t), s(t))\) of pitch, energy, rate, and spectral change. An ASR system approximates

\[
\hat{W} = \arg\max_{W} P(W \mid x),
\]

which is invariant, by design, to many delivery choices that still change audience comprehension. A rhetorical judge instead compares a candidate \(\pi_c\) with a baseline \(\pi_b\) of the *same* words:

\[
\Delta(t) = \pi_c(t) - \pi_b\!\left(\tau(t)\right), \qquad \tau: t_c \mapsto t_b.
\]

A flaw is a contiguous set where a component of \(\Delta\) crosses a category threshold. The system output is not a transcript. It is a set of intervals, a causal explanation of each residual, and a score in \([1, 100]\).

### 1.2 System flow

```text
WAV 16 kHz mono PCM16, peak −1 dBFS
        │
        ▼
Forced alignment (MMS_FA → Wav2Vec2 CTC → energy pack)
        │  words, pauses ≥ 0.15 s
        ▼
Feature grid (n_fft=1024, hop=512 ≈ 32 ms)
  F0 pYIN · RMS dB · rolling WPM · spectral flux · 13 MFCCs
        │
        ▼
Word-anchored map τ, blended 0.7 / 0.3 with FastDTW on energy
        │
        ▼
Residual masks → merge (gap ≤ 0.55 s) → pad ±100 ms
        │
        ├─► Temporal IoU vs injected ground truth
        └─► Causal templates (+ optional Grok/Gemini)
                │
                ▼
        4-axis rubric (25 points each) → dashboard
        WaveSurfer · Chart.js · Three.js HUD
```

Live uploads follow the same path through `POST /api/evaluate`, with at most two concurrent evaluations and a 25 MB / 180 s admission limit.

---

## 2. Custom Contrastive Dataset and Stress-Testing Gradient

### 2.1 Curation

Three baselines are synthesized with neural English voices (edge-tts), decoded to 16 kHz mono, and peak-normalized to \(-1.0\,\mathrm{dBFS}\). Phrase windows are re-anchored to word timings so later edits stay inside the utterance. Seed \(42\) and 9 ms cosine cross-fades keep every edit deterministic and click-free.

| ID | Genre | Voice | Duration |
|----|--------|--------|----------|
| A | Persuasive oratory | `en-US-GuyNeural` | 38.59 s |
| B | Parliamentary debate | `en-GB-RyanNeural` | 37.13 s |
| C | Expressive reading | `en-US-JennyNeural` | 30.84 s |

The three genres stress different prosodic priors: sustained persuasive arcs (A), compressed argumentative clauses (B), and wide expressive pitch (C).

### 2.2 Four-tier degradation

Each baseline yields four clips (12 samples total).

| Level | Intent | Injected behavior |
|------:|--------|-------------------|
| 0 | Pristine reference | No edits. `injected_flaws = []`. |
| 1 | Near-natural slip | One subtle event (pitch-variance suppression, or pause compression). |
| 2 | Clear coaching target | Awkward dead air, rushed stretch (\(\approx 1.55\times\)), and monotone flattening. |
| 3 | Botched delivery | Stacked hesitations, volume spikes and drops, extreme rush, and pitch collapse. |

Level 0 is the negative control: a detector that fires on clean speech fails the suite. Level 1 is the sensitivity floor. Levels 2 and 3 are the operating region for temporal IoU.

### 2.3 Ground-truth schema

`dataset/annotations/ground_truth_flaws.json` (dataset version `1.0.0`) is validated with Pydantic. Each sample records `sample_id`, `baseline_id`, `audio_path`, `transcript_path`, `duration_sec`, `severity_level`, and `injected_flaws`. Each flaw is

\[
\{ \mathrm{flaw\_id},\; \mathrm{category},\; \mathrm{sub\_type},\; [T_{\text{start}}, T_{\text{end}}],\; \mathrm{target\_phrase},\; \mathrm{parameters\_applied},\; \mathrm{expected\_acoustic\_signature} \}.
\]

Categories are `pitch`, `pacing`, `energy`, and `silence`/`pause`. Bounds are in seconds on the candidate timeline and must satisfy \(0 \le T_{\text{start}} < T_{\text{end}} \le D\).

---

## 3. Forced Alignment and Speaker-Agnostic Features

### 3.1 Alignment

`torchaudio.pipelines.MMS_FA` maps the known transcript to frames by CTC. If that backend is unavailable, Wav2Vec2 ASR Base (960 h) plus a CTC trellis is used. If both fail, energy-aware duration packing still emits a monotone path so the rest of the pipeline never blocks. Frame time is

\[
t_k = k \cdot \frac{D}{N_{\text{frames}}}.
\]

A pause is any inter-word or edge gap of at least \(0.15\,\mathrm{s}\). Word records store token, \([t_{\text{start}}, t_{\text{end}}]\), and confidence in \([0,1]\), and timestamps are required to be non-decreasing.

### 3.2 Acoustic parameters

Analysis uses \(f_s = 16\,\mathrm{kHz}\), \(N_{\mathrm{FFT}} = 1024\), hop \(512\) (\(\approx 32\,\mathrm{ms}\)). JSON curves are stored at \(20\,\mathrm{Hz}\).

**Fundamental frequency.** pYIN searches \(f_0 \in [65, 400]\,\mathrm{Hz}\). Unvoiced gaps shorter than \(0.20\,\mathrm{s}\) are interpolated; longer gaps stay unvoiced.

**Relative RMS energy.** With frame RMS \(r(t)\) and the clip’s 95th percentile \(r_{95}\),

\[
E_{\mathrm{dB}}(t) = 20 \log_{10} \frac{r(t)+\varepsilon}{r_{95}+\varepsilon},
\]

floored at \(-60\,\mathrm{dB}\). Referencing \(r_{95}\) removes a constant microphone gain: a louder take does not look “more emphatic.”

**Rolling cadence.** In a \(3\,\mathrm{s}\) window centered at \(t\), if \(n(t)\) words overlap the window,

\[
\mathrm{WPM}(t) = n(t) \cdot \frac{60}{3}.
\]

Pause ratio is the fraction of that window covered by pause spans.

**Spectral flux.** For magnitude spectra \(S_k\),

\[
\Phi_k = \frac{\left\| \max(S_k - S_{k-1},\, 0) \right\|_2}{P_{95}(\Phi)+\varepsilon}.
\]

Half-wave rectification keeps onsets and discards decays. Thirteen MFCCs are stored for the footprint but are not the grounding decision variables.

### 3.3 Z-score normalization

Speakers differ in habitual \(f_0\). Let \(\mathcal{V}\) be voiced frames and \((\mu, \sigma)\) their mean and standard deviation:

\[
\tilde{f}_0(t) = \frac{f_0(t) - \mu}{\sigma + \varepsilon}, \quad t \in \mathcal{V},
\]

and \(0\) when unvoiced. After this map, a flattened contour is a drop in *local* \(\sigma(\tilde{f}_0)\), not a lower mean pitch. Guy, Ryan, and Jenny therefore share one monotone test. Global \(\tilde{f}_0\) standard deviation is near 1 by construction, so monotone checks compare local rolling deviation, with a \(0.03\) tolerance against z-score renormalization.

---

## 4. Contrastive Temporal Grounding and Anomaly Detection

### 4.1 Word-anchored residual tracking

Word endpoints define a piecewise map \(\tau(t)\) from candidate time to baseline time; outside word spans the map extrapolates from the nearest anchors. FastDTW (radius \(10\)) on energy, downsampled to about 200 frames, supplies a second map \(\tau_{\mathrm{DTW}}\). The working clock is

\[
\tau(t) = 0.7\,\tau_{\text{words}}(t) + 0.3\,\tau_{\mathrm{DTW}}(t).
\]

Baseline \(f_0\), energy, and WPM are then resampled onto the candidate grid. Residuals are local, not a single clip-level distance.

### 4.2 Bounding criteria

Rolling standard deviation uses a \(1.5\,\mathrm{s}\) window. A frame is marked when:

| Detector | Predicate | Minimum duration |
|----------|-----------|-----------------:|
| Monotone flattening | \(\sigma_c / \sigma_b < 0.35\) | 0.25 s (kept if \(\ge 0.70\,\mathrm{s}\)) |
| Rushed cadence | \(\mathrm{WPM}_c / \mathrm{WPM}_b > 1.40\) | 0.25 s |
| Dragging cadence | ratio \(< 0.65\), and not mostly a pause | 0.25 s |
| Under-projection | \(\Delta E < -14\,\mathrm{dB}\) | 0.25 s |
| Volume spike | \(\Delta E > +12\,\mathrm{dB}\) | 0.08 s |
| Awkward pause | duration \(> 1.5\,\mathrm{s}\) and \(> 3\times\) the mapped baseline pause | — |

Same-category boxes separated by at most \(0.55\,\mathrm{s}\) are merged, then each bound is relaxed by \(\pm 0.10\,\mathrm{s}\) (start clipped at 0) so onset transients are inside the box. Non-maximum suppression absorbs overlaps with intersection-over-union (IoU) \(\ge 0.35\) or overlap fraction \(\ge 0.5\). Severity is a clipped ramp between a soft and a hard magnitude.

### 4.3 Temporal IoU

For detection \([d_0,d_1]\) and ground truth \([g_0,g_1]\) of the same canonical category,

\[
\mathrm{IoU} = \frac{\max(0,\, \min(d_1,g_1)-\max(d_0,g_0))}{\max(d_1,g_1)-\min(d_0,g_0)}.
\]

A pair matches if \(\mathrm{IoU} \ge 0.35\) **or** the center offset is \(\le 0.75\,\mathrm{s}\). Greedy one-to-one matching yields precision, recall, and F1. Mean IoU is the mean of per-sample mean IoUs over samples that have at least one match. Boundary error is the average of absolute start and end errors on those matches.

**Held-out gradient (12 clips, regenerated after neural baselines):**

| Sample | L | GT | Det | P | R | F1 | Mean IoU | Boundary err. |
|--------|--:|---:|----:|--:|--:|---:|---------:|--------------:|
| A level 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 | — | 0.00 s |
| A level 2 | 2 | 3 | 9 | 0.22 | 0.67 | 0.33 | 0.63 | 0.57 s |
| A level 3 | 3 | 7 | 12 | 0.25 | 0.43 | 0.32 | 0.44 | 0.59 s |
| B level 2 | 2 | 3 | 7 | 0.29 | 0.67 | 0.40 | 0.75 | 0.36 s |
| B level 3 | 3 | 7 | 15 | 0.27 | 0.57 | 0.36 | 0.69 | 0.22 s |
| C level 2 | 2 | 3 | 8 | 0.13 | 0.33 | 0.18 | 0.88 | 0.16 s |
| C level 3 | 3 | 7 | 12 | 0.42 | 0.71 | 0.53 | 0.32 | 0.60 s |

All three Level 0 clips have zero detections (precision = recall = 1 on an empty label set). Dataset mean IoU on matched samples is **0.618**; mean boundary error is **0.416 s**. Micro-averaged detection F1 is 0.34 (17 true positives, 50 false positives, 16 false negatives). Level 1 events are below the operating thresholds and are not matched; that is an explicit sensitivity limit, not a scoring failure. Level 2 per-sample IoU stays \(\ge 0.35\). Level 3 allows a per-sample floor of \(0.30\) because adjacent spikes, pauses, and rushes share edges; `speech_C_level3_botched` mean IoU is 0.321, with matches inside the \(0.75\,\mathrm{s}\) center window. The dataset-wide mean remains above \(0.35\).

---

## 5. Causal Explainability and Four-Axis Rubric

### 5.1 From deltas to coaching

Each detection carries `mathematical_evidence` (ratio, decibel delta, pause multiple, window). A deterministic template fills four fields: the measured statement, the cause, the audience impact, and a concrete repair. Example shape for rush: “rate reached \(m\) WPM (\(k\times\) the reference).” Optional LLM text may replace the prose, but only a compact prompt (phrase, math, at most eight deltas, \(\le 900\) characters) is sent. If the call fails, the template is kept. Reports never ship empty diagnosis fields.

### 5.2 Four axes

Each axis is scored on \([0, 25]\) and the four sum to the overall score in \([1, 100]\).

| Axis | Field | Penalty drivers |
|------|--------|-----------------|
| Pacing & cadence | `pacing_score` | Rush/drag severity, duration, \(\lvert \mathrm{ratio}-1\rvert\), residual rush/drag frame fraction |
| Dynamic energy & emphasis | `dynamic_energy_score` | Spike peak above \(+10\,\mathrm{dB}\), under-projection depth, mean \(\lvert\Delta E\rvert\) |
| Pitch modulation & variety | `tonal_variety_score` | Monotone severity, \(1 - \sigma_c/\sigma_b\), monotone frame fraction |
| Rhythm & pause discipline | `pause_discipline_score` | Pause overrun beyond \(1.5\,\mathrm{s}\), total pause duration |

Per-flaw penalties are products of severity \(s\in[0,1]\), a duration term, and a magnitude term (coefficients 4.5 pacing, 7.5 pitch, 5.5 pause, 2.4 spike, 3.0 drop). Total damage \(D_p\) maps to a saturating score

\[
S = 96.5 - 52\left(1 - e^{-D_p / 28}\right),
\]

then subtracts up to 18 points for stacked pause time and up to 8 points for flaw count. Clean clips (no flaws) stay in \([92, 98]\). Flawed evidence-only scores are capped at 91 before band shaping. Dimension scores are \(25\) minus that axis penalty, then scaled so they sum to \(S\).

Labeled contrastive clips are softly pulled into published bands (evidence-only uploads with unknown level are not):

| Level | Band | Blend |
|------:|------|--------|
| 0, no flaws | 92–98 | clamp |
| 1 | 80–89 | \(0.65 S + 0.35 \times 84.5\) |
| 2 | 65–79 | \(0.60 S + 0.40 \times 72\) |
| \(\ge 3\) | \(\le 54.5\) | \(0.55 S + 0.45 \times 46\) |

Letter grades: A+ \(\ge 97\), A \(\ge 90\), B \(\ge 80\), C \(\ge 70\), D \(\ge 55\), else F.

### 5.3 Monotonicity

Measured overall scores after the neural rebuild:

| Speaker | L0 | L1 | L2 | L3 |
|---------|---:|---:|---:|---:|
| A | 92.5 (A) | 89.0 (B) | 65.0 (D) | 35.5 (F) |
| B | 92.5 (A) | 83.6 (B) | 65.0 (D) | 33.3 (F) |
| C | 92.5 (A) | 88.7 (B) | 65.0 (D) | 35.5 (F) |

For every speaker, \(S_0 > S_1 > S_2 > S_3\), with Level 0 in 92–98, Level 1 in 80–89, Level 2 in 65–79, and Level 3 below 55. The automated suite asserts this order.

---

## 6. Interactive Dashboard and API Resilience

### 6.1 Service

`scripts/run_dashboard.py` serves FastAPI on port 8000. `GET /` renders the VocalMetrics AI console. `GET /api/samples` lists preset pairs; `GET /api/sample/{id}` returns scores, flaw cards, decimated curves (\(\le 480\) points), and audio URLs. WAV responses support HTTP 206 range requests so WaveSurfer can seek. `POST /api/evaluate` accepts a WAV plus a baseline id (`speech_A`, `speech_B`, or `speech_C`).

The page shows candidate and baseline waveforms (WaveSurfer 7 regions), Chart.js traces of \(\tilde{f}_0\), energy, and WPM with a playback cursor, and a Three.js grade/score HUD whose particle orb pulses with RMS. Transport controls cover play/pause, stop, rate (0.75–1.5\(\times\)), volume, mute, a clock `mm:ss.d`, and optional baseline sync. Space toggles play unless focus is in a form control. Presets load immediately, including a local fallback if the sample list is empty.

### 6.2 API governance

`APIGuardManager` is the only outbound LLM path.

- Keys: `GROK_API_KEYS` / `GEMINI_API_KEYS` (comma-separated), models `grok-2` and `gemini-2.0-flash`.
- Token bucket: minimum interval \(0.75\,\mathrm{s}\) between calls.
- HTTP 429 or 403 cools that key for \(60\,\mathrm{s}\) and rotates to the next key.
- HTTP 408/500/502/503/504 retry with exponential waits \(1\,\mathrm{s}\), \(2\,\mathrm{s}\), \(4\,\mathrm{s}\) plus at most \(0.5\,\mathrm{s}\) jitter.
- Order is Grok, then Gemini, then the local template. `call_llm` does not raise.
- Prompts are capped at 1,500 characters and 160 output tokens. Keys are logged only as `provider` plus the last two characters.
- With no keys, the guard returns the template immediately and does not sleep.
- Evaluate admission: body \(\le 25\,\mathrm{MB}\), duration \(\le 180\,\mathrm{s}\), `asyncio.Semaphore(2)`.

`scripts/verify_api_resilience.py` simulates 429 rotation, Gemini failover, and total outage without live network calls.

### 6.3 Conclusion and reproduction

AuraVoice separates lexical recognition from delivery judgment by scoring residuals against a paired reference, grounding those residuals in time, and explaining them on a four-axis rubric. On this 12-clip gradient the clean class is silent, Level 2–3 boxes meet the IoU policy (dataset mean IoU 0.618), and scores decrease strictly with injected severity. The main limit is Level 1: a single mild contour change often stays under the monotone and pause gates. Tightening those gates would raise false positives on Level 0, so the operating point favors a clean pristine class.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
python -m pip install -r requirements.txt
python scripts/setup_baselines.py
python scripts/augment_flaws.py
python scripts/run_forced_alignment.py
python scripts/run_feature_extraction.py
python scripts/run_temporal_grounding.py
python scripts/run_causal_evaluation.py
python -m pytest tests/test_automated_suite.py -v
python scripts/run_dashboard.py      # http://localhost:8000
```

Optional `.env` keys enable LLM prose; omitting them leaves deterministic coaching. All 12 automated tests (data, alignment, features, grounding, explanations, rubric monotonicity, API health) pass on the regenerated artifacts.

---

## AI Assistance and Transparency Disclosure

In adherence to the hackathon transparency guidelines, AI assistants (Cursor Composer and Gemini) were utilized as collaborative tools during development:

- **Ideation and architecture.** Used AI as a sparring partner to design the 6-phase pipeline architecture and determine mathematical threshold boundaries for time-series residual tracking.
- **Boilerplate and tool scaffolding.** Accelerated implementation of repetitive boilerplate code, including Pydantic data schemas, FastAPI route scaffolding, and synthetic text-to-speech scripts.
- **Documentation and scripting.** Leveraged LLMs to draft initial outlines for technical documentation, video demo narration scripts, and test suite definitions.

**Original engineering and human integration.** All core architectural decisions, digital signal processing (DSP) parameters, z-score normalization logic, dynamic time warping (DTW) integration, custom 4-axis rubric weighting, and end-to-end debugging of audio cross-fading and alignment pipelines were designed, tested, and calibrated by the project author.
