/* Track C dashboard — WaveSurfer, Chart.js, Three.js HUD. */
(function () {
  "use strict";

  const state = {
    payload: null,
    candidate: null,
    baseline: null,
    regions: null,
    pendingFlaws: [],
    chart: null,
    playhead: 0,
    font: null,
    scene: null,
    renderer: null,
    camera: null,
    gradeMesh: null,
    scoreMesh: null,
    orb: null,
    orbBase: null,
    raf: 0,
  };

  const DIMS = [
    ["pacing_score", "Pacing & cadence", "bg-cyan-400"],
    ["dynamic_energy_score", "Dynamic energy & emphasis", "bg-amber-400"],
    ["tonal_variety_score", "Pitch modulation & vocal variety", "bg-emerald-400"],
    ["pause_discipline_score", "Rhythm & pause discipline", "bg-rose-400"],
  ];

  const $ = (id) => document.getElementById(id);

  function esc(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function fmtTime(sec) {
    const s = Math.max(0, Number(sec) || 0);
    const m = Math.floor(s / 60);
    const r = Math.floor(s % 60);
    return String(m).padStart(2, "0") + ":" + String(r).padStart(2, "0");
  }

  function fmtPrecise(sec) {
    const s = Math.max(0, Number(sec) || 0);
    const m = Math.floor(s / 60);
    const rem = s - m * 60;
    const whole = Math.floor(rem);
    const tenth = Math.min(9, Math.floor((rem - whole) * 10 + 1e-6));
    return String(m).padStart(2, "0") + ":" + String(whole).padStart(2, "0") + "." + tenth;
  }

  function syncEnabled() {
    return Boolean($("sync-baseline") && $("sync-baseline").checked);
  }

  function setTransportLabel(button, playing) {
    if (!button) return;
    button.textContent = playing ? "⏸ Pause" : "▶ Play";
    button.setAttribute("aria-pressed", playing ? "true" : "false");
  }

  function renderClock(current) {
    const duration = state.candidate && state.candidate.getDuration ? state.candidate.getDuration() || 0 : 0;
    $("clock").textContent = fmtPrecise(current) + " / " + fmtPrecise(duration);
  }

  function applyLevels() {
    if (state.candidate) {
      const volume = Number($("volume").value) / 100;
      const muted = $("btn-mute").getAttribute("aria-pressed") === "true";
      state.candidate.setVolume(volume);
      state.candidate.setMuted(muted || volume === 0);
    }
    if (state.baseline) {
      const volume = Number($("baseline-volume").value) / 100;
      const muted = $("btn-baseline-mute").getAttribute("aria-pressed") === "true";
      const audible = syncEnabled() || state.baseline.isPlaying();
      state.baseline.setVolume(volume);
      state.baseline.setMuted(muted || volume === 0 || !audible);
    }
  }

  function applyRate() {
    const rate = Number($("playback-rate").value) || 1;
    if (state.candidate && state.candidate.setPlaybackRate) state.candidate.setPlaybackRate(rate);
    if (state.baseline && syncEnabled() && state.baseline.setPlaybackRate) state.baseline.setPlaybackRate(rate);
  }

  function setStatus(text) {
    $("status").textContent = text;
  }

  function gradeColor(grade) {
    const g = String(grade || "").toUpperCase();
    if (g.startsWith("A")) return 0x34d399;
    if (g.startsWith("B")) return 0x22d3ee;
    if (g.startsWith("C")) return 0xfbbf24;
    if (g.startsWith("D")) return 0xfb923c;
    return 0xfb7185;
  }

  function regionColor(flaw) {
    const sev = Number(flaw.severity_score) || 0;
    const sub = String(flaw.sub_type || "").toLowerCase();
    const cat = String(flaw.category || "").toLowerCase();
    if (sev >= 0.75 || sub.includes("spike") || sub.includes("hyper") || cat === "energy" && sev >= 0.6) {
      return "rgba(244, 63, 94, 0.38)";
    }
    if (sev >= 0.45 || sub.includes("rush") || sub.includes("monotone") || sub.includes("pacing")) {
      return "rgba(251, 191, 36, 0.35)";
    }
    return "rgba(34, 211, 238, 0.32)";
  }

  function deltaBadge(flaw) {
    const raw = flaw.mathematical_evidence_raw || {};
    if (raw.peak_delta_db != null) return "ΔdB " + Number(raw.peak_delta_db).toFixed(1);
    if (raw.median_delta_db != null) return "ΔdB " + Number(raw.median_delta_db).toFixed(1);
    if (raw.pacing_ratio_median != null) return "Speed " + Number(raw.pacing_ratio_median).toFixed(2) + "x";
    if (raw.sigma_ratio_median != null) {
      const pct = (1 - Number(raw.sigma_ratio_median)) * 100;
      return "ΔF0 " + pct.toFixed(0) + "%";
    }
    if (raw.delta_pause_sec != null) return "Pause +" + Number(raw.delta_pause_sec).toFixed(2) + "s";
    if (raw.cand_pause_sec != null) return "Pause " + Number(raw.cand_pause_sec).toFixed(2) + "s";
    return flaw.category || "flaw";
  }

  function energyAt(t) {
    const feat = state.payload && state.payload.features;
    if (!feat || !feat.time_seconds || !feat.time_seconds.length) return 0.35;
    const times = feat.time_seconds;
    const energy = feat.energy_norm;
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < times.length; i++) {
      const d = Math.abs(times[i] - t);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    const db = Number(energy[best]);
    if (!Number.isFinite(db)) return 0.35;
    return Math.min(1, Math.max(0, (db + 60) / 60));
  }

  function initHud() {
    const canvas = $("hud-canvas");
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 2, 0.1, 100);
    camera.position.set(0, 0.15, 7.2);
    scene.add(new THREE.AmbientLight(0x88aacc, 0.55));
    const key = new THREE.PointLight(0x22d3ee, 1.4, 30);
    key.position.set(4, 3, 6);
    scene.add(key);
    const rim = new THREE.PointLight(0xfb7185, 0.8, 24);
    rim.position.set(-4, -1, 4);
    scene.add(rim);

    const N = 900;
    const positions = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 1.55;
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const orb = new THREE.Points(
      geo,
      new THREE.PointsMaterial({ color: 0x67e8f9, size: 0.045, transparent: true, opacity: 0.85 })
    );
    orb.position.x = 3.15;
    scene.add(orb);

    state.renderer = renderer;
    state.scene = scene;
    state.camera = camera;
    state.orb = orb;
    state.orbBase = positions;

    function resize() {
      const w = canvas.clientWidth || canvas.parentElement.clientWidth;
      const h = canvas.clientHeight || 220;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    }
    resize();
    window.addEventListener("resize", resize);

    const loader = new THREE.FontLoader();
    loader.load(
      "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/fonts/helvetiker_bold.typeface.json",
      (font) => {
        state.font = font;
        setHudText("—", "0.0");
      }
    );

    function tick() {
      const t = performance.now() * 0.001;
      if (state.orb) {
        state.orb.rotation.y = t * 0.35;
        state.orb.rotation.x = Math.sin(t * 0.2) * 0.15;
        const pulse = 0.82 + energyAt(state.playhead) * 0.55;
        state.orb.scale.setScalar(pulse);
      }
      if (state.gradeMesh) state.gradeMesh.rotation.y = Math.sin(t * 0.6) * 0.08;
      if (state.scoreMesh) state.scoreMesh.rotation.y = Math.sin(t * 0.6) * 0.08;
      renderer.render(scene, camera);
      state.raf = requestAnimationFrame(tick);
    }
    tick();
  }

  function disposeMesh(mesh) {
    if (!mesh) return;
    state.scene.remove(mesh);
    if (mesh.geometry) mesh.geometry.dispose();
    if (mesh.material) mesh.material.dispose();
  }

  function makeText(text, size, color, x, y) {
    const geometry = new THREE.TextGeometry(text, {
      font: state.font,
      size,
      height: size * 0.28,
      curveSegments: 6,
      bevelEnabled: true,
      bevelThickness: size * 0.04,
      bevelSize: size * 0.025,
      bevelSegments: 2,
    });
    geometry.computeBoundingBox();
    geometry.center();
    const material = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.45,
      metalness: 0.35,
      roughness: 0.28,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(x, y, 0);
    state.scene.add(mesh);
    return mesh;
  }

  function setHudText(grade, score) {
    if (!state.font || !state.scene) return;
    disposeMesh(state.gradeMesh);
    disposeMesh(state.scoreMesh);
    const color = gradeColor(grade);
    state.gradeMesh = makeText(String(grade || "—"), 1.15, color, -1.35, 0.35);
    state.scoreMesh = makeText(Number(score).toFixed(1), 0.62, 0xe2e8f0, -1.35, -0.95);
  }

  function seriesXY(times, values) {
    const out = [];
    const n = Math.min(times.length, values.length);
    for (let i = 0; i < n; i++) {
      const y = Number(values[i]);
      if (Number.isFinite(Number(times[i])) && Number.isFinite(y)) {
        out.push({ x: Number(times[i]), y });
      }
    }
    return out;
  }

  function renderChart(payload) {
    const ctx = $("metric-chart").getContext("2d");
    if (state.chart) state.chart.destroy();
    const cand = payload.features || {};
    const base = payload.baseline_features || { time_seconds: [], f0_norm: [], energy_norm: [], rolling_wpm: [] };
    const playhead = {
      id: "playhead",
      afterDraw(chart) {
        const xScale = chart.scales.x;
        if (!xScale) return;
        const x = xScale.getPixelForValue(state.playhead);
        const { top, bottom } = chart.chartArea;
        const g = chart.ctx;
        g.save();
        g.beginPath();
        g.moveTo(x, top);
        g.lineTo(x, bottom);
        g.strokeStyle = "rgba(248, 250, 252, 0.85)";
        g.lineWidth = 1.25;
        g.stroke();
        g.restore();
      },
    };
    state.chart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          { label: "F0 z (candidate)", data: seriesXY(cand.time_seconds || [], cand.f0_norm || []), borderColor: "#22d3ee", yAxisID: "y", borderWidth: 1.6, pointRadius: 0, tension: 0.15 },
          { label: "F0 z (baseline)", data: seriesXY(base.time_seconds || [], base.f0_norm || []), borderColor: "#64748b", yAxisID: "y", borderWidth: 1.2, borderDash: [4, 3], pointRadius: 0, tension: 0.15 },
          { label: "Energy dB (candidate)", data: seriesXY(cand.time_seconds || [], cand.energy_norm || []), borderColor: "#fbbf24", yAxisID: "y1", borderWidth: 1.6, pointRadius: 0, tension: 0.15 },
          { label: "Energy dB (baseline)", data: seriesXY(base.time_seconds || [], base.energy_norm || []), borderColor: "#a16207", yAxisID: "y1", borderWidth: 1.1, borderDash: [4, 3], pointRadius: 0, tension: 0.15 },
          { label: "WPM (candidate)", data: seriesXY(cand.time_seconds || [], cand.rolling_wpm || []), borderColor: "#34d399", yAxisID: "y2", borderWidth: 1.6, pointRadius: 0, tension: 0.15 },
          { label: "WPM (baseline)", data: seriesXY(base.time_seconds || [], base.rolling_wpm || []), borderColor: "#065f46", yAxisID: "y2", borderWidth: 1.1, borderDash: [4, 3], pointRadius: 0, tension: 0.15 },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "nearest", intersect: false },
        plugins: {
          legend: { labels: { color: "#cbd5e1", boxWidth: 12, font: { size: 11 } } },
        },
        scales: {
          x: {
            type: "linear",
            title: { display: true, text: "Time (s)", color: "#94a3b8" },
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(51,65,85,0.45)" },
          },
          y: {
            position: "left",
            title: { display: true, text: "F0 z", color: "#67e8f9" },
            ticks: { color: "#67e8f9" },
            grid: { color: "rgba(51,65,85,0.35)" },
          },
          y1: {
            position: "right",
            title: { display: true, text: "dB", color: "#fbbf24" },
            ticks: { color: "#fbbf24" },
            grid: { drawOnChartArea: false },
          },
          y2: {
            position: "right",
            offset: true,
            title: { display: true, text: "WPM", color: "#34d399" },
            ticks: { color: "#34d399" },
            grid: { drawOnChartArea: false },
          },
        },
        onClick(_evt, _els, chart) {
          const xs = chart.scales.x;
          const rect = chart.canvas.getBoundingClientRect();
          const px = (_evt.native && _evt.native.offsetX) || 0;
          const t = xs.getValueForPixel(px);
          if (Number.isFinite(t)) seekTo(t, true);
        },
      },
      plugins: [playhead],
    });
  }

  function renderMeters(scores) {
    const root = $("meters");
    root.innerHTML = DIMS.map(([key, label, color]) => {
      const val = Number((scores || {})[key] || 0);
      const pct = Math.max(0, Math.min(100, (val / 25) * 100));
      return (
        '<div>' +
        '<div class="mb-1 flex justify-between text-xs text-slate-300"><span>' + esc(label) + '</span><span class="font-mono">' + val.toFixed(1) + '/25</span></div>' +
        '<div class="meter-track"><div class="meter-fill ' + color + '" style="width:' + pct.toFixed(1) + '%"></div></div>' +
        "</div>"
      );
    }).join("");
  }

  function highlightFlaw(id) {
    document.querySelectorAll(".flaw-card").forEach((el) => {
      el.classList.toggle("active", el.dataset.flawId === id);
    });
    const card = document.querySelector('.flaw-card[data-flaw-id="' + id + '"]');
    if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderFlaws(flaws) {
    const root = $("flaw-list");
    if (!flaws || !flaws.length) {
      root.innerHTML = '<p class="text-sm text-emerald-300">No material anomalies against the baseline.</p>';
      return;
    }
    root.innerHTML = flaws.map((flaw) => {
      const text = flaw.llm_enrichment || flaw.causal_rationale || "";
      return (
        '<article class="flaw-card rounded-2xl border border-slate-800 bg-slate-900/70 p-4" data-flaw-id="' + esc(flaw.flaw_id) + '">' +
        '<div class="mb-2 flex flex-wrap items-center gap-2">' +
        '<span class="rounded-full bg-slate-800 px-2 py-1 font-mono text-xs text-cyan-200">[' + fmtTime(flaw.start_sec) + " - " + fmtTime(flaw.end_sec) + "]</span>" +
        '<span class="rounded-full bg-slate-800 px-2 py-1 text-xs text-amber-200">' + esc(deltaBadge(flaw)) + "</span>" +
        '<span class="text-xs uppercase tracking-wide text-slate-400">' + esc(flaw.category) + " · " + esc(flaw.sub_type) + "</span>" +
        "</div>" +
        '<p class="text-sm text-slate-200">' + esc(text) + "</p>" +
        '<p class="mt-2 text-xs text-slate-400">' + esc(flaw.math_evidence) + "</p>" +
        '<p class="mt-2 text-sm text-emerald-200/90">' + esc(flaw.actionable_feedback) + "</p>" +
        '<button type="button" data-seek="' + Number(flaw.start_sec) + '" class="seek-btn mt-3 rounded-lg border border-cyan-400/40 px-3 py-1 text-xs text-cyan-200 hover:bg-cyan-400/10">Seek audio</button>' +
        "</article>"
      );
    }).join("");
    root.querySelectorAll(".seek-btn").forEach((btn) => {
      btn.addEventListener("click", () => seekTo(Number(btn.dataset.seek), true));
    });
    root.querySelectorAll(".flaw-card").forEach((card) => {
      card.addEventListener("click", (ev) => {
        if (ev.target.closest(".seek-btn")) return;
        highlightFlaw(card.dataset.flawId);
      });
    });
  }

  function destroyWaves() {
    if (state.candidate) {
      state.candidate.destroy();
      state.candidate = null;
    }
    if (state.baseline) {
      state.baseline.destroy();
      state.baseline = null;
    }
    state.regions = null;
  }

  function regionsPlugin() {
    if (window.WaveSurfer && window.WaveSurfer.Regions) return window.WaveSurfer.Regions.create();
    if (window.RegionsPlugin) return window.RegionsPlugin.create();
    return null;
  }

  function paintRegions() {
    if (!state.regions) return;
    if (typeof state.regions.clearRegions === "function") state.regions.clearRegions();
    (state.pendingFlaws || []).forEach((flaw) => {
      state.regions.addRegion({
        id: flaw.flaw_id,
        start: Number(flaw.start_sec),
        end: Number(flaw.end_sec),
        color: regionColor(flaw),
        drag: false,
        resize: false,
      });
    });
  }

  function mountWaves(payload) {
    state.pendingFlaws = payload.flaws || [];
    const candidateWaveSurfer = state.candidate;
    const baselineWaveSurfer = state.baseline;
    if (candidateWaveSurfer && baselineWaveSurfer) {
      stopWave(candidateWaveSurfer);
      stopWave(baselineWaveSurfer);
      applyLevels();
      applyRate();
      setTransportLabel($("btn-play"), false);
      setTransportLabel($("btn-baseline-play"), false);
      renderClock(0);
      paintRegions();
      candidateWaveSurfer.load(payload.audio_url);
      baselineWaveSurfer.load(payload.baseline_audio_url || payload.audio_url);
      return;
    }

    const regions = regionsPlugin();
    const plugins = regions ? [regions] : [];
    state.regions = regions;
    state.candidate = WaveSurfer.create({
      container: "#wave-candidate",
      waveColor: "#155e75",
      progressColor: "#22d3ee",
      cursorColor: "#f8fafc",
      height: 88,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
      plugins,
    });
    state.baseline = WaveSurfer.create({
      container: "#wave-baseline",
      waveColor: "#064e3b",
      progressColor: "#34d399",
      cursorColor: "#a7f3d0",
      height: 72,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
    });
    applyLevels();
    applyRate();
    setTransportLabel($("btn-play"), false);
    setTransportLabel($("btn-baseline-play"), false);
    renderClock(0);

    if (regions) {
      regions.on("region-clicked", (region, e) => {
        e.stopPropagation();
        highlightFlaw(region.id);
        seekTo(region.start, true);
      });
    }
    state.candidate.load(payload.audio_url);
    state.baseline.load(payload.baseline_audio_url || payload.audio_url);

    let lastSync = 0;
    let lastDraw = 0;
    const onTime = (t) => {
      const current = Number(t);
      state.playhead = Number.isFinite(current) ? current : state.candidate.getCurrentTime();
      renderClock(state.playhead);
      const now = performance.now();
      if (state.chart && now - lastDraw > 50) {
        lastDraw = now;
        state.chart.draw();
      }
      if (now - lastSync > 80 && state.baseline && syncEnabled()) {
        lastSync = now;
        const playing = state.candidate.isPlaying();
        if (playing && !state.baseline.isPlaying()) state.baseline.play();
        if (!playing && state.baseline.isPlaying()) state.baseline.pause();
        state.baseline.setTime(Math.min(state.playhead, state.baseline.getDuration() || state.playhead));
        applyRate();
      }
    };
    state.candidate.on("timeupdate", onTime);
    state.candidate.on("audioprocess", onTime);
    state.candidate.on("seeking", onTime);
    state.candidate.on("interaction", () => onTime(state.candidate.getCurrentTime()));
    state.candidate.on("ready", () => {
      paintRegions();
      renderClock(state.candidate.getCurrentTime());
    });
    state.candidate.on("play", () => {
      setTransportLabel($("btn-play"), true);
      if (syncEnabled() && state.baseline) {
        applyLevels();
        state.baseline.setTime(state.candidate.getCurrentTime());
        state.baseline.play();
      }
    });
    state.candidate.on("pause", () => {
      setTransportLabel($("btn-play"), false);
      if (syncEnabled() && state.baseline && state.baseline.isPlaying()) state.baseline.pause();
    });
    state.candidate.on("finish", () => setTransportLabel($("btn-play"), false));
    state.baseline.on("play", () => {
      setTransportLabel($("btn-baseline-play"), true);
      applyLevels();
    });
    state.baseline.on("pause", () => setTransportLabel($("btn-baseline-play"), false));
    state.baseline.on("finish", () => setTransportLabel($("btn-baseline-play"), false));
  }

  function stopWave(wave) {
    if (!wave) return;
    if (typeof wave.stop === "function") wave.stop();
    else {
      wave.pause();
      wave.setTime(0);
    }
  }

  function seekTo(sec, play) {
    const t = Math.max(0, Number(sec) || 0);
    state.playhead = t;
    if (state.candidate) {
      state.candidate.setTime(t);
      if (play) state.candidate.play();
    }
    if (state.baseline && syncEnabled()) {
      state.baseline.setTime(Math.min(t, state.baseline.getDuration() || t));
      if (play) state.baseline.play();
    }
    renderClock(t);
    if (state.chart) state.chart.draw();
  }

  function applyPayload(payload) {
    state.payload = payload;
    state.playhead = 0;
    $("summary").textContent = payload.executive_summary || "";
    setHudText(payload.letter_grade, payload.overall_score);
    renderMeters(payload.dimension_scores);
    renderFlaws(payload.flaws || []);
    renderChart(payload);
    mountWaves(payload);
    const level = payload.severity_level < 0 ? "live" : "L" + payload.severity_level;
    setStatus(payload.sample_id + " · " + level + " · " + payload.letter_grade + " " + Number(payload.overall_score).toFixed(1));
  }

  async function loadSample(id) {
    setStatus("Loading " + id + "…");
    const res = await fetch("/api/sample/" + encodeURIComponent(id));
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      setStatus(err.detail || "Failed to load sample");
      return;
    }
    applyPayload(await res.json());
  }

  const PRESET_GROUPS = [
    ["A", "Speech A (Steve Jobs Commencement)"],
    ["B", "Speech B (Parliamentary Debate)"],
    ["C", "Speech C (Expressive Reading)"],
  ];

  const FALLBACK_SAMPLES = [
    { sample_id: "speech_A_level0_baseline", level: 0, overall_score: null, letter_grade: "" },
    { sample_id: "speech_A_level1_subtle_pitch_flatten", level: 1, overall_score: null, letter_grade: "" },
    { sample_id: "speech_A_level2_pacing_monotone", level: 2, overall_score: null, letter_grade: "" },
    { sample_id: "speech_A_level3_botched", level: 3, overall_score: null, letter_grade: "" },
  ];

  function optionLabel(sample) {
    const score = sample.overall_score == null ? "—" : Number(sample.overall_score).toFixed(1);
    const grade = sample.letter_grade || "—";
    return "[Level " + sample.level + "] " + sample.sample_id + " — Score: " + score + " (" + grade + ")";
  }

  function fillPresetSelect(samples) {
    const select = $("sample-select");
    const buckets = { A: [], B: [], C: [], other: [] };
    samples.forEach((sample) => {
      const match = String(sample.sample_id || "").match(/^speech_([ABC])/i);
      const key = match ? match[1].toUpperCase() : "other";
      buckets[key].push(sample);
    });
    Object.values(buckets).forEach((list) => list.sort((a, b) => a.level - b.level || String(a.sample_id).localeCompare(String(b.sample_id))));
    const groups = PRESET_GROUPS.map(([key, label]) => {
      const options = buckets[key].map((sample) => {
        return '<option value="' + esc(sample.sample_id) + '">' + esc(optionLabel(sample)) + "</option>";
      }).join("");
      return '<optgroup label="' + esc(label) + '">' + options + "</optgroup>";
    });
    if (buckets.other.length) {
      const options = buckets.other.map((sample) => {
        return '<option value="' + esc(sample.sample_id) + '">' + esc(optionLabel(sample)) + "</option>";
      }).join("");
      groups.push('<optgroup label="Custom uploads">' + options + "</optgroup>");
    }
    select.innerHTML = groups.join("");
  }

  async function loadSampleList() {
    const select = $("sample-select");
    const preferred = "speech_A_level2_pacing_monotone";
    let samples = FALLBACK_SAMPLES.slice();
    fillPresetSelect(samples);
    select.value = preferred;
    try {
      const res = await fetch("/api/samples");
      if (!res.ok) throw new Error("Sample list returned " + res.status);
      const data = await res.json();
      if (data.samples && data.samples.length) samples = data.samples;
    } catch (err) {
      console.error("[AuraVoice] Sample list failed", err);
    }
    console.log("[AuraVoice] Samples loaded:", samples);
    const keep = select.value;
    fillPresetSelect(samples);
    const hasKeep = keep && Array.from(select.options).some((option) => option.value === keep);
    select.value = hasKeep ? keep : (samples.some((sample) => sample.sample_id === preferred) ? preferred : samples[0].sample_id);
    if (select.value) await loadSample(select.value);
    else setStatus("No evaluations found");
  }

  $("sample-select").addEventListener("change", async (e) => {
    const id = e.target.value;
    if (!id) return;
    await loadSample(id);
  });
  function setUploadModal(open) {
    const modal = $("upload-modal");
    modal.style.display = open ? "flex" : "none";
  }
  $("upload-modal-btn").addEventListener("click", () => setUploadModal(true));
  $("upload-browse").addEventListener("click", () => $("upload-audio").click());
  $("upload-audio").addEventListener("change", () => {
    const file = $("upload-audio").files && $("upload-audio").files[0];
    $("upload-filename").textContent = file ? file.name : "No file selected";
  });
  $("upload-cancel").addEventListener("click", () => setUploadModal(false));
  $("upload-form").addEventListener("click", (ev) => ev.stopPropagation());
  $("upload-modal").addEventListener("click", () => setUploadModal(false));
  $("btn-play").addEventListener("click", () => {
    if (state.candidate) state.candidate.playPause();
  });
  $("btn-stop").addEventListener("click", () => {
    stopWave(state.candidate);
    if (syncEnabled()) stopWave(state.baseline);
    state.playhead = 0;
    renderClock(0);
    if (state.chart) state.chart.draw();
    setTransportLabel($("btn-play"), false);
  });
  $("btn-mute").addEventListener("click", () => {
    const next = $("btn-mute").getAttribute("aria-pressed") !== "true";
    $("btn-mute").setAttribute("aria-pressed", next ? "true" : "false");
    $("btn-mute").textContent = next ? "Unmute" : "Mute";
    applyLevels();
  });
  $("volume").addEventListener("input", applyLevels);
  $("playback-rate").addEventListener("change", applyRate);
  $("btn-baseline-play").addEventListener("click", () => {
    if (!state.baseline) return;
    if (syncEnabled() && state.candidate) {
      state.candidate.playPause();
      return;
    }
    applyLevels();
    state.baseline.setMuted($("btn-baseline-mute").getAttribute("aria-pressed") === "true");
    state.baseline.playPause();
  });
  $("btn-baseline-stop").addEventListener("click", () => {
    stopWave(state.baseline);
    setTransportLabel($("btn-baseline-play"), false);
  });
  $("btn-baseline-mute").addEventListener("click", () => {
    const next = $("btn-baseline-mute").getAttribute("aria-pressed") !== "true";
    $("btn-baseline-mute").setAttribute("aria-pressed", next ? "true" : "false");
    $("btn-baseline-mute").textContent = next ? "Unmute" : "Mute";
    applyLevels();
  });
  $("baseline-volume").addEventListener("input", () => {
    if (state.baseline && !syncEnabled()) state.baseline.setMuted(false);
    applyLevels();
  });
  $("sync-baseline").addEventListener("change", () => {
    applyLevels();
    applyRate();
    if (!state.baseline || !state.candidate) return;
    if (syncEnabled() && state.candidate.isPlaying()) {
      state.baseline.setTime(state.candidate.getCurrentTime());
      state.baseline.play();
    } else if (!syncEnabled()) {
      state.baseline.pause();
      applyLevels();
    }
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.code !== "Space") return;
    const tag = ev.target && ev.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || tag === "BUTTON") return;
    ev.preventDefault();
    if (state.candidate) state.candidate.playPause();
  });
  $("upload-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const file = $("upload-audio").files && $("upload-audio").files[0];
    if (!file) {
      setStatus("Choose a WAV file first");
      $("upload-filename").textContent = "Choose a .wav file before running the pipeline";
      return;
    }
    if (!/\.wav$/i.test(file.name)) {
      setStatus("Upload a .wav file");
      return;
    }
    setStatus("Aligning, extracting, grounding, scoring…");
    const body = new FormData();
    body.append("audio", file);
    body.append("baseline_id", $("upload-baseline").value);
    body.append("sample_id", "upload_" + Date.now());
    const res = await fetch("/api/evaluate", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setStatus(data.detail || "Evaluation failed");
      return;
    }
    setUploadModal(false);
    applyPayload(data);
    if (data.sample_id && !$("sample-select").querySelector('option[value="' + data.sample_id + '"]')) {
      const group = $("sample-select").querySelector('optgroup[label="Custom uploads"]') || document.createElement("optgroup");
      group.label = "Custom uploads";
      const option = document.createElement("option");
      option.value = data.sample_id;
      option.textContent = optionLabel({
        sample_id: data.sample_id,
        level: data.severity_level < 0 ? 0 : data.severity_level,
        overall_score: data.overall_score,
        letter_grade: data.letter_grade,
      });
      group.appendChild(option);
      $("sample-select").appendChild(group);
    }
    if (data.sample_id) $("sample-select").value = data.sample_id;
  });

  loadSampleList().catch((err) => {
    console.error("[AuraVoice] Preset boot failed", err);
    try {
      fillPresetSelect(FALLBACK_SAMPLES);
      $("sample-select").value = "speech_A_level2_pacing_monotone";
    } catch (fillErr) {
      console.error(fillErr);
    }
    setStatus(String(err));
  });
  try {
    initHud();
  } catch (err) {
    console.error("[AuraVoice] HUD failed", err);
  }
})();
