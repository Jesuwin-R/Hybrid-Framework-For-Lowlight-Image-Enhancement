const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const runBtn = document.getElementById('runBtn');
const statusEl = document.getElementById('status');
const progressBarFill = document.getElementById('progressBarFill');
const progressWrap = document.getElementById('progressWrap');
const resultsSection = document.getElementById('resultsSection');
const originalImg = document.getElementById('originalImg');
const overviewThumbs = document.getElementById('overviewThumbs');
const overviewChart = document.getElementById('overviewChart');
const fusionCardWrap = document.getElementById('fusionCardWrap');

let selectedFile = null;

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

function handleFile(file) {
  selectedFile = file;
  runBtn.disabled = false;
  dropzone.querySelector('.dropzone-label').textContent = `Selected: ${file.name}`;
}

function setStatus(text) { statusEl.classList.remove('hidden'); statusEl.textContent = text; }
function setProgress(done, total) {
  progressWrap.classList.remove('hidden');
  progressBarFill.style.width = `${Math.round((done / total) * 100)}%`;
}

runBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  runBtn.disabled = true;
  fusionCardWrap.innerHTML = '';
  resultsSection.classList.add('hidden');
  progressBarFill.style.width = '0%';
  setStatus('Uploading...');

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const startRes = await fetch('/api/start', { method: 'POST', body: formData });
    const startData = await startRes.json();

    if (!startRes.ok) {
      setStatus(`Error: ${startData.error || 'Upload failed.'}`);
      runBtn.disabled = false;
      return;
    }

    const { session_id, input_url, model_order, model_names } = startData;
    originalImg.src = input_url;
    resultsSection.classList.remove('hidden');
    buildThumbPlaceholders(model_order, model_names);

    const totalSteps = model_order.length + 1; // + fusion
    let step = 0;
    setProgress(step, totalSteps);

    const qualityScores = []; // [{key, name, composite}] — actual Qi, not the fusion weight

    // Run each of the 5 models sequentially — guaranteed one-at-a-time reveal
    for (const key of model_order) {
      setStatus(`Running ${model_names[key]}... (${step + 1}/${totalSteps})`);

      const res = await fetch('/api/run_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id, model_key: key }),
      });
      const data = await res.json();

      step++;
      setProgress(step, totalSteps);

      if (!res.ok) {
        fillThumb(key, null, data.error || 'Failed');
        continue;
      }
      fillThumb(key, data.image_url);
      qualityScores.push({ key, name: model_names[key], composite: data.quality.composite });
    }

    renderQualityScoreChart(qualityScores);

    // Quality-weighted fusion of all 5
    setStatus(`Fusing all 5 models by quality weight... (${step + 1}/${totalSteps})`);
    const fusionRes = await fetch('/api/run_fusion', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id }),
    });
    const fusionData = await fusionRes.json();
    step++;
    setProgress(step, totalSteps);

    if (fusionRes.ok) {
      fusionData.input_url = input_url;
      addFusionCard(fusionData);
    } else {
      fusionCardWrap.innerHTML = `<div class="result-card error-card"><div class="result-body">
        <h3 class="result-name">Fusion</h3>
        <p class="error-text">Failed: ${fusionData.error || 'Failed'}</p>
      </div></div>`;
    }

    statusEl.classList.add('hidden');
    progressWrap.classList.add('hidden');
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    runBtn.disabled = false;
  }
});

function buildThumbPlaceholders(modelOrder, modelNames) {
  overviewThumbs.innerHTML = '';
  for (const key of modelOrder) {
    const box = document.createElement('div');
    box.className = 'overview-thumb pending';
    box.dataset.key = key;
    box.innerHTML = `
      <img alt="${modelNames[key]}">
      <div class="overview-thumb-label">${modelNames[key]}</div>
    `;
    overviewThumbs.appendChild(box);
  }
}

function fillThumb(key, imageUrl, errorText) {
  const box = overviewThumbs.querySelector(`[data-key="${key}"]`);
  if (!box) return;
  box.classList.remove('pending');
  if (imageUrl) {
    box.querySelector('img').src = imageUrl;
  } else {
    box.classList.add('error');
    box.querySelector('.overview-thumb-label').textContent = errorText || 'Failed';
  }
}

// Plots each model's actual composite quality score Qi (0-1) — NOT a
// percentage, and NOT the softmax fusion weight. Qi = weighted average of
// brightness-balance, contrast, entropy, sharpness and colorfulness scores,
// each already normalized to 0-1 (see metrics.py:score_no_reference).
function renderQualityScoreChart(qualityScores) {
  if (!qualityScores || !qualityScores.length) return;

  const width = 560;
  const height = 200;
  const barGap = 16;
  const barWidth = (width - barGap * (qualityScores.length - 1)) / qualityScores.length;
  const chartTop = 20;
  const chartBottom = height - 36;
  const chartHeight = chartBottom - chartTop;
  const axisMax = 1.0; // Qi is bounded to [0, 1] by construction

  let bars = '';
  qualityScores.forEach((s, i) => {
    const barHeight = (s.composite / axisMax) * chartHeight;
    const x = i * (barWidth + barGap);
    const y = chartBottom - barHeight;
    bars += `
      <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="4" fill="var(--amber)"></rect>
      <text x="${x + barWidth / 2}" y="${y - 8}" text-anchor="middle" font-family="var(--font-mono)" font-size="12" fill="var(--text)">${s.composite.toFixed(3)}</text>
      <text x="${x + barWidth / 2}" y="${chartBottom + 18}" text-anchor="middle" font-family="var(--font-mono)" font-size="10.5" fill="var(--muted)">${s.name}</text>
    `;
  });

  overviewChart.innerHTML = `
    <svg width="100%" viewBox="0 0 ${width} ${height}" style="overflow:visible">
      <line x1="0" y1="${chartBottom}" x2="${width}" y2="${chartBottom}" stroke="var(--border)" stroke-width="1"></line>
      ${bars}
    </svg>
  `;
}

function addFusionCard(r) {
  const card = document.createElement('div');
  card.className = 'result-card fusion-card';
  card.dataset.key = r.key;

  const psnrRow = ('psnr' in r)
    ? `<div class="metric-row"><span>PSNR</span><span>${r.psnr} dB</span></div>
       <div class="metric-row"><span>SSIM</span><span>${r.ssim}</span></div>`
    : `<div class="metric-row"><span>Quality score</span><span>${r.quality.composite}</span></div>
       <div class="metric-row"><span>Contrast</span><span>${r.quality.contrast}</span></div>
       <div class="metric-row"><span>Entropy</span><span>${r.quality.entropy}</span></div>`;

  let weightsHtml = '';
  if (r.weights) {
    const rows = r.weights.map(w => `
      <div class="weight-row">
        <span class="weight-name">${w.name}</span>
        <div class="weight-bar-track"><div class="weight-bar-fill" style="width:${w.weight_percent}%"></div></div>
        <span class="weight-pct">${w.weight_percent}%</span>
      </div>
    `).join('');
    weightsHtml = `
      <div class="weights-block">
        <div class="weights-title">Contribution to fusion (%) — softmax of Qi</div>
        ${rows}
      </div>
    `;
  }

  card.innerHTML = `
    <h3 class="result-name">${r.name}</h3>
    <div class="card-compare">
      <div class="compare-half">
        <span class="compare-label">Original</span>
        <img src="${r.input_url}" alt="Original input">
      </div>
      <div class="compare-half">
        <span class="compare-label">${r.name}</span>
        <img src="${r.image_url}" alt="${r.name} output">
      </div>
    </div>
    <div class="result-body">
      ${psnrRow}
      <div class="metric-row"><span>Time</span><span>${r.time_seconds}s</span></div>
    </div>
    ${weightsHtml}
  `;
  fusionCardWrap.appendChild(card);
  requestAnimationFrame(() => card.classList.add('visible'));
}

// ---------- Click-to-enlarge lightbox (stays on the same page, no new tab) ----------
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightboxImg');
const lightboxCaption = document.getElementById('lightboxCaption');
const lightboxClose = document.getElementById('lightboxClose');
const lightboxCompare = document.getElementById('lightboxCompare');

let lbEnhancedSrc = '';
let lbOriginalSrc = '';
let lbCaption = '';

function openLightbox(src, caption, originalSrc) {
  lbEnhancedSrc = src;
  lbOriginalSrc = originalSrc || '';
  lbCaption = caption || '';
  lightboxImg.src = src;
  lightboxCaption.textContent = lbCaption;
  if (lbOriginalSrc) {
    new Image().src = lbOriginalSrc; // preload so the swap is instant
    lightboxCompare.classList.remove('hidden');
  } else {
    lightboxCompare.classList.add('hidden');
  }
  lightbox.classList.remove('hidden');
  document.body.classList.add('lightbox-open');
}
function closeLightbox() {
  showEnhanced();
  lightbox.classList.add('hidden');
  document.body.classList.remove('lightbox-open');
  lightboxImg.removeAttribute('src');
}

// Compare: show the original while the button is pressed, enhanced when released
function showOriginal() {
  if (!lbOriginalSrc) return;
  lightboxImg.src = lbOriginalSrc;
  lightboxCaption.textContent = 'Original';
  lightboxCompare.classList.add('active');
}
function showEnhanced() {
  if (!lbEnhancedSrc) return;
  lightboxImg.src = lbEnhancedSrc;
  lightboxCaption.textContent = lbCaption;
  lightboxCompare.classList.remove('active');
}
lightboxCompare.addEventListener('pointerdown', (e) => { e.preventDefault(); showOriginal(); });
['pointerup', 'pointerleave', 'pointercancel'].forEach(ev =>
  lightboxCompare.addEventListener(ev, showEnhanced));
lightboxCompare.addEventListener('contextmenu', (e) => e.preventDefault());
lightboxCompare.addEventListener('keydown', (e) => {
  if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) { e.preventDefault(); showOriginal(); }
});
lightboxCompare.addEventListener('keyup', (e) => {
  if (e.key === ' ' || e.key === 'Enter') showEnhanced();
});

// Event delegation so dynamically added images (thumbs, fusion card) work too
document.addEventListener('click', (e) => {
  const img = e.target.closest('.original-block img, .overview-thumb img, .compare-half img');
  if (!img || !img.getAttribute('src')) return;
  if (img.closest('.overview-thumb.pending, .overview-thumb.error')) return;

  // Original images get no compare button; enhanced ones compare against the input
  const isOriginal = img.closest('.original-block') || img.alt === 'Original input';
  openLightbox(img.src, img.alt, isOriginal ? '' : originalImg.src);
});

lightboxClose.addEventListener('click', closeLightbox);
lightbox.addEventListener('click', (e) => { if (e.target === lightbox) closeLightbox(); });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !lightbox.classList.contains('hidden')) closeLightbox();
});
