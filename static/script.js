// ============================================================
// FaceRate AI — Frontend Logic
// ============================================================

const $ = (sel) => document.querySelector(sel);
const uploadZone  = $('#uploadZone');
const fileInput   = $('#fileInput');
const previewCont = $('#previewContainer');
const previewImg  = $('#previewImg');
const analyzeBtn  = $('#analyzeBtn');
const results     = $('#results');
const errorMsg    = $('#errorMsg');
const scoreNumber = $('#scoreNumber');
const ringFg      = $('#ringFg');
const verdictText = $('#verdictText');
const traitsList  = $('#traitsList');

let selectedFile = null;

// ---------- Drag & drop ----------
uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});

// ---------- Click to select ----------
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ---------- Handle selected file ----------
function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewCont.classList.add('visible');
    results.classList.remove('visible');
    errorMsg.classList.remove('visible');
  };
  reader.readAsDataURL(file);
}

// ---------- Analyze ----------
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  analyzeBtn.classList.add('loading');
  results.classList.remove('visible');
  errorMsg.classList.remove('visible');

  const form = new FormData();
  form.append('file', selectedFile);

  try {
    const res = await fetch('/analyze', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Произошла ошибка');
      return;
    }

    renderResults(data);
  } catch (err) {
    showError('Не удалось связаться с сервером');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('loading');
  }
});

// ---------- Render results ----------
function renderResults(data) {
  // Animate score number
  const target = data.score;
  animateNumber(scoreNumber, 0, target, 1200);

  // Animate ring
  const circumference = 2 * Math.PI * 65; // r=65
  const offset = circumference * (1 - target / 10);
  ringFg.style.strokeDasharray = circumference;
  ringFg.style.strokeDashoffset = circumference;
  requestAnimationFrame(() => {
    ringFg.style.strokeDashoffset = offset;
  });

  // Verdict
  verdictText.textContent = data.verdict;

  // Traits
  traitsList.innerHTML = '';
  data.traits.forEach((t) => {
    const div = document.createElement('div');
    div.className = 'trait';
    div.innerHTML = `
      <div class="trait-score">${t.score}</div>
      <div class="trait-info">
        <div class="trait-name">${t.name}</div>
        <div class="trait-comment">${t.comment}</div>
      </div>
    `;
    traitsList.appendChild(div);
  });

  results.classList.add('visible');
}

// ---------- Animate number ----------
function animateNumber(el, from, to, duration) {
  const start = performance.now();
  const step = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3); // ease-out cubic
    el.textContent = (from + (to - from) * ease).toFixed(1);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ---------- Show error ----------
function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.add('visible');
}
