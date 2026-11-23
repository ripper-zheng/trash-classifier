/* 前端交互：结果全部来自后端 /predict 真实推理，严禁写死类别或置信度。 */

const $ = (id) => document.getElementById(id);

const els = {
  fileInput: $('file-input'),
  dropzone: $('dropzone'),
  dropzoneInner: $('dropzone-inner'),
  preview: $('preview'),
  fileMeta: $('file-meta'),
  fileName: $('file-name'),
  fileSize: $('file-size'),
  btnSelect: $('btn-select'),
  btnPredict: $('btn-predict'),
  btnCamera: $('btn-camera'),
  cameraBox: $('camera-box'),
  cameraVideo: $('camera-video'),
  btnCapture: $('btn-capture'),
  errorMsg: $('error-msg'),
  serviceStatus: $('service-status'),
  modelBadge: $('model-badge'),
  stateLoading: $('state-loading'),
  stateEmpty: $('state-empty'),
  stateResult: $('state-result'),
  resultCn: $('result-cn'),
  resultClass: $('result-class'),
  resultCategory: $('result-category'),
  confidenceValue: $('confidence-value'),
  confidenceBar: $('confidence-bar'),
  top3: $('top3'),
  latency: $('latency'),
  resultModel: $('result-model'),
  resultVersion: $('result-version'),
  advice: $('advice'),
  reviewHint: $('review-hint'),
  classList: $('class-list'),
  inputSize: $('input-size'),
};

const ALLOWED = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_SIZE = 10 * 1024 * 1024;
let currentFile = null;
let cameraStream = null;

/* ---------- 状态检查 ---------- */
async function checkService() {
  try {
    const r = await fetch('/health');
    const data = await r.json();
    if (data.status === 'ok' && data.model_loaded) {
      els.serviceStatus.textContent = '服务正常 · 模型已加载';
      els.serviceStatus.className = 'badge badge-ok';
      const info = await (await fetch('/model-info')).json();
      els.modelBadge.textContent = `模型：${info.model_name} ${info.model_version}`;
      els.classList.textContent = info.classes.join(' / ');
      els.inputSize.textContent = `${info.input_size} × ${info.input_size}`;
    } else {
      throw new Error('模型未加载');
    }
  } catch (e) {
    els.serviceStatus.textContent = '服务异常 · 请检查后端';
    els.serviceStatus.className = 'badge badge-bad';
    showError('后端服务或模型未就绪，请先启动 src/app.py');
  }
}

/* ---------- 文件处理 ---------- */
function showError(msg) {
  els.errorMsg.textContent = msg;
  els.errorMsg.classList.remove('hidden');
}
function clearError() { els.errorMsg.classList.add('hidden'); }

function acceptFile(file) {
  clearError();
  if (!file) return;
  if (!ALLOWED.includes(file.type)) {
    showError('不支持的文件类型，请选择 JPG / PNG / WEBP 图片。');
    return;
  }
  if (file.size > MAX_SIZE) {
    showError('文件超过 10 MB，请选择更小的图片。');
    return;
  }
  if (file.size === 0) {
    showError('空文件，请重新选择。');
    return;
  }
  currentFile = file;
  const url = URL.createObjectURL(file);
  els.preview.src = url;
  els.preview.classList.remove('hidden');
  els.dropzoneInner.classList.add('hidden');
  els.fileMeta.classList.remove('hidden');
  els.fileName.textContent = file.name;
  els.fileSize.textContent = (file.size / 1024).toFixed(1) + ' KB';
  els.btnPredict.disabled = false;
  showState('empty');
}

function resetFile() {
  currentFile = null;
  els.fileInput.value = '';
  els.preview.classList.add('hidden');
  els.dropzoneInner.classList.remove('hidden');
  els.fileMeta.classList.add('hidden');
  els.btnPredict.disabled = true;
  clearError();
  showState('empty');
}

/* ---------- 状态切换：loading / empty / result ---------- */
function showState(which) {
  els.stateLoading.classList.add('hidden');
  els.stateEmpty.classList.add('hidden');
  els.stateResult.classList.add('hidden');
  if (which === 'loading') els.stateLoading.classList.remove('hidden');
  else if (which === 'empty') els.stateEmpty.classList.remove('hidden');
  else if (which === 'result') els.stateResult.classList.remove('hidden');
}

/* ---------- 推理 ---------- */
async function doPredict() {
  if (!currentFile) { showError('请先选择一张图片。'); return; }
  clearError();
  showState('loading');
  els.btnPredict.disabled = true;

  const form = new FormData();
  form.append('file', currentFile);
  try {
    const resp = await fetch('/predict', { method: 'POST', body: form });
    const data = await resp.json();
    if (!resp.ok || data.status !== 'success') {
      throw new Error(data.message || `请求失败 (${resp.status})`);
    }
    renderResult(data);
    showState('result');
  } catch (e) {
    showState('empty');
    showError('识别失败：' + e.message);
  } finally {
    els.btnPredict.disabled = false;
  }
}

function renderResult(d) {
  els.resultCn.textContent = d.cn;
  els.resultClass.textContent = d.class;
  els.resultCategory.textContent = d.category;
  els.confidenceValue.textContent = (d.confidence * 100).toFixed(1) + '%';
  els.confidenceBar.style.width = (d.confidence * 100).toFixed(1) + '%';
  els.top3.innerHTML = '';
  d.top3.forEach((item, i) => {
    const row = document.createElement('div');
    row.className = 'top3-item';
    row.innerHTML = `<span class="rank">${i + 1}</span>
      <span class="name">${item.cn} (${item.class})</span>
      <span class="mini-bar"><i style="width:${(item.score * 100).toFixed(1)}%"></i></span>
      <span class="score">${(item.score * 100).toFixed(1)}%</span>`;
    els.top3.appendChild(row);
  });
  els.latency.textContent = d.latency_ms + ' ms';
  els.resultModel.textContent = d.model_name;
  els.resultVersion.textContent = d.model_version;
  els.advice.textContent = d.advice;
  els.reviewHint.classList.toggle('hidden', !d.need_review);
}

/* ---------- 摄像头 ---------- */
async function toggleCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
    els.cameraBox.classList.add('hidden');
    els.btnCamera.textContent = '调用摄像头';
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    els.cameraVideo.srcObject = cameraStream;
    els.cameraBox.classList.remove('hidden');
    els.btnCamera.textContent = '关闭摄像头';
  } catch (e) {
    showError('无法访问摄像头：' + e.message);
  }
}

function capture() {
  const canvas = document.createElement('canvas');
  canvas.width = els.cameraVideo.videoWidth;
  canvas.height = els.cameraVideo.videoHeight;
  canvas.getContext('2d').drawImage(els.cameraVideo, 0, 0);
  canvas.toBlob((blob) => {
    if (!blob) return;
    const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
    acceptFile(file);
  }, 'image/jpeg', 0.92);
}

/* ---------- 事件绑定 ---------- */
els.dropzone.addEventListener('click', () => els.fileInput.click());
els.btnSelect.addEventListener('click', () => { resetFile(); els.fileInput.click(); });
els.fileInput.addEventListener('change', (e) => acceptFile(e.target.files[0]));
els.btnPredict.addEventListener('click', doPredict);
els.btnCamera.addEventListener('click', toggleCamera);
els.btnCapture.addEventListener('click', capture);

['dragenter', 'dragover'].forEach(ev => els.dropzone.addEventListener(ev, (e) => {
  e.preventDefault(); els.dropzone.classList.add('dragover');
}));
['dragleave', 'drop'].forEach(ev => els.dropzone.addEventListener(ev, (e) => {
  e.preventDefault(); els.dropzone.classList.remove('dragover');
}));
els.dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  acceptFile(file);
});

checkService();
