/**
 * Smart Fridge Bounding Box Labeler - Client Application
 */

// ── State ────────────────────────────────────────────────────────
const state = {
  currentDirectory: "",
  images: [], // [ { filename, path, width, height, size_bytes } ]
  currentIndex: 0,
  annotations: {}, // { filename: [ { id, label, bbox: [x1, y1, x2, y2] } ] }
  imageMetadata: {}, // { filename: { width, height } }
  vocabulary: [],
  selectedBoxId: null,
  activeTool: "select", // 'select' | 'draw'
  isDirty: false,

  // Canvas Viewport Transform
  scale: 1,
  panX: 0,
  panY: 0,
  isPanning: false,
  panStartX: 0,
  panStartY: 0,
  spacePressed: false,

  // Drawing & Interaction State
  isDrawing: false,
  drawStart: null,
  isDraggingBox: false,
  dragOffset: null,
  isResizing: false,
  activeHandle: null, // 'tl' | 'tr' | 'br' | 'bl' | 't' | 'r' | 'b' | 'l'
  resizeStartBox: null,
  hoverHandle: null,

  // Model Predictions Overlay
  showModelOverlay: false,
  availablePredictionFiles: [],
  currentPredictionFile: "",
  modelPredictions: {}, // { filename: [ { label, confidence, bbox } ] }

  // History for Undo / Redo (per current image)
  history: [],
  historyIndex: -1,

  // Current filter tab
  filter: "all", // 'all' | 'labeled' | 'unlabeled'
};

// Distinct color palette for labels
const COLOR_PALETTE = [
  "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
  "#EC4899", "#06B6D4", "#F97316", "#14B8A6", "#6366F1",
  "#84CC16", "#D946EF", "#EAB308", "#22C55E", "#A855F7",
  "#0EA5E9", "#F43F5E", "#E11D48", "#475569", "#78716C"
];

const labelColorMap = new Map();
function getLabelColor(label) {
  if (!label) return "#3B82F6";
  const normalized = label.toLowerCase().trim();
  if (!labelColorMap.has(normalized)) {
    let hash = 0;
    for (let i = 0; i < normalized.length; i++) {
      hash = normalized.charCodeAt(i) + ((hash << 5) - hash);
    }
    const colorIndex = Math.abs(hash) % COLOR_PALETTE.length;
    labelColorMap.set(normalized, COLOR_PALETTE[colorIndex]);
  }
  return labelColorMap.get(normalized);
}

// ── DOM Elements ────────────────────────────────────────────────
const canvas = document.getElementById("mainCanvas");
const ctx = canvas.getContext("2d");
const canvasViewport = document.getElementById("canvasViewport");
const dirInput = document.getElementById("dirInput");
const loadDirBtn = document.getElementById("loadDirBtn");
const saveBtn = document.getElementById("saveBtn");
const saveStatus = document.getElementById("saveStatus");
const imageListEl = document.getElementById("imageList");
const prevImgBtn = document.getElementById("prevImgBtn");
const nextImgBtn = document.getElementById("nextImgBtn");
const imgCounter = document.getElementById("imgCounter");
const totalImagesBadge = document.getElementById("totalImagesBadge");
const imgDimensionsEl = document.getElementById("imgDimensions");
const mouseCoordsEl = document.getElementById("mouseCoords");
const activeFileNameEl = document.getElementById("activeFileName");
const zoomLevelText = document.getElementById("zoomLevelText");

// Tools
const toolSelectBtn = document.getElementById("toolSelect");
const toolDrawBtn = document.getElementById("toolDraw");
const zoomInBtn = document.getElementById("zoomInBtn");
const zoomOutBtn = document.getElementById("zoomOutBtn");
const zoomFitBtn = document.getElementById("zoomFitBtn");
const undoBtn = document.getElementById("undoBtn");
const redoBtn = document.getElementById("redoBtn");
const deleteBoxBtn = document.getElementById("deleteBoxBtn");

// Model Overlay
const showModelOverlayCheck = document.getElementById("showModelOverlay");
const modelPredictionSelect = document.getElementById("modelPredictionSelect");
const importPredictionsBtn = document.getElementById("importPredictionsBtn");
const modelLegendCard = document.getElementById("modelLegendCard");
const modelBoxCountBadge = document.getElementById("modelBoxCountBadge");
const modelDetectionsList = document.getElementById("modelDetectionsList");

// Inspector
const boxEditorContent = document.getElementById("boxEditorContent");
const selectedBoxIdBadge = document.getElementById("selectedBoxId");
const labelInput = document.getElementById("labelInput");
const autocompleteList = document.getElementById("autocompleteList");
const quickChipsList = document.getElementById("quickChipsList");
const coordX1 = document.getElementById("coordX1");
const coordY1 = document.getElementById("coordY1");
const coordX2 = document.getElementById("coordX2");
const coordY2 = document.getElementById("coordY2");
const coordW = document.getElementById("coordW");
const coordH = document.getElementById("coordH");
const deleteActiveBoxBtn = document.getElementById("deleteActiveBoxBtn");
const annotationsList = document.getElementById("annotationsList");
const currentBoxesCount = document.getElementById("currentBoxesCount");
const clearCurrentBoxesBtn = document.getElementById("clearCurrentBoxesBtn");

// Help Modal
const helpBtn = document.getElementById("helpBtn");
const helpModal = document.getElementById("helpModal");
const closeHelpBtn = document.getElementById("closeHelpBtn");
const gotItBtn = document.getElementById("gotItBtn");
const toastContainer = document.getElementById("toastContainer");

// Active image HTML Image element
let activeImageObj = null;

// ── Notifications ───────────────────────────────────────────────
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s";
    setTimeout(() => toast.remove(), 300);
  }, 2800);
}

function setDirty(isDirty = true) {
  state.isDirty = isDirty;
  if (isDirty) {
    saveStatus.classList.add("dirty");
    saveStatus.querySelector(".status-text").textContent = "Unsaved Changes";
  } else {
    saveStatus.classList.remove("dirty");
    saveStatus.querySelector(".status-text").textContent = "Saved";
  }
}

// ── History (Undo / Redo) ───────────────────────────────────────
function pushHistory() {
  const curImg = getCurrentFilename();
  if (!curImg) return;
  const boxes = JSON.parse(JSON.stringify(state.annotations[curImg] || []));
  
  // Truncate forward history if we made changes after undo
  state.history = state.history.slice(0, state.historyIndex + 1);
  state.history.push(boxes);
  state.historyIndex++;
  updateUndoRedoButtons();
}

function undo() {
  if (state.historyIndex > 0) {
    state.historyIndex--;
    const curImg = getCurrentFilename();
    state.annotations[curImg] = JSON.parse(JSON.stringify(state.history[state.historyIndex]));
    state.selectedBoxId = null;
    setDirty(true);
    renderAll();
    updateUndoRedoButtons();
  }
}

function redo() {
  if (state.historyIndex < state.history.length - 1) {
    state.historyIndex++;
    const curImg = getCurrentFilename();
    state.annotations[curImg] = JSON.parse(JSON.stringify(state.history[state.historyIndex]));
    state.selectedBoxId = null;
    setDirty(true);
    renderAll();
    updateUndoRedoButtons();
  }
}

function updateUndoRedoButtons() {
  undoBtn.style.opacity = state.historyIndex > 0 ? "1" : "0.4";
  redoBtn.style.opacity = state.historyIndex < state.history.length - 1 ? "1" : "0.4";
}

// ── Helpers ─────────────────────────────────────────────────────
function getCurrentFilename() {
  if (state.images.length === 0 || state.currentIndex >= state.images.length) return null;
  return state.images[state.currentIndex].filename;
}

function getCurrentBoxes() {
  const name = getCurrentFilename();
  if (!name) return [];
  if (!state.annotations[name]) state.annotations[name] = [];
  return state.annotations[name];
}

function getSelectedBox() {
  if (!state.selectedBoxId) return null;
  const boxes = getCurrentBoxes();
  return boxes.find(b => b.id === state.selectedBoxId) || null;
}

// Coordinate conversions
function screenToImageCoords(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const screenX = clientX - rect.left;
  const screenY = clientY - rect.top;
  const imgX = (screenX - state.panX) / state.scale;
  const imgY = (screenY - state.panY) / state.scale;
  return { x: imgX, y: imgY };
}

function imageToScreenCoords(imgX, imgY) {
  return {
    x: imgX * state.scale + state.panX,
    y: imgY * state.scale + state.panY
  };
}

// ── App Initialization ──────────────────────────────────────────
async function initApp() {
  setupEventListeners();
  resizeCanvas();
  window.addEventListener("resize", () => {
    resizeCanvas();
    renderAll();
  });

  try {
    const res = await fetch("/api/config");
    const data = await res.json();
    state.vocabulary = data.vocabulary || [];
    renderQuickChips();

    // Default directory from backend
    if (data.default_directory) {
      dirInput.value = data.default_directory;
      await loadDirectory(data.default_directory);
    }
  } catch (err) {
    console.error("Failed to load initial config:", err);
    showToast("Error connecting to backend server", "error");
  }
}

// ── Directory Loading ───────────────────────────────────────────
async function loadDirectory(path) {
  if (!path || !path.trim()) return;
  loadDirBtn.classList.add("loading");
  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ directory_path: path.trim() }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to scan folder");
    }

    const data = await res.json();
    state.currentDirectory = data.directory;
    state.images = data.images;
    state.annotations = data.existing_annotations || {};
    state.availablePredictionFiles = data.prediction_files || [];
    state.currentIndex = 0;
    state.selectedBoxId = null;

    // Cache metadata
    state.imageMetadata = {};
    for (const img of state.images) {
      state.imageMetadata[img.filename] = { width: img.width, height: img.height };
    }

    // Populate prediction files dropdown
    populatePredictionSelect();

    updateImageListUI();
    if (state.images.length > 0) {
      await loadImage(0);
      showToast(`Loaded ${state.images.length} images from folder`, "success");
    } else {
      showToast("No images found in folder", "info");
      clearCanvas();
    }
    setDirty(false);
  } catch (err) {
    console.error("Load directory error:", err);
    showToast(err.message, "error");
  } finally {
    loadDirBtn.classList.remove("loading");
  }
}

function populatePredictionSelect() {
  modelPredictionSelect.innerHTML = "";
  if (state.availablePredictionFiles.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No predictions found";
    modelPredictionSelect.appendChild(opt);
    return;
  }

  for (const pf of state.availablePredictionFiles) {
    const opt = document.createElement("option");
    opt.value = pf.path;
    opt.textContent = pf.label;
    modelPredictionSelect.appendChild(opt);
  }

  // Pre-select first prediction file and load it
  if (state.availablePredictionFiles.length > 0) {
    state.currentPredictionFile = state.availablePredictionFiles[0].path;
    modelPredictionSelect.value = state.currentPredictionFile;
    loadPredictionFile(state.currentPredictionFile);
  }
}

async function loadPredictionFile(filePath) {
  if (!filePath) return;
  try {
    const res = await fetch(`/api/predictions?path=${encodeURIComponent(filePath)}`);
    if (res.ok) {
      const data = await res.json();
      state.modelPredictions = data.predictions || {};
      renderAll();
      updateModelLegend();
    }
  } catch (e) {
    console.error("Error loading predictions:", e);
  }
}

// ── Load Single Image ───────────────────────────────────────────
async function loadImage(index) {
  if (index < 0 || index >= state.images.length) return;
  state.currentIndex = index;
  state.selectedBoxId = null;

  const imgData = state.images[index];
  activeFileNameEl.textContent = imgData.filename;
  imgCounter.textContent = `Image ${index + 1} of ${state.images.length}`;
  prevImgBtn.disabled = index === 0;
  nextImgBtn.disabled = index === state.images.length - 1;

  // Load Image Object
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      activeImageObj = img;
      imgData.width = img.naturalWidth;
      imgData.height = img.naturalHeight;
      state.imageMetadata[imgData.filename] = { width: img.naturalWidth, height: img.naturalHeight };
      imgDimensionsEl.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;

      fitImageToViewport();

      // Reset image history
      state.history = [JSON.parse(JSON.stringify(getCurrentBoxes()))];
      state.historyIndex = 0;
      updateUndoRedoButtons();

      updateImageListUI();
      renderAll();
      updateAnnotationsSidebar();
      updateInspector();
      updateModelLegend();
      resolve();
    };
    img.src = `/api/image?path=${encodeURIComponent(imgData.path)}`;
  });
}

// ── Canvas Fitting & Sizing ─────────────────────────────────────
function resizeCanvas() {
  const rect = canvasViewport.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
}

function fitImageToViewport() {
  if (!activeImageObj) return;
  const padding = 40;
  const availableW = canvas.width - padding * 2;
  const availableH = canvas.height - padding * 2;
  const scaleW = availableW / activeImageObj.naturalWidth;
  const scaleH = availableH / activeImageObj.naturalHeight;
  state.scale = Math.min(scaleW, scaleH, 1.0); // Don't scale up past 100% by default

  state.panX = Math.round((canvas.width - activeImageObj.naturalWidth * state.scale) / 2);
  state.panY = Math.round((canvas.height - activeImageObj.naturalHeight * state.scale) / 2);
  zoomLevelText.textContent = `${Math.round(state.scale * 100)}%`;
}

function clearCanvas() {
  activeImageObj = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ── Rendering Engine ────────────────────────────────────────────
function renderAll() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!activeImageObj) return;

  ctx.save();
  // Apply viewport Pan & Zoom
  ctx.translate(state.panX, state.panY);
  ctx.scale(state.scale, state.scale);

  // 1. Draw Image
  ctx.drawImage(activeImageObj, 0, 0);

  // 2. Draw Model Predictions Layer (if toggled)
  if (state.showModelOverlay) {
    drawModelPredictions();
  }

  // 3. Draw Ground Truth Bounding Boxes
  drawGroundTruthBoxes();

  // 4. Draw Current Drawing Box
  if (state.isDrawing && state.drawStart) {
    drawDrawingBox();
  }

  ctx.restore();
}

function drawModelPredictions() {
  const curFilename = getCurrentFilename();
  if (!curFilename) return;
  const preds = state.modelPredictions[curFilename] || [];

  ctx.save();
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 2 / state.scale;

  for (const pred of preds) {
    const [x1, y1, x2, y2] = pred.bbox;
    const w = x2 - x1;
    const h = y2 - y1;

    // Dotted cyan outline
    ctx.strokeStyle = "#06b6d4";
    ctx.strokeRect(x1, y1, w, h);

    // Label with confidence
    const conf = pred.confidence ? `${Math.round(pred.confidence * 100)}%` : "";
    const text = `🤖 ${pred.label} ${conf}`.trim();

    ctx.font = `bold ${Math.max(12, 13 / state.scale)}px 'Inter', sans-serif`;
    const textMetrics = ctx.measureText(text);
    const textHeight = 16 / state.scale;
    const textWidth = textMetrics.width + 8 / state.scale;

    ctx.fillStyle = "rgba(6, 182, 212, 0.85)";
    ctx.fillRect(x1, Math.max(0, y1 - textHeight), textWidth, textHeight);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, x1 + 4 / state.scale, Math.max(textHeight - 4 / state.scale, y1 - 4 / state.scale));
  }
  ctx.restore();
}

function drawGroundTruthBoxes() {
  const boxes = getCurrentBoxes();

  for (const box of boxes) {
    const isSelected = box.id === state.selectedBoxId;
    const color = getLabelColor(box.label);
    const [x1, y1, x2, y2] = box.bbox;
    const w = x2 - x1;
    const h = y2 - y1;

    ctx.save();

    // Box fill
    ctx.fillStyle = isSelected ? "rgba(59, 130, 246, 0.22)" : hexToRgba(color, 0.15);
    ctx.fillRect(x1, y1, w, h);

    // Box stroke
    ctx.strokeStyle = isSelected ? "#3B82F6" : color;
    ctx.lineWidth = (isSelected ? 3.5 : 2.5) / state.scale;
    ctx.strokeRect(x1, y1, w, h);

    // Box Label Tag
    const labelText = box.label || "unlabeled";
    const fontSize = Math.max(12, 13 / state.scale);
    ctx.font = `600 ${fontSize}px 'Inter', sans-serif`;
    const textMetrics = ctx.measureText(labelText);
    const tagH = 18 / state.scale;
    const tagW = textMetrics.width + 12 / state.scale;
    const tagY = Math.max(0, y1 - tagH);

    // Tag background
    ctx.fillStyle = isSelected ? "#2563EB" : color;
    ctx.fillRect(x1, tagY, tagW, tagH);

    // Tag text
    ctx.fillStyle = "#ffffff";
    ctx.fillText(labelText, x1 + 6 / state.scale, tagY + tagH - 4 / state.scale);

    // Draw Resize Handles if Selected
    if (isSelected) {
      drawResizeHandles(box);
    }

    ctx.restore();
  }
}

function drawResizeHandles(box) {
  const [x1, y1, x2, y2] = box.bbox;
  const handleSize = 8 / state.scale;
  const half = handleSize / 2;

  const handles = [
    { name: "tl", x: x1, y: y1 },
    { name: "tr", x: x2, y: y1 },
    { name: "br", x: x2, y: y2 },
    { name: "bl", x: x1, y: y2 },
    { name: "t",  x: (x1 + x2) / 2, y: y1 },
    { name: "r",  x: x2, y: (y1 + y2) / 2 },
    { name: "b",  x: (x1 + x2) / 2, y: y2 },
    { name: "l",  x: x1, y: (y1 + y2) / 2 },
  ];

  for (const h of handles) {
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1.5 / state.scale;
    ctx.fillRect(h.x - half, h.y - half, handleSize, handleSize);
    ctx.strokeRect(h.x - half, h.y - half, handleSize, handleSize);
  }
}

function drawDrawingBox() {
  const { startX, startY, currentX, currentY } = state.drawStart;
  const x1 = Math.min(startX, currentX);
  const y1 = Math.min(startY, currentY);
  const x2 = Math.max(startX, currentX);
  const y2 = Math.max(startY, currentY);
  const w = x2 - x1;
  const h = y2 - y1;

  ctx.save();
  ctx.fillStyle = "rgba(59, 130, 246, 0.2)";
  ctx.fillRect(x1, y1, w, h);
  ctx.strokeStyle = "#3B82F6";
  ctx.lineWidth = 2 / state.scale;
  ctx.setLineDash([4, 4]);
  ctx.strokeRect(x1, y1, w, h);

  // Dimension tag
  const dimText = `${Math.round(w)} × ${Math.round(h)} px`;
  ctx.font = `500 ${11 / state.scale}px 'JetBrains Mono', monospace`;
  ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
  const m = ctx.measureText(dimText);
  ctx.fillRect(x1, y2 + 4 / state.scale, m.width + 8 / state.scale, 16 / state.scale);
  ctx.fillStyle = "#93c5fd";
  ctx.fillText(dimText, x1 + 4 / state.scale, y2 + 16 / state.scale);

  ctx.restore();
}

function hexToRgba(hex, alpha) {
  let c;
  if (/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)) {
    c = hex.substring(1).split("");
    if (c.length === 3) {
      c = [c[0], c[0], c[1], c[1], c[2], c[2]];
    }
    c = "0x" + c.join("");
    return `rgba(${[(c >> 16) & 255, (c >> 8) & 255, c & 255].join(",")},${alpha})`;
  }
  return `rgba(59, 130, 246, ${alpha})`;
}

// ── Handle Detection ────────────────────────────────────────────
function getHandleUnderCursor(imgX, imgY, box) {
  if (!box) return null;
  const [x1, y1, x2, y2] = box.bbox;
  const tolerance = 9 / state.scale;

  const handles = {
    tl: { x: x1, y: y1 },
    tr: { x: x2, y: y1 },
    br: { x: x2, y: y2 },
    bl: { x: x1, y: y2 },
    t:  { x: (x1 + x2) / 2, y: y1 },
    r:  { x: x2, y: (y1 + y2) / 2 },
    b:  { x: (x1 + x2) / 2, y: y2 },
    l:  { x: x1, y: (y1 + y2) / 2 },
  };

  for (const [name, pos] of Object.entries(handles)) {
    if (Math.abs(imgX - pos.x) <= tolerance && Math.abs(imgY - pos.y) <= tolerance) {
      return name;
    }
  }
  return null;
}

function getCursorForHandle(handle) {
  switch (handle) {
    case "tl":
    case "br": return "nwse-resize";
    case "tr":
    case "bl": return "nesw-resize";
    case "t":
    case "b":  return "ns-resize";
    case "l":
    case "r":  return "ew-resize";
    default:   return "default";
  }
}

// ── Mouse & Interaction Events ──────────────────────────────────
function setupEventListeners() {
  // Directory & Navigation
  loadDirBtn.addEventListener("click", () => loadDirectory(dirInput.value));
  dirInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadDirectory(dirInput.value);
  });
  prevImgBtn.addEventListener("click", () => loadImage(state.currentIndex - 1));
  nextImgBtn.addEventListener("click", () => loadImage(state.currentIndex + 1));

  // Save Annotations
  saveBtn.addEventListener("click", saveGroundTruth);

  // Tools Switch
  toolSelectBtn.addEventListener("click", () => setTool("select"));
  toolDrawBtn.addEventListener("click", () => setTool("draw"));

  // Zoom Controls
  zoomInBtn.addEventListener("click", () => zoomCanvas(1.2));
  zoomOutBtn.addEventListener("click", () => zoomCanvas(0.8));
  zoomFitBtn.addEventListener("click", () => {
    fitImageToViewport();
    renderAll();
  });

  // Undo / Redo
  undoBtn.addEventListener("click", undo);
  redoBtn.addEventListener("click", redo);

  // Delete Box
  deleteBoxBtn.addEventListener("click", deleteSelectedBox);
  deleteActiveBoxBtn.addEventListener("click", deleteSelectedBox);
  clearCurrentBoxesBtn.addEventListener("click", clearCurrentBoxes);

  // Model Predictions Overlay Toggle & Selection
  showModelOverlayCheck.addEventListener("change", (e) => {
    state.showModelOverlay = e.target.checked;
    modelLegendCard.classList.toggle("hidden", !state.showModelOverlay);
    renderAll();
    updateModelLegend();
  });
  modelPredictionSelect.addEventListener("change", (e) => {
    state.currentPredictionFile = e.target.value;
    loadPredictionFile(e.target.value);
  });
  importPredictionsBtn.addEventListener("click", importModelPredictionsAsLabels);

  // Filter Tabs
  document.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.filter = btn.dataset.filter;
      updateImageListUI();
    });
  });

  // Canvas Mouse Events
  canvas.addEventListener("mousedown", onCanvasMouseDown);
  window.addEventListener("mousemove", onCanvasMouseMove);
  window.addEventListener("mouseup", onCanvasMouseUp);
  canvas.addEventListener("wheel", onCanvasWheel, { passive: false });

  // Inspector coordinate inputs
  [coordX1, coordY1, coordX2, coordY2].forEach((input) => {
    input.addEventListener("change", updateBoxFromCoordInputs);
  });

  // Label Autocomplete Input
  labelInput.addEventListener("input", onLabelInput);
  labelInput.addEventListener("keydown", onLabelKeydown);
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".autocomplete-wrapper")) {
      autocompleteList.classList.add("hidden");
    }
  });

  // Help Modal
  helpBtn.addEventListener("click", () => helpModal.classList.remove("hidden"));
  closeHelpBtn.addEventListener("click", () => helpModal.classList.add("hidden"));
  gotItBtn.addEventListener("click", () => helpModal.classList.add("hidden"));

  // Global Keyboard Shortcuts
  window.addEventListener("keydown", handleGlobalShortcuts);
  window.addEventListener("keyup", (e) => {
    if (e.code === "Space") {
      state.spacePressed = false;
      canvas.style.cursor = state.activeTool === "draw" ? "crosshair" : "default";
    }
  });
}

function setTool(tool) {
  state.activeTool = tool;
  toolSelectBtn.classList.toggle("active", tool === "select");
  toolDrawBtn.classList.toggle("active", tool === "draw");
  canvas.style.cursor = tool === "draw" ? "crosshair" : "default";
}

function zoomCanvas(factor, centerScreenX = canvas.width / 2, centerScreenY = canvas.height / 2) {
  const oldScale = state.scale;
  const newScale = Math.max(0.1, Math.min(10.0, oldScale * factor));
  if (oldScale === newScale) return;

  // Zoom centered on cursor/point
  state.panX = centerScreenX - (centerScreenX - state.panX) * (newScale / oldScale);
  state.panY = centerScreenY - (centerScreenY - state.panY) * (newScale / oldScale);
  state.scale = newScale;
  zoomLevelText.textContent = `${Math.round(state.scale * 100)}%`;
  renderAll();
}

function onCanvasWheel(e) {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
  zoomCanvas(zoomFactor, mouseX, mouseY);
}

function onCanvasMouseDown(e) {
  if (!activeImageObj) return;

  // Middle click or Space + Left click = Pan
  if (e.button === 1 || (e.button === 0 && state.spacePressed)) {
    state.isPanning = true;
    state.panStartX = e.clientX - state.panX;
    state.panStartY = e.clientY - state.panY;
    canvas.style.cursor = "grabbing";
    return;
  }

  if (e.button !== 0) return; // Only process left click for annotations

  const { x, y } = screenToImageCoords(e.clientX, e.clientY);
  const selectedBox = getSelectedBox();

  // Check if clicked on a resize handle of the selected box
  if (selectedBox) {
    const handle = getHandleUnderCursor(x, y, selectedBox);
    if (handle) {
      state.isResizing = true;
      state.activeHandle = handle;
      state.resizeStartBox = JSON.parse(JSON.stringify(selectedBox));
      return;
    }
  }

  // Draw Tool Active
  if (state.activeTool === "draw") {
    state.isDrawing = true;
    state.drawStart = { startX: x, startY: y, currentX: x, currentY: y };
    return;
  }

  // Select Tool Active: Check if clicked inside any bounding box
  const boxes = getCurrentBoxes();
  let clickedBox = null;
  // Iterate in reverse so top-most box is selected
  for (let i = boxes.length - 1; i >= 0; i--) {
    const b = boxes[i];
    const [x1, y1, x2, y2] = b.bbox;
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
      clickedBox = b;
      break;
    }
  }

  if (clickedBox) {
    selectBox(clickedBox.id);
    state.isDraggingBox = true;
    const [x1, y1] = clickedBox.bbox;
    state.dragOffset = { offsetX: x - x1, offsetY: y - y1 };
    canvas.style.cursor = "move";
  } else {
    // Clicked empty canvas space
    selectBox(null);
  }
}

function onCanvasMouseMove(e) {
  const rect = canvas.getBoundingClientRect();
  const screenX = e.clientX - rect.left;
  const screenY = e.clientY - rect.top;

  // Track coordinates in footer
  if (activeImageObj) {
    const { x, y } = screenToImageCoords(e.clientX, e.clientY);
    const clampedX = Math.max(0, Math.min(activeImageObj.naturalWidth, Math.round(x)));
    const clampedY = Math.max(0, Math.min(activeImageObj.naturalHeight, Math.round(y)));
    mouseCoordsEl.textContent = `X: ${clampedX}, Y: ${clampedY}`;
  }

  // Panning
  if (state.isPanning) {
    state.panX = e.clientX - state.panStartX;
    state.panY = e.clientY - state.panStartY;
    renderAll();
    return;
  }

  // Drawing Box
  if (state.isDrawing && state.drawStart) {
    const { x, y } = screenToImageCoords(e.clientX, e.clientY);
    state.drawStart.currentX = Math.max(0, Math.min(activeImageObj.naturalWidth, x));
    state.drawStart.currentY = Math.max(0, Math.min(activeImageObj.naturalHeight, y));
    renderAll();
    return;
  }

  // Resizing Box
  if (state.isResizing && state.activeHandle && state.resizeStartBox) {
    const { x, y } = screenToImageCoords(e.clientX, e.clientY);
    const box = getSelectedBox();
    if (!box) return;

    let [x1, y1, x2, y2] = state.resizeStartBox.bbox;
    const h = state.activeHandle;

    if (h.includes("l")) x1 = Math.min(x, x2 - 5);
    if (h.includes("r")) x2 = Math.max(x, x1 + 5);
    if (h.includes("t")) y1 = Math.min(y, y2 - 5);
    if (h.includes("b")) y2 = Math.max(y, y1 + 5);

    box.bbox = [
      Math.max(0, Math.round(x1)),
      Math.max(0, Math.round(y1)),
      Math.min(activeImageObj.naturalWidth, Math.round(x2)),
      Math.min(activeImageObj.naturalHeight, Math.round(y2)),
    ];

    updateInspector();
    renderAll();
    return;
  }

  // Dragging / Moving Box
  if (state.isDraggingBox && state.dragOffset) {
    const { x, y } = screenToImageCoords(e.clientX, e.clientY);
    const box = getSelectedBox();
    if (!box) return;

    const w = box.bbox[2] - box.bbox[0];
    const h = box.bbox[3] - box.bbox[1];

    let newX1 = x - state.dragOffset.offsetX;
    let newY1 = y - state.dragOffset.offsetY;

    // Clamp within image bounds
    newX1 = Math.max(0, Math.min(activeImageObj.naturalWidth - w, newX1));
    newY1 = Math.max(0, Math.min(activeImageObj.naturalHeight - h, newY1));

    box.bbox = [
      Math.round(newX1),
      Math.round(newY1),
      Math.round(newX1 + w),
      Math.round(newY1 + h),
    ];

    updateInspector();
    renderAll();
    return;
  }

  // Hover cursor changes
  if (state.activeTool === "select" && !state.isPanning) {
    const { x, y } = screenToImageCoords(e.clientX, e.clientY);
    const selectedBox = getSelectedBox();
    if (selectedBox) {
      const handle = getHandleUnderCursor(x, y, selectedBox);
      if (handle) {
        canvas.style.cursor = getCursorForHandle(handle);
        return;
      }
    }

    // Check if hovering over any box
    const boxes = getCurrentBoxes();
    let isOverBox = false;
    for (let i = boxes.length - 1; i >= 0; i--) {
      const [x1, y1, x2, y2] = boxes[i].bbox;
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        isOverBox = true;
        break;
      }
    }
    canvas.style.cursor = isOverBox ? "move" : "default";
  }
}

function onCanvasMouseUp(e) {
  if (state.isPanning) {
    state.isPanning = false;
    canvas.style.cursor = state.activeTool === "draw" ? "crosshair" : "default";
    return;
  }

  // Finalize Drawing
  if (state.isDrawing && state.drawStart) {
    state.isDrawing = false;
    const { startX, startY, currentX, currentY } = state.drawStart;
    state.drawStart = null;

    let x1 = Math.min(startX, currentX);
    let y1 = Math.min(startY, currentY);
    let x2 = Math.max(startX, currentX);
    let y2 = Math.max(startY, currentY);

    const w = x2 - x1;
    const h = y2 - y1;

    // Minimum size threshold (ignore accidental clicks)
    if (w > 6 && h > 6) {
      const boxes = getCurrentBoxes();
      // Default label is last used label or first suggested
      const defaultLabel = labelInput.value.trim() || (state.vocabulary[0] || "tomato");
      const newBox = {
        id: "box_" + Date.now().toString(36) + "_" + Math.floor(Math.random() * 1000),
        label: defaultLabel,
        bbox: [
          Math.max(0, Math.round(x1)),
          Math.max(0, Math.round(y1)),
          Math.min(activeImageObj.naturalWidth, Math.round(x2)),
          Math.min(activeImageObj.naturalHeight, Math.round(y2)),
        ],
      };

      boxes.push(newBox);
      selectBox(newBox.id);
      pushHistory();
      setDirty(true);
      updateAnnotationsSidebar();
      updateImageListUI();
      renderAll();

      // Automatically focus label input for rapid tagging!
      setTimeout(() => {
        labelInput.focus();
        labelInput.select();
      }, 50);
    } else {
      renderAll();
    }
    return;
  }

  // Finalize Resize or Drag
  if (state.isResizing || state.isDraggingBox) {
    state.isResizing = false;
    state.isDraggingBox = false;
    state.activeHandle = null;
    state.resizeStartBox = null;
    state.dragOffset = null;
    pushHistory();
    setDirty(true);
    updateAnnotationsSidebar();
    renderAll();
  }
}

// ── Box Selection & Inspector ───────────────────────────────────
function selectBox(boxId) {
  state.selectedBoxId = boxId;
  updateInspector();
  renderAll();

  // Scroll active item in annotations list
  if (boxId) {
    const itemEl = document.querySelector(`.annotation-item[data-id="${boxId}"]`);
    if (itemEl) itemEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

function updateInspector() {
  const box = getSelectedBox();
  if (!box) {
    boxEditorContent.classList.add("disabled");
    selectedBoxIdBadge.textContent = "None";
    labelInput.value = "";
    coordX1.value = "";
    coordY1.value = "";
    coordX2.value = "";
    coordY2.value = "";
    coordW.value = "";
    coordH.value = "";
    return;
  }

  boxEditorContent.classList.remove("disabled");
  selectedBoxIdBadge.textContent = box.id.split("_")[1] || box.id;
  labelInput.value = box.label || "";

  const [x1, y1, x2, y2] = box.bbox;
  coordX1.value = x1;
  coordY1.value = y1;
  coordX2.value = x2;
  coordY2.value = y2;
  coordW.value = x2 - x1;
  coordH.value = y2 - y1;
}

function updateBoxFromCoordInputs() {
  const box = getSelectedBox();
  if (!box || !activeImageObj) return;

  const x1 = Math.max(0, parseInt(coordX1.value) || 0);
  const y1 = Math.max(0, parseInt(coordY1.value) || 0);
  const x2 = Math.min(activeImageObj.naturalWidth, parseInt(coordX2.value) || x1 + 10);
  const y2 = Math.min(activeImageObj.naturalHeight, parseInt(coordY2.value) || y1 + 10);

  box.bbox = [Math.min(x1, x2 - 5), Math.min(y1, y2 - 5), Math.max(x2, x1 + 5), Math.max(y2, y1 + 5)];
  pushHistory();
  setDirty(true);
  updateInspector();
  updateAnnotationsSidebar();
  renderAll();
}

function deleteSelectedBox() {
  const box = getSelectedBox();
  if (!box) return;

  const boxes = getCurrentBoxes();
  state.annotations[getCurrentFilename()] = boxes.filter((b) => b.id !== box.id);
  state.selectedBoxId = null;
  pushHistory();
  setDirty(true);
  updateInspector();
  updateAnnotationsSidebar();
  updateImageListUI();
  renderAll();
  showToast("Bounding box deleted", "info");
}

function clearCurrentBoxes() {
  const filename = getCurrentFilename();
  if (!filename) return;
  const boxes = getCurrentBoxes();
  if (boxes.length === 0) return;

  if (confirm(`Remove all ${boxes.length} bounding boxes on this image?`)) {
    state.annotations[filename] = [];
    state.selectedBoxId = null;
    pushHistory();
    setDirty(true);
    updateInspector();
    updateAnnotationsSidebar();
    updateImageListUI();
    renderAll();
    showToast("Cleared annotations on image", "info");
  }
}

// ── Annotations Sidebar List ────────────────────────────────────
function updateAnnotationsSidebar() {
  const boxes = getCurrentBoxes();
  currentBoxesCount.textContent = boxes.length;
  annotationsList.innerHTML = "";

  if (boxes.length === 0) {
    annotationsList.innerHTML = `<div class="empty-state-small">No bounding boxes drawn yet.<br>Click & drag to label!</div>`;
    return;
  }

  boxes.forEach((box, idx) => {
    const item = document.createElement("div");
    item.className = `annotation-item ${box.id === state.selectedBoxId ? "active" : ""}`;
    item.dataset.id = box.id;

    const color = getLabelColor(box.label);
    const [x1, y1, x2, y2] = box.bbox;

    item.innerHTML = `
      <div class="annot-info">
        <span class="annot-color-dot" style="background-color: ${color}"></span>
        <div>
          <div class="annot-label">${escapeHtml(box.label)}</div>
          <div class="annot-coords">[${x1}, ${y1}, ${x2}, ${y2}]</div>
        </div>
      </div>
      <button class="annot-del-btn" title="Delete box">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
      </button>
    `;

    item.addEventListener("click", (e) => {
      if (e.target.closest(".annot-del-btn")) {
        selectBox(box.id);
        deleteSelectedBox();
      } else {
        selectBox(box.id);
      }
    });

    annotationsList.appendChild(item);
  });
}

// ── Image Gallery Sidebar ───────────────────────────────────────
function updateImageListUI() {
  totalImagesBadge.textContent = state.images.length;
  imageListEl.innerHTML = "";

  if (state.images.length === 0) {
    imageListEl.innerHTML = `<div class="empty-state">No images found.</div>`;
    return;
  }

  const filtered = state.images.filter((img, idx) => {
    const count = (state.annotations[img.filename] || []).length;
    if (state.filter === "labeled") return count > 0;
    if (state.filter === "unlabeled") return count === 0;
    return true;
  });

  filtered.forEach((img) => {
    const origIdx = state.images.findIndex((i) => i.filename === img.filename);
    const boxCount = (state.annotations[img.filename] || []).length;

    const card = document.createElement("div");
    card.className = `image-card ${origIdx === state.currentIndex ? "active" : ""}`;
    card.innerHTML = `
      <img class="thumb-preview" src="/api/image?path=${encodeURIComponent(img.path)}" loading="lazy">
      <div class="image-meta">
        <div class="image-name" title="${img.filename}">${escapeHtml(img.filename)}</div>
        <div class="image-sub">
          <span>${img.width}×${img.height}</span>
          <span class="box-badge ${boxCount > 0 ? "labeled" : ""}">${boxCount} boxes</span>
        </div>
      </div>
    `;

    card.addEventListener("click", () => loadImage(origIdx));
    imageListEl.appendChild(card);
  });
}

// ── Autocomplete & Chips ────────────────────────────────────────
function renderQuickChips() {
  // Common fridge ingredients and items for instant 1-click tagging
  const common = [
    "tomato", "iceberg lettuce", "musk melon", "water bottle", "cucumber",
    "bell pepper", "green capsicum", "yellow capsicum", "milk", "onion",
    "carrot", "lemon", "apple", "jalapeno", "cheese", "yogurt", "eggs",
    "zucchini", "spinach", "broccoli"
  ];

  quickChipsList.innerHTML = "";
  common.forEach((food) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = food;
    chip.addEventListener("click", () => {
      applyLabelToSelectedBox(food);
    });
    quickChipsList.appendChild(chip);
  });
}

function onLabelInput() {
  const query = labelInput.value.toLowerCase().trim();
  const box = getSelectedBox();
  if (box) {
    box.label = labelInput.value.trim();
    setDirty(true);
    updateAnnotationsSidebar();
    renderAll();
  }

  if (!query) {
    autocompleteList.classList.add("hidden");
    return;
  }

  const matches = state.vocabulary
    .filter((item) => item.toLowerCase().includes(query))
    .slice(0, 10);

  if (matches.length === 0) {
    autocompleteList.classList.add("hidden");
    return;
  }

  autocompleteList.innerHTML = "";
  matches.forEach((item, i) => {
    const div = document.createElement("div");
    div.className = `autocomplete-item ${i === 0 ? "highlighted" : ""}`;
    div.textContent = item;
    div.addEventListener("mousedown", (e) => {
      e.preventDefault();
      applyLabelToSelectedBox(item);
      autocompleteList.classList.add("hidden");
    });
    autocompleteList.appendChild(div);
  });
  autocompleteList.classList.remove("hidden");
}

function onLabelKeydown(e) {
  if (e.key === "Enter" || e.key === "Tab") {
    const highlighted = autocompleteList.querySelector(".autocomplete-item.highlighted");
    if (highlighted && !autocompleteList.classList.contains("hidden")) {
      e.preventDefault();
      applyLabelToSelectedBox(highlighted.textContent);
      autocompleteList.classList.add("hidden");
    } else if (labelInput.value.trim()) {
      applyLabelToSelectedBox(labelInput.value.trim());
      autocompleteList.classList.add("hidden");
    }
  } else if (e.key === "Escape") {
    autocompleteList.classList.add("hidden");
  }
}

function applyLabelToSelectedBox(label) {
  const box = getSelectedBox();
  if (!box) return;
  const cleanLabel = label.trim();
  box.label = cleanLabel;
  labelInput.value = cleanLabel;

  // If label is not yet in vocabulary, add it locally and persist to backend
  if (!state.vocabulary.some((v) => v.toLowerCase() === cleanLabel.toLowerCase())) {
    state.vocabulary.unshift(cleanLabel);
    fetch("/api/vocabulary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: cleanLabel }),
    }).catch((err) => console.warn("Failed to persist custom label:", err));
  }

  pushHistory();
  setDirty(true);
  updateAnnotationsSidebar();
  renderAll();
}

// ── Model Predictions Import ────────────────────────────────────
function importModelPredictionsAsLabels() {
  const curFilename = getCurrentFilename();
  if (!curFilename) return;

  const preds = state.modelPredictions[curFilename] || [];
  if (preds.length === 0) {
    showToast("No model predictions available for this image", "info");
    return;
  }

  if (confirm(`Import ${preds.length} model predictions as ground-truth labels for this image?`)) {
    const boxes = getCurrentBoxes();
    preds.forEach((pred) => {
      boxes.push({
        id: "box_" + Date.now().toString(36) + "_" + Math.floor(Math.random() * 1000),
        label: pred.label,
        bbox: [round(pred.bbox[0]), round(pred.bbox[1]), round(pred.bbox[2]), round(pred.bbox[3])],
      });
    });

    pushHistory();
    setDirty(true);
    updateAnnotationsSidebar();
    updateImageListUI();
    renderAll();
    showToast(`Imported ${preds.length} detections as editable labels`, "success");
  }
}

function updateModelLegend() {
  const curFilename = getCurrentFilename();
  if (!curFilename) return;
  const preds = state.modelPredictions[curFilename] || [];
  modelBoxCountBadge.textContent = `${preds.length} detected`;
  modelDetectionsList.innerHTML = "";

  preds.forEach((p) => {
    const div = document.createElement("div");
    div.className = "model-item";
    const conf = p.confidence ? `${Math.round(p.confidence * 100)}%` : "";
    div.innerHTML = `<span>${escapeHtml(p.label)}</span><span class="model-conf">${conf}</span>`;
    modelDetectionsList.appendChild(div);
  });
}

// ── Save Ground Truth ───────────────────────────────────────────
async function saveGroundTruth() {
  if (!state.currentDirectory) {
    showToast("No active directory to save to", "error");
    return;
  }

  saveBtn.classList.add("loading");
  try {
    const res = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        directory_path: state.currentDirectory,
        annotations: state.annotations,
        image_metadata: state.imageMetadata,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to save annotations");
    }

    const result = await res.json();
    setDirty(false);
    showToast(`Saved ${result.total_boxes} boxes to annotations.json and ground_truth_bounding_boxes.json!`, "success");
  } catch (err) {
    console.error("Save error:", err);
    showToast(err.message, "error");
  } finally {
    saveBtn.classList.remove("loading");
  }
}

// ── Global Keyboard Shortcuts ───────────────────────────────────
function handleGlobalShortcuts(e) {
  // If user is currently typing in an input, do not trigger single-key tools
  const isInputActive = ["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName);

  // Ctrl+S / Cmd+S: Save Annotations
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    saveGroundTruth();
    return;
  }

  // Ctrl+Z: Undo
  if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === "z") {
    e.preventDefault();
    undo();
    return;
  }

  // Ctrl+Y or Ctrl+Shift+Z: Redo
  if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === "y" || (e.shiftKey && e.key.toLowerCase() === "z"))) {
    e.preventDefault();
    redo();
    return;
  }

  if (isInputActive) return;

  // Space for Canvas Pan
  if (e.code === "Space") {
    e.preventDefault();
    state.spacePressed = true;
    canvas.style.cursor = "grab";
    return;
  }

  // Tools
  if (e.key.toLowerCase() === "v" || e.key === "Escape") {
    setTool("select");
    return;
  }
  if (e.key.toLowerCase() === "b") {
    setTool("draw");
    return;
  }

  // Delete / Backspace
  if (e.key === "Delete" || e.key === "Backspace") {
    deleteSelectedBox();
    return;
  }

  // Prev / Next Image Navigation
  if (e.key === "[" || e.key === "ArrowLeft") {
    if (state.currentIndex > 0) loadImage(state.currentIndex - 1);
    return;
  }
  if (e.key === "]" || e.key === "ArrowRight") {
    if (state.currentIndex < state.images.length - 1) loadImage(state.currentIndex + 1);
    return;
  }

  // Fit to screen
  if (e.key === "0") {
    fitImageToViewport();
    renderAll();
    return;
  }

  // Nudge selected box with Arrow keys
  const box = getSelectedBox();
  if (box && ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) {
    e.preventDefault();
    const step = e.shiftKey ? 10 : 1;
    let [x1, y1, x2, y2] = box.bbox;
    if (e.key === "ArrowUp") { y1 -= step; y2 -= step; }
    if (e.key === "ArrowDown") { y1 += step; y2 += step; }
    if (e.key === "ArrowLeft") { x1 -= step; x2 -= step; }
    if (e.key === "ArrowRight") { x1 += step; x2 += step; }
    box.bbox = [Math.max(0, x1), Math.max(0, y1), x2, y2];
    pushHistory();
    setDirty(true);
    updateInspector();
    renderAll();
  }
}

// ── Utility ─────────────────────────────────────────────────────
function round(num, decimals = 1) {
  return Number(Math.round(num + "e" + decimals) + "e-" + decimals);
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// Bootstrap
window.addEventListener("DOMContentLoaded", initApp);
