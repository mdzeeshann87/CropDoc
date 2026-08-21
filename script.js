(() => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const dzEmpty = document.getElementById("dz-empty");
  const dzPreview = document.getElementById("dz-preview");
  const previewImg = document.getElementById("preview-img");
  const scanReadout = document.getElementById("scan-readout");
  const analyzeBtn = document.getElementById("analyze-btn");
  const form = document.getElementById("scan-form");
  const formError = document.getElementById("form-error");
  const cropSelect = document.getElementById("crop-select");
  const resultsSection = document.getElementById("results");
  const scanAgainBtn = document.getElementById("scan-again");

  let selectedFile = null;

  const showError = (msg) => {
    formError.textContent = msg;
    formError.classList.remove("hidden");
  };
  const clearError = () => formError.classList.add("hidden");

  const setFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      showError("Please choose an image file (JPG, PNG, or WEBP).");
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      showError("That file is over 8MB — try a smaller photo.");
      return;
    }
    clearError();
    selectedFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    dzEmpty.classList.add("hidden");
    dzPreview.classList.remove("hidden");
    analyzeBtn.disabled = false;
  };

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    setFile(file);
  });

  const readouts = ["ANALYZING PIXEL FIELD…", "MAPPING CHLOROPHYLL…", "SCORING LESION TEXTURE…"];

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) return;
    clearError();

    analyzeBtn.disabled = true;
    analyzeBtn.classList.add("loading");
    dzPreview.classList.add("scanning");
    let ri = 0;
    scanReadout.textContent = readouts[0];
    const readoutTimer = setInterval(() => {
      ri = (ri + 1) % readouts.length;
      scanReadout.textContent = readouts[ri];
    }, 550);

    const fd = new FormData();
    fd.append("image", selectedFile);
    fd.append("crop", cropSelect.value);

    const minWait = new Promise((res) => setTimeout(res, 1400)); // let the scan animation read as real work

    try {
      const [resp] = await Promise.all([fetch("/api/analyze", { method: "POST", body: fd }), minWait]);
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Something went wrong analyzing that image.");
      renderResults(data);
    } catch (err) {
      showError(err.message || "Could not reach the analysis service.");
    } finally {
      clearInterval(readoutTimer);
      dzPreview.classList.remove("scanning");
      analyzeBtn.disabled = false;
      analyzeBtn.classList.remove("loading");
    }
  });

  const sevLabel = { none: "No disease detected", moderate: "Moderate severity", severe: "Severe — act promptly" };

  function fillList(el, items) {
    el.innerHTML = "";
    items.forEach((t) => {
      const li = document.createElement("li");
      li.textContent = t;
      el.appendChild(li);
    });
  }

  function renderResults(data) {
    document.getElementById("result-img").src = data.image_url;
    document.getElementById("result-crop-label").textContent =
      `${data.crop.toUpperCase()} · ${data.engine === "deep_model" ? "CNN MODEL" : "SIGNATURE ANALYSIS"}`;
    document.getElementById("result-name").textContent = data.top_result.name;
    document.getElementById("result-summary").textContent = data.top_result.summary;
    document.getElementById("cause-text").textContent = data.top_result.cause;

    const fill = document.getElementById("confidence-fill");
    const num = document.getElementById("confidence-num");
    requestAnimationFrame(() => { fill.style.width = data.top_result.confidence + "%"; });
    num.textContent = data.top_result.confidence + "%";

    const chip = document.getElementById("severity-chip");
    chip.dataset.level = data.top_result.severity;
    chip.textContent = sevLabel[data.top_result.severity] || data.top_result.severity;

    fillList(document.getElementById("symptom-list"), data.top_result.symptoms);
    fillList(document.getElementById("treatment-list"), data.top_result.treatment);
    fillList(document.getElementById("prevention-list"), data.top_result.prevention);

    const altRow = document.getElementById("alt-row");
    altRow.innerHTML = "";
    data.alternatives.forEach((a) => {
      const pill = document.createElement("div");
      pill.className = "alt-pill";
      pill.innerHTML = `${a.name} <span>${a.confidence}%</span>`;
      altRow.appendChild(pill);
    });

    const strip = document.getElementById("feature-strip");
    strip.innerHTML = "";
    const labels = { green: "chlorophyll", brown: "browning", yellow: "chlorosis", spots: "lesion texture" };
    Object.entries(data.features).forEach(([k, v]) => {
      if (!(k in labels)) return;
      const chip2 = document.createElement("span");
      chip2.className = "feature-chip";
      chip2.textContent = `${labels[k]} ${Math.round(v * 100)}%`;
      strip.appendChild(chip2);
    });

    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  scanAgainBtn.addEventListener("click", () => {
    resultsSection.classList.add("hidden");
    document.getElementById("lab").scrollIntoView({ behavior: "smooth", block: "start" });
  });
})();
