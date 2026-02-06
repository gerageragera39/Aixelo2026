const apiBase = ""; // если фронтенд и backend на одном хосте, оставьте пустым

const uploadForm = document.getElementById("upload-form");
const uploadFile = document.getElementById("upload-file");
const uploadResult = document.getElementById("upload-result");

const runPredictBtn = document.getElementById("run-predict");
const predictLogs = document.getElementById("predict-logs");

const loadPredictionsBtn = document.getElementById("load-predictions");
const predictionsTableBody = document.querySelector("#predictions-table tbody");

const predictNewForm = document.getElementById("predict-new-form");
const predictNewFile = document.getElementById("predict-new-file");
const predictNewActual = document.getElementById("predict-new-actual");
const predictNewResult = document.getElementById("predict-new-result");

const pretty = (obj) => JSON.stringify(obj, null, 2);

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!uploadFile.files.length) return;

  uploadResult.textContent = "⏳ Uploading...";
  const formData = new FormData();
  formData.append("file", uploadFile.files[0]);

  try {
    const res = await fetch(`${apiBase}/upload_cif/`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    uploadResult.textContent = pretty(data);
  } catch (err) {
    uploadResult.textContent = `❌ ${err}`;
  }
});

runPredictBtn.addEventListener("click", async () => {
  predictLogs.textContent = "⏳ Запуск предсказаний...\n";
  try {
    const res = await fetch(`${apiBase}/predict/`, { method: "POST" });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      predictLogs.textContent += decoder.decode(value);
      predictLogs.scrollTop = predictLogs.scrollHeight;
    }
  } catch (err) {
    predictLogs.textContent += `\n❌ ${err}`;
  }
});

loadPredictionsBtn.addEventListener("click", async () => {
  predictionsTableBody.innerHTML = "";
  try {
    const res = await fetch(`${apiBase}/predictions_data/`);
    const data = await res.json();

    data.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.cif ?? ""}</td>
        <td>${row.predicted ?? ""}</td>
        <td>${row.actual ?? ""}</td>
      `;
      predictionsTableBody.appendChild(tr);
    });
  } catch (err) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="3">❌ ${err}</td>`;
    predictionsTableBody.appendChild(tr);
  }
});

predictNewForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!predictNewFile.files.length) return;

  predictNewResult.textContent = "⏳ Predicting...";
  const formData = new FormData();
  formData.append("file", predictNewFile.files[0]);

  if (predictNewActual.value.trim() !== "") {
    formData.append("actual", predictNewActual.value.trim());
  }

  try {
    const res = await fetch(`${apiBase}/predict_new_cif/`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    predictNewResult.textContent = pretty(data);
  } catch (err) {
    predictNewResult.textContent = `❌ ${err}`;
  }
});