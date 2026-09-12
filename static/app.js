const EMOTION_COLORS = {
  sadness: "#4A6FA5",
  joy: "#E1A83E",
  love: "#C64B6B",
  anger: "#C0433A",
  fear: "#6E5A8E",
  surprise: "#2E9C93",
};

const btn = document.getElementById("analyzeBtn");
const input = document.getElementById("textInput");
const emptyState = document.getElementById("emptyState");
const resultBox = document.getElementById("result");
const emotionText = document.getElementById("emotionText");
const confidenceText = document.getElementById("confidenceText");
const scoreBars = document.getElementById("scoreBars");
const errorText = document.getElementById("errorText");

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function setAccent(color) {
  document.documentElement.style.setProperty("--accent", color);
  document.documentElement.style.setProperty("--accent-soft", hexToRgba(color, 0.16));
}

async function analyze() {
  const text = input.value.trim();
  errorText.classList.add("hidden");

  if (!text) {
    errorText.textContent = "Type something first — there's nothing to read yet.";
    errorText.classList.remove("hidden");
    return;
  }

  btn.disabled = true;
  btn.textContent = "Analyzing…";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Something went wrong.");
    }

    const color = EMOTION_COLORS[data.emotion] || "#8b8578";
    setAccent(color);

    emotionText.textContent = data.emotion;
    confidenceText.textContent = `${Math.round(data.confidence * 100)}% sure`;

    scoreBars.innerHTML = "";
    const sorted = Object.entries(data.all_scores).sort((a, b) => b[1] - a[1]);
    for (const [label, score] of sorted) {
      const pct = (score * 100).toFixed(0);
      const barColor = EMOTION_COLORS[label] || "#8b8578";
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <div class="bar-label">${label}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%; background:${barColor}"></div></div>
        <div class="bar-pct">${pct}%</div>
      `;
      scoreBars.appendChild(row);
    }

    emptyState.classList.add("hidden");
    resultBox.classList.remove("hidden");
  } catch (err) {
    errorText.textContent = err.message;
    errorText.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze";
  }
}

btn.addEventListener("click", analyze);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    analyze();
  }
});
