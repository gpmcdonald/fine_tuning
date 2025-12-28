const healthDot = document.getElementById("healthDot");
const healthText = document.getElementById("healthText");
const logBox = document.getElementById("log");
const img = document.getElementById("resultImage");

const promptBox = document.getElementById("prompt");
const generateBtn = document.getElementById("generateBtn");
const clearBtn = document.getElementById("clearBtn");

/* ---------- helpers ---------- */

function log(msg) {
  logBox.textContent += msg + "\n";
  logBox.scrollTop = logBox.scrollHeight;
}

/* ---------- health check ---------- */

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const json = await res.json();

    healthDot.className = "dot good";
    healthText.textContent = json.status ?? "ok";
  } catch {
    healthDot.className = "dot bad";
    healthText.textContent = "offline";
  }
}

/* ---------- image generation ---------- */

generateBtn.onclick = async () => {
  const prompt = promptBox.value.trim();
  if (!prompt) return;

  generateBtn.disabled = true;
  log("Generating image…");

  try {
    const res = await fetch(
      `/image?prompt=${encodeURIComponent(prompt)}`,
      { method: "POST" }
    );

    if (!res.ok) {
      const text = await res.text();
      throw new Error(text);
    }

    const json = await res.json();

    if (!json || !json.image_path) {
      throw new Error("API did not return image_path");
    }

    /*
      Convert:
        C:\Users\...\outputs\images\file.png
      Into:
        /outputs/images/file.png
    */
    const url =
      "/outputs" +
      json.image_path
        .replace(/\\/g, "/")
        .split("outputs")[1];

    img.src = url;
    img.hidden = false;

    log("Image ready ✔");
  } catch (e) {
    log("ERROR: " + (e?.message ?? e));
  }

  generateBtn.disabled = false;
};

/* ---------- clear ---------- */

clearBtn.onclick = () => {
  promptBox.value = "";
  logBox.textContent = "";
  img.hidden = true;
};

/* ---------- init ---------- */

checkHealth();