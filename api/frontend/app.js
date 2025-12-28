const toggleChatBtn = document.getElementById("toggleChatBtn");
const closeChatBtn = document.getElementById("closeChatBtn");
const chatPanel = document.getElementById("chatPanel");

const promptBox = document.getElementById("prompt");
const generateBtn = document.getElementById("generateBtn");
const clearBtn = document.getElementById("clearBtn");
const img = document.getElementById("resultImage");
const logBox = document.getElementById("log");

const chatLog = document.getElementById("chatLog");
const chatBox = document.getElementById("chatBox");
const sendChatBtn = document.getElementById("sendChatBtn");

let history = [];

/* ─── Sidekick ─────────────────────────── */
toggleChatBtn.onclick = () => chatPanel.classList.add("open");
closeChatBtn.onclick = () => chatPanel.classList.remove("open");

function addChat(role, text){
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

/* ─── Image Generation ─────────────────── */
generateBtn.onclick = async () => {
  const prompt = promptBox.value.trim();
  if(!prompt) return;

  generateBtn.disabled = true;
  logBox.textContent = "Generating image...\n";

  try{
    const res = await fetch(`/image?prompt=${encodeURIComponent(prompt)}`, {method:"POST"});
    const json = await res.json();

    if(!json.image_path) throw new Error("No image returned");

    const url = "/outputs" + json.image_path.replace(/\\/g,"/").split("outputs")[1];
    img.src = url;
    img.hidden = false;

    logBox.textContent += "✔ Image ready\n";
  }catch(e){
    logBox.textContent += "ERROR: " + e.message + "\n";
  }

  generateBtn.disabled = false;
};

clearBtn.onclick = () => {
  promptBox.value = "";
  img.hidden = true;
  logBox.textContent = "";
};

/* ─── Chat (Phase A stub) ───────────────── */
sendChatBtn.onclick = () => {
  const msg = chatBox.value.trim();
  if(!msg) return;

  addChat("user", msg);
  chatBox.value = "";

  // Placeholder intelligent response
  addChat(
    "assistant",
    "Try adding style, lighting, or camera details. For example: cinematic lighting, shallow depth of field."
  );
};