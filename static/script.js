// script.js
const form = document.getElementById("chatForm");
const input = document.getElementById("queryInput");
const chatWindow = document.getElementById("chatWindow");
const status = document.getElementById("status");
const sendBtn = document.getElementById("sendBtn");

function appendMessage(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);

  // auto scroll to bottom
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendQuestion() {
  const q = input.value.trim();
  if (!q) return;

  // Show user message immediately
  appendMessage("user", q);

  input.value = "";
  input.disabled = true;
  sendBtn.disabled = true;
  status.textContent = "Thinking...";

  try {
    const resp = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    });

    const data = await resp.json();

    if (!resp.ok) {
      appendMessage("assistant", "Error: " + (data.error || resp.statusText));
    } else {
      appendMessage("assistant", data.answer || "No answer");
    }

  } catch (err) {
    appendMessage("assistant", "Network or server error.");
    console.error(err);
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
    status.textContent = "";
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  sendQuestion();
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});
