// Trie mirroring the Python implementation from assignment1.ipynb.
class Trie {
  constructor() {
    this.children = new Map();
    this.order = null;
  }

  insert(word, idx) {
    let node = this;
    for (const c of word) {
      if (!node.children.has(c)) node.children.set(c, new Trie());
      node = node.children.get(c);
    }
    node.order = idx;
  }

  autocomplete(prefix, limit = Infinity) {
    let node = this;
    for (const c of prefix) {
      if (!node.children.has(c)) return [];
      node = node.children.get(c);
    }
    const results = [];
    const dfs = (n, path) => {
      if (n.order !== null) results.push([prefix + path, n.order]);
      for (const [c, child] of n.children) dfs(child, path + c);
    };
    dfs(node, "");
    results.sort((a, b) => a[1] - b[1]);
    return results.slice(0, limit);
  }
}

// ---------- Boot ----------

const pad = document.getElementById("pad");
const mirror = document.getElementById("mirror");
const suggestionsEl = document.getElementById("suggestions");
const statusEl = document.getElementById("status");

const trie = new Trie();
let vocabSize = 0;
let suggestions = [];
let activeIdx = 0;
let currentPrefix = "";
let currentWordStart = 0;

async function loadVocab() {
  const resp = await fetch("vocab.txt");
  if (!resp.ok) throw new Error(`Failed to load vocab.txt: ${resp.status}`);
  const text = await resp.text();
  const words = text.split("\n").map((w) => w.trim()).filter(Boolean);
  words.forEach((w, i) => trie.insert(w, i));
  vocabSize = words.length;
  statusEl.textContent = `Loaded ${vocabSize.toLocaleString()} unique words.`;
}

// ---------- Caret / current-word tracking ----------

function findWordAtCaret() {
  const value = pad.value;
  const caret = pad.selectionStart;
  if (caret !== pad.selectionEnd) return null; // ignore selections
  let start = caret;
  while (start > 0 && /[\w'-]/.test(value[start - 1])) start--;
  const prefix = value.slice(start, caret).toLowerCase();
  return { start, prefix };
}

// Mirror-based caret position measurement.
function getCaretCoords() {
  const style = getComputedStyle(pad);
  // Copy every computed property so shorthand pieces (padding-top, border-left-width, etc.) land correctly.
  for (let i = 0; i < style.length; i++) {
    const prop = style[i];
    mirror.style.setProperty(prop, style.getPropertyValue(prop));
  }
  mirror.style.position = "absolute";
  mirror.style.visibility = "hidden";
  mirror.style.overflow = "auto";
  mirror.style.top = pad.offsetTop + "px";
  mirror.style.left = pad.offsetLeft + "px";
  mirror.style.width = pad.offsetWidth + "px";
  mirror.style.height = pad.offsetHeight + "px";

  const value = pad.value.substring(0, pad.selectionStart);
  mirror.textContent = value;
  const marker = document.createElement("span");
  marker.textContent = ".";
  marker.style.visibility = "hidden";
  mirror.appendChild(marker);

  const top =
    marker.offsetTop - pad.scrollTop + parseFloat(style.lineHeight);
  const left = marker.offsetLeft - pad.scrollLeft;
  return {
    top: pad.offsetTop + top,
    left: pad.offsetLeft + left,
  };
}

// ---------- Suggestion box ----------

function renderSuggestions() {
  suggestionsEl.innerHTML = "";
  if (suggestions.length === 0) {
    suggestionsEl.hidden = true;
    return;
  }
  suggestions.forEach(([word, rank], i) => {
    const li = document.createElement("li");
    if (i === activeIdx) li.classList.add("active");
    const matchLen = currentPrefix.length;
    li.innerHTML =
      `<span><strong>${word.slice(0, matchLen)}</strong>${word.slice(matchLen)}</span>` +
      `<span class="rank">#${rank + 1}</span>`;
    li.addEventListener("mousedown", (e) => {
      e.preventDefault();
      acceptSuggestion(i);
    });
    suggestionsEl.appendChild(li);
  });

  const { top, left } = getCaretCoords();
  suggestionsEl.style.top = top + 4 + "px";
  suggestionsEl.style.left = left + "px";
  suggestionsEl.hidden = false;

  const activeLi = suggestionsEl.children[activeIdx];
  if (activeLi) activeLi.scrollIntoView({ block: "nearest" });
}

function updateSuggestions() {
  const info = findWordAtCaret();
  if (!info || info.prefix.length === 0) {
    suggestions = [];
    suggestionsEl.hidden = true;
    return;
  }
  currentPrefix = info.prefix;
  currentWordStart = info.start;
  suggestions = trie.autocomplete(info.prefix);
  activeIdx = 0;
  renderSuggestions();
}

function acceptSuggestion(idx) {
  if (!suggestions[idx]) return;
  const [word] = suggestions[idx];
  const caret = pad.selectionStart;
  const before = pad.value.slice(0, currentWordStart);
  const after = pad.value.slice(caret);
  pad.value = before + word + after;
  const newCaret = before.length + word.length;
  pad.setSelectionRange(newCaret, newCaret);
  suggestions = [];
  suggestionsEl.hidden = true;
  pad.focus();
}

// ---------- Events ----------

pad.addEventListener("input", updateSuggestions);
pad.addEventListener("click", updateSuggestions);
pad.addEventListener("keyup", (e) => {
  if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) {
    updateSuggestions();
  }
});

pad.addEventListener("keydown", (e) => {
  if (suggestionsEl.hidden || suggestions.length === 0) return;
  switch (e.key) {
    case "ArrowDown":
      e.preventDefault();
      activeIdx = (activeIdx + 1) % suggestions.length;
      renderSuggestions();
      break;
    case "ArrowUp":
      e.preventDefault();
      activeIdx = (activeIdx - 1 + suggestions.length) % suggestions.length;
      renderSuggestions();
      break;
    case "Tab":
      e.preventDefault();
      if (e.shiftKey) {
        activeIdx = (activeIdx - 1 + suggestions.length) % suggestions.length;
      } else {
        activeIdx = (activeIdx + 1) % suggestions.length;
      }
      renderSuggestions();
      break;
    case "Enter":
      e.preventDefault();
      acceptSuggestion(activeIdx);
      break;
    case "Escape":
      e.preventDefault();
      suggestions = [];
      suggestionsEl.hidden = true;
      break;
  }
});

pad.addEventListener("blur", () => {
  setTimeout(() => (suggestionsEl.hidden = true), 100);
});

// ---------- Go ----------

loadVocab().catch((err) => {
  statusEl.textContent = "Error loading vocabulary. Run the notebook cell to generate vocab.txt.";
  console.error(err);
});
