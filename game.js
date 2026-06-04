/**
 * Chronicles of the Abyss — game.js
 * AI D&D Game with multi-LLM backend adapter
 * Neofilisoft / Studio Balmung
 *
 * Supported providers:
 *   groq      — Groq LPU Inference (ultra-fast, OpenAI-compatible)  ← DEFAULT for demo
 *   claude    — Anthropic Messages API
 *   openai    — OpenAI Chat Completions API
 *   gemini    — Google Gemini API (generateContent)
 *   deepseek  — DeepSeek Chat Completions (OpenAI-compatible)
 *   ollama    — Local Ollama REST API (OpenAI-compatible)
 *   custom    — Any OpenAI-compatible endpoint
 *
 * What is an AI Engine?
 *   An AI Engine = the LLM (Large Language Model) that acts as the Dungeon Master.
 *   It consists of three layers:
 *     1. Model     — the neural network (e.g. llama-3.3-70b, gpt-4o, gemini-2.0-flash)
 *     2. Inference — the server that runs the model and returns text (Groq LPU, GPU cluster, etc.)
 *     3. Provider  — the platform/API you call (Groq, Anthropic, OpenAI, Google, local Ollama)
 *   Groq specifically is NOT a model creator — it's a blazing-fast inference engine using
 *   custom LPU chips, running open-source models (LLaMA, Mixtral, Gemma) up to 100× faster
 *   than standard GPUs.
 */

'use strict';

/* ═══════════════════════════════════════════════════════
   LLM PROVIDER REGISTRY
   Each entry defines: models[], defaultModel, endpoint(),
   headers(), buildBody(), parseResponse()
═══════════════════════════════════════════════════════ */
const LLM_PROVIDERS = {

  /* ── Groq — ultra-fast LPU inference, OpenAI-compatible ── */
  groq: {
    label: 'Groq (Fast Inference)',
    needsKey: true,
    badge: '⚡',
    models: [
      'llama-3.3-70b-versatile',
      'llama-3.1-8b-instant',
      'llama3-70b-8192',
      'llama3-8b-8192',
      'mixtral-8x7b-32768',
      'gemma2-9b-it',
      'gemma-7b-it',
    ],
    endpoint: (cfg) => 'https://api.groq.com/openai/v1/chat/completions',
    headers: (cfg) => ({
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${cfg.apiKey}`,
    }),
    buildBody: (cfg, systemPrompt, history, maxTokens) => JSON.stringify({
      model: cfg.model,
      max_tokens: maxTokens,
      temperature: 0.85,
      messages: [
        { role: 'system', content: systemPrompt },
        ...history,
      ],
    }),
    parseResponse: (data) => {
      if (data.choices && data.choices[0]) {
        return data.choices[0].message.content;
      }
      throw new Error(data.error?.message || 'No choices in Groq response');
    },
  },

  claude: {
    label: 'Anthropic Claude',
    needsKey: true,
    models: [
      'claude-sonnet-4-20250514',
      'claude-opus-4-20250514',
      'claude-haiku-4-5-20251001',
    ],
    endpoint: (cfg) => 'https://api.anthropic.com/v1/messages',
    headers: (cfg) => ({
      'Content-Type': 'application/json',
      'x-api-key': cfg.apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    }),
    buildBody: (cfg, systemPrompt, history, maxTokens) => JSON.stringify({
      model: cfg.model,
      max_tokens: maxTokens,
      system: systemPrompt,
      messages: history,
    }),
    parseResponse: (data) => {
      if (data.content && data.content[0] && data.content[0].text) {
        return data.content[0].text;
      }
      throw new Error(data.error?.message || 'No content in Claude response');
    },
  },

  openai: {
    label: 'OpenAI GPT',
    needsKey: true,
    models: [
      'gpt-4o',
      'gpt-4o-mini',
      'gpt-4-turbo',
      'gpt-3.5-turbo',
    ],
    endpoint: (cfg) => 'https://api.openai.com/v1/chat/completions',
    headers: (cfg) => ({
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${cfg.apiKey}`,
    }),
    buildBody: (cfg, systemPrompt, history, maxTokens) => JSON.stringify({
      model: cfg.model,
      max_tokens: maxTokens,
      messages: [
        { role: 'system', content: systemPrompt },
        ...history,
      ],
    }),
    parseResponse: (data) => {
      if (data.choices && data.choices[0]) {
        return data.choices[0].message.content;
      }
      throw new Error(data.error?.message || 'No choices in OpenAI response');
    },
  },

  gemini: {
    label: 'Google Gemini',
    needsKey: true,
    models: [
      'gemini-2.0-flash',
      'gemini-2.0-flash-lite',
      'gemini-1.5-pro',
      'gemini-1.5-flash',
    ],
    endpoint: (cfg) =>
      `https://generativelanguage.googleapis.com/v1beta/models/${cfg.model}:generateContent?key=${cfg.apiKey}`,
    headers: (cfg) => ({ 'Content-Type': 'application/json' }),
    buildBody: (cfg, systemPrompt, history, maxTokens) => {
      // Gemini uses a different conversation format
      const contents = history.map((m) => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      }));
      return JSON.stringify({
        systemInstruction: { parts: [{ text: systemPrompt }] },
        contents,
        generationConfig: { maxOutputTokens: maxTokens },
      });
    },
    parseResponse: (data) => {
      const candidate = data.candidates?.[0];
      if (candidate?.content?.parts?.[0]?.text) {
        return candidate.content.parts[0].text;
      }
      throw new Error(data.error?.message || 'No text in Gemini response');
    },
  },

  deepseek: {
    label: 'DeepSeek',
    needsKey: true,
    models: [
      'deepseek-chat',
      'deepseek-reasoner',
    ],
    endpoint: (cfg) => 'https://api.deepseek.com/chat/completions',
    headers: (cfg) => ({
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${cfg.apiKey}`,
    }),
    buildBody: (cfg, systemPrompt, history, maxTokens) => JSON.stringify({
      model: cfg.model,
      max_tokens: maxTokens,
      messages: [
        { role: 'system', content: systemPrompt },
        ...history,
      ],
    }),
    parseResponse: (data) => {
      if (data.choices && data.choices[0]) {
        return data.choices[0].message.content;
      }
      throw new Error(data.error?.message || 'No choices in DeepSeek response');
    },
  },

  ollama: {
    label: 'Local Ollama',
    needsKey: false,
    models: [
      'llama3',
      'llama3.2',
      'mistral',
      'gemma3',
      'phi4',
      'qwen2.5',
    ],
    endpoint: (cfg) => `${cfg.url || 'http://localhost:11434'}/api/chat`,
    headers: (cfg) => ({ 'Content-Type': 'application/json' }),
    buildBody: (cfg, systemPrompt, history, maxTokens) => JSON.stringify({
      model: cfg.model,
      stream: false,
      messages: [
        { role: 'system', content: systemPrompt },
        ...history,
      ],
    }),
    parseResponse: (data) => {
      if (data.message?.content) return data.message.content;
      throw new Error('No message content in Ollama response');
    },
  },

  custom: {
    label: 'Custom Endpoint',
    needsKey: false,
    models: ['gpt-3.5-turbo', 'mistral', 'llama3', 'custom-model'],
    endpoint: (cfg) => `${cfg.url || 'http://localhost:8080'}/v1/chat/completions`,
    headers: (cfg) => ({
      'Content-Type': 'application/json',
      ...(cfg.apiKey ? { 'Authorization': `Bearer ${cfg.apiKey}` } : {}),
    }),
    buildBody: (cfg, systemPrompt, history, maxTokens) => JSON.stringify({
      model: cfg.model,
      max_tokens: maxTokens,
      messages: [
        { role: 'system', content: systemPrompt },
        ...history,
      ],
    }),
    parseResponse: (data) => {
      if (data.choices?.[0]) return data.choices[0].message.content;
      throw new Error(data.error?.message || 'No choices in response');
    },
  },
};

/* ═══════════════════════════════════════════════════════
   GAME STATE
═══════════════════════════════════════════════════════ */
const G = {
  // Character
  name: '', heroClass: '', bg: '', campaign: '',
  hp: 60, maxHp: 60, level: 1, xp: 0, gold: 10,
  inventory: ['Adventurer\'s Pack', 'Torch × 3', 'Rations × 5', '50 ft. Rope'],
  quests: [{ text: 'Survive the first encounter', active: true, done: false }],

  // Conversation
  history: [],
  isLoading: false,
  turnCount: 0,

  // LLM config (active) — default: Groq (free tier available at console.groq.com)
  llm: {
    provider: 'groq',
    model: 'llama-3.3-70b-versatile',
    apiKey: '',
    url: 'http://localhost:11434',
    connected: false,
  },
};

/* ═══════════════════════════════════════════════════════
   PROVIDER / MODEL UI HELPERS
═══════════════════════════════════════════════════════ */
function populateProviderSelect(selectEl) {
  selectEl.innerHTML = Object.entries(LLM_PROVIDERS)
    .map(([k, v]) => `<option value="${k}">${v.label}</option>`)
    .join('');
  selectEl.value = G.llm.provider;
}

function populateModelSelect(selectEl, providerKey) {
  const prov = LLM_PROVIDERS[providerKey];
  selectEl.innerHTML = prov.models
    .map((m) => `<option value="${m}">${m}</option>`)
    .join('');
  selectEl.value = G.llm.model || prov.models[0];
}

function onProviderChange(providerKey, modelSelectEl, apikeyFieldEl, urlFieldEl) {
  const prov = LLM_PROVIDERS[providerKey];
  populateModelSelect(modelSelectEl, providerKey);
  if (apikeyFieldEl) apikeyFieldEl.style.display = prov.needsKey ? '' : 'none';
  if (urlFieldEl) {
    urlFieldEl.style.display =
      (providerKey === 'ollama' || providerKey === 'custom') ? '' : 'none';
  }
  // Show Groq tip only on setup screen when Groq is selected
  const tip = document.getElementById('groq-tip');
  if (tip) tip.style.display = providerKey === 'groq' ? '' : 'none';
}

/* ═══════════════════════════════════════════════════════
   SETUP SCREEN INIT
═══════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  // Populate setup-screen dropdowns
  const provSel   = document.getElementById('llm-provider');
  const modelSel  = document.getElementById('llm-model');
  const apiField  = document.getElementById('api-key-field');
  const urlField  = document.getElementById('custom-url-field');

  populateProviderSelect(provSel);
  populateModelSelect(modelSel, G.llm.provider);

  provSel.addEventListener('change', () => {
    onProviderChange(provSel.value, modelSel, apiField, urlField);
  });

  // Class selection
  document.querySelectorAll('.class-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.class-btn').forEach((b) => b.classList.remove('selected'));
      btn.classList.add('selected');
    });
  });

  // Restore saved key if any
  const saved = sessionStorage.getItem('dnd_llm_config');
  if (saved) {
    try {
      const c = JSON.parse(saved);
      G.llm = { ...G.llm, ...c };
      provSel.value = G.llm.provider;
      onProviderChange(G.llm.provider, modelSel, apiField, urlField);
      modelSel.value = G.llm.model;
      document.getElementById('llm-apikey').value = G.llm.apiKey || '';
      document.getElementById('llm-url').value = G.llm.url || '';
    } catch (_) {}
  }

  // Modal provider dropdown
  const mProv = document.getElementById('m-provider');
  populateProviderSelect(mProv);
  mProv.addEventListener('change', () => {
    onProviderChange(
      mProv.value,
      document.getElementById('m-model'),
      document.getElementById('m-apikey-field'),
      document.getElementById('m-url-field'),
    );
  });
  populateModelSelect(document.getElementById('m-model'), G.llm.provider);

  // Trigger initial tip visibility
  const tip = document.getElementById('groq-tip');
  if (tip) tip.style.display = G.llm.provider === 'groq' ? '' : 'none';
});

/* ═══════════════════════════════════════════════════════
   CONNECTION TESTING
═══════════════════════════════════════════════════════ */
async function testConnection() {
  await _testConnectionWith(
    document.getElementById('llm-provider').value,
    document.getElementById('llm-model').value,
    document.getElementById('llm-apikey').value.trim(),
    document.getElementById('llm-url').value.trim(),
    document.getElementById('llm-status'),
    document.getElementById('begin-btn'),
    true,
  );
}

async function testConnectionModal() {
  await _testConnectionWith(
    document.getElementById('m-provider').value,
    document.getElementById('m-model').value,
    document.getElementById('m-apikey').value.trim(),
    document.getElementById('m-url').value.trim(),
    document.getElementById('m-status'),
    null,
    false,
  );
}

async function _testConnectionWith(provKey, model, apiKey, url, statusEl, beginBtn, updateGlobal) {
  const prov = LLM_PROVIDERS[provKey];
  if (!prov) return;

  const cfg = { provider: provKey, model, apiKey, url };

  statusEl.className = 'llm-status testing';
  statusEl.textContent = '◌ Connecting…';

  try {
    const endpoint = prov.endpoint(cfg);
    const headers  = prov.headers(cfg);
    const body     = prov.buildBody(
      cfg,
      'You are a test assistant. Reply with exactly: "Connection successful."',
      [{ role: 'user', content: 'Test' }],
      20,
    );

    const res  = await fetch(endpoint, { method: 'POST', headers, body });
    const data = await res.json();
    const text = prov.parseResponse(data);

    statusEl.className = 'llm-status ok';
    statusEl.textContent = `✓ Connected — ${prov.label} · ${model}`;

    if (updateGlobal) {
      G.llm = { provider: provKey, model, apiKey, url, connected: true };
      sessionStorage.setItem('dnd_llm_config', JSON.stringify(G.llm));
    }

    if (beginBtn) {
      beginBtn.disabled = false;
      beginBtn.classList.add('ready');
      beginBtn.textContent = 'Begin Your Legend';
    }

  } catch (err) {
    statusEl.className = 'llm-status error';
    statusEl.textContent = `✗ Failed — ${err.message.slice(0, 80)}`;
    console.error('[LLM Test]', err);
  }
}

/* ═══════════════════════════════════════════════════════
   SETTINGS MODAL (IN-GAME)
═══════════════════════════════════════════════════════ */
function openSettings() {
  const modal = document.getElementById('settings-modal');
  modal.style.display = 'flex';

  const mProv  = document.getElementById('m-provider');
  const mModel = document.getElementById('m-model');
  mProv.value = G.llm.provider;
  populateModelSelect(mModel, G.llm.provider);
  mModel.value = G.llm.model;
  document.getElementById('m-apikey').value = G.llm.apiKey || '';
  document.getElementById('m-url').value = G.llm.url || '';

  onProviderChange(
    G.llm.provider,
    mModel,
    document.getElementById('m-apikey-field'),
    document.getElementById('m-url-field'),
  );
}

function closeSettings() {
  document.getElementById('settings-modal').style.display = 'none';
}

function saveSettings() {
  const provKey = document.getElementById('m-provider').value;
  G.llm = {
    provider: provKey,
    model: document.getElementById('m-model').value,
    apiKey: document.getElementById('m-apikey').value.trim(),
    url: document.getElementById('m-url').value.trim(),
    connected: true,
  };
  sessionStorage.setItem('dnd_llm_config', JSON.stringify(G.llm));

  document.getElementById('ui-provider').textContent =
    LLM_PROVIDERS[provKey]?.label.replace('Anthropic ', '') || provKey;

  document.getElementById('m-status').className = 'llm-status ok';
  document.getElementById('m-status').textContent = `✓ Saved — ${LLM_PROVIDERS[provKey]?.label} · ${G.llm.model}`;

  appendMsg('system', `⚙ LLM switched to ${LLM_PROVIDERS[provKey]?.label} · ${G.llm.model}`);
  setTimeout(closeSettings, 800);
}

/* ═══════════════════════════════════════════════════════
   KEY VISIBILITY TOGGLES
═══════════════════════════════════════════════════════ */
function toggleKeyVisibility() {
  const inp = document.getElementById('llm-apikey');
  inp.type = inp.type === 'password' ? 'text' : 'password';
}
function toggleModalKeyVisibility() {
  const inp = document.getElementById('m-apikey');
  inp.type = inp.type === 'password' ? 'text' : 'password';
}

/* ═══════════════════════════════════════════════════════
   BEGIN GAME
═══════════════════════════════════════════════════════ */
function beginGame() {
  const selectedClassBtn = document.querySelector('.class-btn.selected');
  const campaignSel      = document.getElementById('campaign-type');

  G.name      = document.getElementById('hero-name').value.trim() || 'Traveller';
  G.bg        = document.getElementById('hero-bg').value;
  G.campaign  = campaignSel.value;
  G.heroClass = selectedClassBtn?.dataset.class || 'Fighter';
  G.hp        = parseInt(selectedClassBtn?.dataset.hp || 60);
  G.maxHp     = G.hp;

  // Read final LLM settings from setup form
  const provKey = document.getElementById('llm-provider').value;
  G.llm = {
    provider: provKey,
    model:    document.getElementById('llm-model').value,
    apiKey:   document.getElementById('llm-apikey').value.trim(),
    url:      document.getElementById('llm-url').value.trim() || 'http://localhost:11434',
    connected: true,
  };
  sessionStorage.setItem('dnd_llm_config', JSON.stringify(G.llm));

  document.getElementById('setup-screen').style.display = 'none';
  document.getElementById('app').style.display = 'flex';

  document.getElementById('ui-provider').textContent =
    LLM_PROVIDERS[provKey]?.label.replace('Anthropic ', '') || provKey;

  updateCharBar();
  updateInventoryUI();
  updateQuestsUI();
  startAdventure(campaignSel.options[campaignSel.selectedIndex].text.split('—')[0].trim());
}

/* ═══════════════════════════════════════════════════════
   UI HELPERS
═══════════════════════════════════════════════════════ */
function updateCharBar() {
  document.getElementById('ui-name').textContent  = G.name;
  document.getElementById('ui-class').textContent = G.heroClass;
  document.getElementById('ui-hp').textContent    = `${G.hp}/${G.maxHp}`;
  document.getElementById('ui-level').textContent = G.level;
  document.getElementById('ui-xp').textContent    = G.xp;
  document.getElementById('ui-gold').textContent  = G.gold;
}

function updateInventoryUI() {
  document.getElementById('inventory-list').innerHTML =
    G.inventory.map((i) => `<li>${escapeHtml(i)}</li>`).join('');
}

function updateQuestsUI() {
  document.getElementById('quests-list').innerHTML =
    G.quests.map((q) =>
      `<li class="quest ${q.active ? 'active' : ''} ${q.done ? 'done' : ''}">${escapeHtml(q.text)}</li>`
    ).join('');
}

const scrollArea = document.getElementById('scroll-area');

function appendMsg(type, text, speaker) {
  const div = document.createElement('div');
  div.className = `msg ${type}`;
  if (type === 'dm') {
    div.innerHTML =
      `<div class="speaker">${escapeHtml(speaker || 'Dungeon Master')}</div>` +
      `<div class="body">${formatDMText(text)}</div>`;
  } else {
    div.innerHTML = `<div class="body">${escapeHtml(text)}</div>`;
  }
  scrollArea.appendChild(div);
  scrollArea.scrollTop = scrollArea.scrollHeight;
  return div;
}

function formatDMText(text) {
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n\n/g, '</p><p style="margin-top:0.7rem;">')
    .replace(/\n/g, '<br>');
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function showTyping() {
  const el = document.createElement('div');
  el.id = 'typing-el';
  el.className = 'typing-indicator';
  el.innerHTML = `<div class="dots"><i></i><i></i><i></i></div><span>DM is weaving fate…</span>`;
  scrollArea.appendChild(el);
  scrollArea.scrollTop = scrollArea.scrollHeight;
}

function removeTyping() {
  document.getElementById('typing-el')?.remove();
}

/* ═══════════════════════════════════════════════════════
   DICE ROLLER
═══════════════════════════════════════════════════════ */
function rollDice(sides) {
  const result  = Math.floor(Math.random() * sides) + 1;
  const quality =
    result === sides         ? '💥 Critical!' :
    result === 1             ? '💀 Fumble!'   :
    result >= Math.ceil(sides * 0.75) ? '✨ Great'  : '';
  appendMsg('roll', `d${sides} → ${result}  ${quality}`);
  scrollArea.scrollTop = scrollArea.scrollHeight;
  return result;
}

/* ═══════════════════════════════════════════════════════
   SYSTEM PROMPT
═══════════════════════════════════════════════════════ */
function buildSystemPrompt() {
  const settings = {
    dungeon:  'ancient cursed ruins beneath a dead city, filled with undead, traps, dark magic, and forgotten secrets',
    forest:   'a cursed ancient elven woodland where time flows strangely, spirits haunt twisted trees, and dark fey lurk',
    sea:      'the ruins of a sunken empire — half-flooded halls, drowned spirits, and deep-sea horrors',
    mountain: "a dragon's fortress carved into a frozen peak, full of draconic cultists and elemental dangers",
    city:     'a corrupt shadow city where guilds scheme, assassins lurk, and ancient conspiracies hide in plain sight',
  };

  return `You are the Dungeon Master for a dark, atmospheric D&D 5e adventure called "Chronicles of the Abyss."

PLAYER CHARACTER:
- Name: ${G.name}
- Class: ${G.heroClass}
- Background: ${G.bg}
- HP: ${G.hp}/${G.maxHp}
- Level: ${G.level}
- XP: ${G.xp}
- Gold: ${G.gold} gp
- Inventory: ${G.inventory.join(', ')}

SETTING: ${settings[G.campaign] || settings.dungeon}

YOUR ROLE AS DM:
- Write vivid, immersive narrative prose in 2-4 paragraphs. Use sensory detail: sounds, smells, cold air, flickering shadows.
- Present meaningful choices and consequences. The world reacts to the player.
- Honor the character's class abilities — a Wizard casts spells, a Rogue can sneak, etc.
- Include NPC dialogue in quotes. Make enemies feel menacing, allies feel mortal.
- Call for skill checks when appropriate, e.g. "(Roll Perception)" or "(Roll Strength — DC 14)".
- Use **bold** for important names/places, *italics* for sensory emphasis.
- Keep tone dark, tense, cinematic — like a skilled author.
- End each message with a clear choice or question for the player.
- NEVER break character or mention being an AI.
- Grant XP with [+X XP] in brackets. Deal damage with [HP -X]. Heal with [HP +X]. Give gold with [+X GP]. Give items with [+Item: item name].
- Keep responses under 350 words unless the drama demands more.`;
}

/* ═══════════════════════════════════════════════════════
   PARSE DM RESPONSE → GAME EVENTS
═══════════════════════════════════════════════════════ */
function parseDMResponse(text) {
  const xpMatch = text.match(/\[\+?\s*(\d+)\s*XP\]/i);
  if (xpMatch) {
    G.xp += parseInt(xpMatch[1]);
    if (G.xp >= G.level * 100) {
      G.level++;
      G.maxHp += 8;
      G.hp = Math.min(G.hp + 8, G.maxHp);
      setTimeout(() => appendMsg('system', `⬆ Level Up! You are now Level ${G.level}. Max HP increased.`), 300);
    }
  }

  const dmgMatch = text.match(/\[HP\s*[-–]\s*(\d+)\]/i);
  if (dmgMatch) {
    G.hp = Math.max(0, G.hp - parseInt(dmgMatch[1]));
    if (G.hp === 0) {
      G.hp = 1;
      setTimeout(() => appendMsg('system', '💀 You have fallen. Type "continue" to be revived by fate.'), 300);
    }
  }

  const healMatch = text.match(/\[HP\s*\+\s*(\d+)\]/i);
  if (healMatch) G.hp = Math.min(G.maxHp, G.hp + parseInt(healMatch[1]));

  const goldMatch = text.match(/\[\+?\s*(\d+)\s*(?:GP|gold)\]/i);
  if (goldMatch) G.gold += parseInt(goldMatch[1]);

  const itemMatch = text.match(/\[\+Item:\s*([^\]]+)\]/i);
  if (itemMatch) {
    const item = itemMatch[1].trim();
    if (!G.inventory.includes(item)) G.inventory.push(item);
    updateInventoryUI();
  }

  updateCharBar();
}

/* ═══════════════════════════════════════════════════════
   CORE LLM CALL  (provider-agnostic)
═══════════════════════════════════════════════════════ */
async function callDM(userMsg) {
  if (G.isLoading) return;
  G.isLoading = true;
  document.getElementById('send-btn').disabled = true;

  if (userMsg) G.history.push({ role: 'user', content: userMsg });

  showTyping();

  try {
    const prov     = LLM_PROVIDERS[G.llm.provider];
    const endpoint = prov.endpoint(G.llm);
    const headers  = prov.headers(G.llm);
    const body     = prov.buildBody(G.llm, buildSystemPrompt(), G.history, 1000);

    const res  = await fetch(endpoint, { method: 'POST', headers, body });
    const data = await res.json();

    removeTyping();

    const dmText = prov.parseResponse(data);

    G.history.push({ role: 'assistant', content: dmText });
    if (G.history.length > 32) G.history = G.history.slice(-28); // trim context

    parseDMResponse(dmText);
    appendMsg('dm', dmText);

    G.turnCount++;
    if (G.turnCount % 5 === 0) advanceQuestHint();

  } catch (err) {
    removeTyping();
    appendMsg('system', `The magic fades… ${err.message}`);
    console.error('[callDM]', err);
  } finally {
    G.isLoading = false;
    document.getElementById('send-btn').disabled = false;
  }
}

/* ═══════════════════════════════════════════════════════
   QUEST PROGRESSION
═══════════════════════════════════════════════════════ */
function advanceQuestHint() {
  const steps = [
    'Explore the entrance and find a way deeper',
    'Uncover the dark secret at the heart of the lair',
    'Confront the final guardian and claim your destiny',
  ];
  const idx = Math.min(Math.floor(G.turnCount / 5), steps.length - 1);
  G.quests[0].text = steps[idx];
  updateQuestsUI();
}

/* ═══════════════════════════════════════════════════════
   START ADVENTURE
═══════════════════════════════════════════════════════ */
function startAdventure(settingLabel) {
  const openingPrompt =
    `Begin the adventure. Set the opening scene — ${G.name} ` +
    `(${G.heroClass}, ${G.bg} background) stands at the entrance to ` +
    `${settingLabel}. Establish the atmosphere vividly, hint at the dangers ` +
    `ahead, and present the first decision for the player.`;
  callDM(openingPrompt);
}

/* ═══════════════════════════════════════════════════════
   PLAYER ACTIONS
═══════════════════════════════════════════════════════ */
function sendAction() {
  const input = document.getElementById('action-input');
  const text  = input.value.trim();
  if (!text || G.isLoading) return;

  appendMsg('player', text);
  input.value = '';
  input.style.height = '';
  callDM(text);
}

function quickAction(text) {
  if (G.isLoading) return;
  const input = document.getElementById('action-input');
  input.value = text;
  sendAction();
}

/* ═══════════════════════════════════════════════════════
   INPUT CONTROLS
═══════════════════════════════════════════════════════ */
const actionInput = document.getElementById('action-input');

actionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendAction();
  }
});

actionInput.addEventListener('input', () => {
  actionInput.style.height = 'auto';
  actionInput.style.height = Math.min(actionInput.scrollHeight, 120) + 'px';
});
