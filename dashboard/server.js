const express = require('express');
const axios = require('axios');
const dotenv = require('dotenv');
const { parseServers, requireEnv } = require('./config');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

const serverUrls = parseServers(process.env.VLLM_SERVERS);

const POLL_INTERVAL = parseInt(process.env.POLL_INTERVAL) || 10000;
const MAX_HISTORY = 30;

const serverStates = new Map();
serverUrls.forEach(url => {
  serverStates.set(url, {
    lastUpdated: null,
    status: 'unknown',
    metrics: {},
    history: [],
    modelInfo: null,
  });
});

function parseMetrics(raw) {
  const result = {};
  const lines = raw.split('\n');
  for (const line of lines) {
    const clean = line.trim();
    if (!clean || clean.startsWith('#')) continue;
    const match = /^(vllm[\w:]+)\{?([^}]*)\}?\s+([\d.eE+\-]+)$/.exec(clean);
    if (match) {
      result[match[1]] = parseFloat(match[3]);
    }
  }
  return result;
}

function stripPrefix(metrics) {
  const out = {};
  for (const [key, val] of Object.entries(metrics)) {
    out[key.replace(/^vllm:/, '')] = val;
  }
  return out;
}

async function pollModelInfo(url) {
  try {
    const res = await axios.get(`${url}/v1/models`, { timeout: 5000 });
    const data = res.data;
    const models = Array.isArray(data) ? data : (data.data || []);
    if (models.length > 0) {
      const m = models[0];
      return { modelId: m.id, rootModel: m.root, maxModelLen: m.max_model_len };
    }
  } catch (e) {}
  return null;
}

function computeTokensPerSec(history) {
  if (!history || history.length < 2) return 0;
  const hist = history;
  if (hist[hist.length - 1]?.tokensPerSec != null) return hist[hist.length - 1].tokensPerSec;
  const curr = hist[hist.length - 1];
  const prev = hist[hist.length - 2];
  const dt = (curr.time - prev.time) / 1000;
  if (dt <= 0 || !curr.genTokens || !prev.genTokens) return 0;
  return Math.round((curr.genTokens - prev.genTokens) / dt);
}

function computePoolTotals() {
  let totalTokPerSec = 0, totalRunning = 0, totalWaiting = 0;
  serverStates.forEach(state => {
    totalTokPerSec += computeTokensPerSec(state.history) || 0;
    const m = state.metrics ? stripPrefix(state.metrics) : {};
    totalRunning += m.num_requests_running || 0;
    totalWaiting += m.num_requests_waiting || 0;
  });
  return {
    serverCount: serverUrls.length,
    totalRunning,
    totalWaiting,
    totalTokensPerSec: totalTokPerSec,
  };
}

async function pollServer(url) {
  const state = serverStates.get(url);
  if (!state) return;
  try {
    const res = await axios.get(`${url}/metrics`, { timeout: 5000 });
    const rawMetrics = parseMetrics(res.data);
    state.lastUpdated = new Date();
    state.status = 'healthy';
    state.metrics = rawMetrics;
    const m = stripPrefix(rawMetrics);
    state.history.push({
      time: state.lastUpdated,
      running: m.num_requests_running || 0,
      pending: m.num_requests_waiting || 0,
      gpuCache: m.kv_cache_usage_perc || 0,
      promptTokens: m.prompt_tokens_total || 0,
      genTokens: m.generation_tokens_total || 0,
    });
    if (state.history.length > MAX_HISTORY) state.history.shift();
    // Compute tokensPerSec from history delta
    const hist = state.history;
    if (hist.length >= 2) {
      const curr = hist[hist.length - 1];
      const prev = hist[hist.length - 2];
      const dt = (curr.time - prev.time) / 1000;
      if (dt > 0) {
        curr.tokensPerSec = Math.round((curr.genTokens - prev.genTokens) / dt);
      }
    }
  } catch (err) {
    state.status = 'unavailable';
    state.lastUpdated = new Date();
  }

  state.modelInfo = await pollModelInfo(url);
}

setInterval(() => {
  const promises = serverUrls.map(url => pollServer(url));
  Promise.allSettled(promises);
}, POLL_INTERVAL);

serverUrls.forEach(url => pollServer(url));

// ── The LB is the source of truth ────────────────────────────────────────────
// This app used to poll the vLLM boxes directly, which shows "down" for the NORMAL case
// (broker_parrot only runs vLLM on queue demand) and says nothing about the ollama fallback
// that is actually serving. The LB knows backends, leases, capacity, queue depth and the
// fallback, so it answers both panels here.
//
// Everything the BROWSER needs comes from THIS origin: the page never dials :8010 itself.
// That is what lets one tunnelled hostname behind one authenticated nginx vhost be enough,
// and it keeps the LB off the public internet entirely.
const LB_URL = requireEnv('LB_URL', process.env.LB_URL,
  'the gateway this dashboard shows, e.g. LB_URL=http://host.docker.internal:8010').replace(/\/$/, '');
const CHAT_MODEL = requireEnv('CHAT_MODEL', process.env.CHAT_MODEL,
  'the model name the chat panel labels and sends, e.g. CHAT_MODEL=<served model name>');
const CHAT_TIMEOUT_MS = parseInt(process.env.CHAT_TIMEOUT_MS) || 3600000;

app.use(express.json({ limit: '2mb' }));

app.get('/api/lb', async (req, res) => {
  try {
    const r = await axios.get(`${LB_URL}/lb/status`, { timeout: 8000 });
    res.json(r.data);
  } catch (e) {
    res.status(503).json({ error: `load balancer unreachable at ${LB_URL}: ${e.message}` });
  }
});

// Chat STREAMS. A turn can take minutes on this fleet, and a spinner with no output is
// exactly why a slow answer could not be told apart from a dead one.
app.post('/api/chat', async (req, res) => {
  const messages = Array.isArray(req.body && req.body.messages) ? req.body.messages : null;
  if (!messages || messages.length === 0) {
    return res.status(400).json({ error: 'messages[] required' });
  }
  const body = {
    model: req.body.model || CHAT_MODEL,
    max_tokens: Math.min(parseInt(req.body.max_tokens) || 1024, 8192),
    stream: true,
    messages,
  };
  // `low` keeps the reasoning block short: the fleet's default (`high`) maps to the model's
  // verbose tier and once spent ~6m47s thinking about the word "ping".
  if (req.body.effort) body.output_config = { effort: req.body.effort };

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  let upstream;
  try {
    upstream = await axios.post(`${LB_URL}/v1/messages`, body, {
      responseType: 'stream', timeout: CHAT_TIMEOUT_MS,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (e) {
    res.write(`event: error\ndata: ${JSON.stringify({ error: e.message })}\n\n`);
    return res.end();
  }
  req.on('close', () => { try { upstream.data.destroy(); } catch (_) {} });
  upstream.data.on('data', chunk => res.write(chunk));
  upstream.data.on('end', () => res.end());
  upstream.data.on('error', e => {
    res.write(`event: error\ndata: ${JSON.stringify({ error: e.message })}\n\n`);
    res.end();
  });
});

app.use(express.static('public'));

function serverResponse(url) {
  const state = serverStates.get(url);
  const metrics = stripPrefix(state?.metrics ?? {});
  const tokensPerSec = computeTokensPerSec(state?.history);
  return {
    url,
    status: state ? state.status : 'unknown',
    lastUpdated: state?.lastUpdated?.toISOString() || null,
    ...metrics,
    modelId: state?.modelInfo?.modelId || '',
    rootModel: state?.modelInfo?.rootModel || '',
    maxModelLen: state?.modelInfo?.maxModelLen || null,
    tokensPerSec,
  };
}

app.get('/api/servers', (req, res) => {
  const servers = serverUrls.map(url => serverResponse(url));
  res.json({ servers, pool: computePoolTotals() });
});

app.get('/api/history', (req, res) => {
  const url = req.query.url;
  if (!url || !serverStates.get(url)) {
    return res.status(400).json({ error: 'Invalid server URL' });
  }
  res.json({
    url,
    history: serverStates.get(url).history.slice(0, MAX_HISTORY),
  });
});

app.get('/stream', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  const send = () => {
    const servers = serverUrls.map(url => serverResponse(url));
    res.write(`data: ${JSON.stringify({ servers, pool: computePoolTotals() })}\n\n`);
  };

  send();
  const interval = setInterval(send, POLL_INTERVAL / 2);

  req.on('close', () => clearInterval(interval));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`vLLM Dashboard: http://0.0.0.0:${PORT}`);
  console.log(`Servers: ${serverUrls.join(', ')}`);
  console.log(`Poll interval: ${POLL_INTERVAL}ms`);
});
