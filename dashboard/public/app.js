const servers = [];
const COLORS = {
  green: '#3fb950',
  yellow: '#d29922',
  red: '#f85149',
  blue: '#58a6ff',
  purple: '#bc8cff',
  muted: '#8b949e',
};

function formatNum(n) {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
  return `${n}`;
}

function formatTokSec(n) {
  if (n >= 1000000) return (n / 1000000).toFixed(2) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
  return `${n}`;
}

function getBarColor(val) {
  if (val < 0.6) return 'green';
  if (val < 0.85) return 'yellow';
  return 'red';
}

function renderCard(item) {
  const gpuCache = item['kv_cache_usage_perc'] || 0;
  const running = item.num_requests_running || 0;
  const pending = item.num_requests_waiting || 0;
  const tokensPerSec = item.tokensPerSec || 0;
  const promptTok = item.prompt_tokens_total || 0;
  const genTok = item.generation_tokens_total || 0;

  const color = getBarColor(gpuCache);
  const statusClass = item.status === 'healthy' ? 'healthy' :
    item.status === 'unhealthy' ? 'unhealthy' : 'unknown';
  const statusLabel = item.status === 'healthy' ? 'Online' :
    item.status === 'unavailable' ? 'Offline' : 'Unknown';

  const lastUpdated = item.lastUpdated ?
    new Date(item.lastUpdated).toLocaleTimeString() : '—';

  const modelLine = [];
  if (item.rootModel) modelLine.push(item.rootModel);
  if (item.maxModelLen) modelLine.push(formatNum(item.maxModelLen) + ' tokens');
  if (item.modelId && item.rootModel) modelLine.push(item.modelId);

  return `
    <div class="server-card" data-url="${item.url}">
      <div class="server-header">
        <div class="server-url">${item.url} <span class="server-status ${statusClass}">${statusLabel}</span></div>
        ${modelLine.length ? `<div class="server-meta">${modelLine.join(' · ')}</div>` : ''}
        ${tokensPerSec > 0 ? `<div class="server-meta token-rate">Token Rate: <span class="tok-rate-value">${formatTokSec(tokensPerSec)}</span>/sec</div>` : ''}
      </div>
      <div class="status-bar-container">
        <div class="status-bar-label">
          <span>GPU KV Cache</span>
          <span>${(gpuCache * 100).toFixed(0)}%</span>
        </div>
        <div class="status-bar">
          <div class="status-bar-fill ${color}" style="width: ${Math.min(gpuCache * 100, 100)}%"></div>
        </div>
      </div>
      <div class="metrics-grid">
        <div class="metric-item">
          <div class="metric-value">${running}</div>
          <div class="metric-label">Running</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${pending}</div>
          <div class="metric-label">Waiting</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${formatTokSec(tokensPerSec)}</div>
          <div class="metric-label">Tok/sec</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${formatNum(promptTok)}</div>
          <div class="metric-label">Prompt Toks</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${formatNum(genTok)}</div>
          <div class="metric-label">Gen Tokens</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${(gpuCache * 100).toFixed(0)}%</div>
          <div class="metric-label">GPU Cache</div>
        </div>
      </div>
      <div class="sparkline-container">
        <canvas data-url="${item.url}" data-type="gpu" height="40"></canvas>
        <span class="sparkline-label">GPU KV Cache</span>
      </div>
      <div class="sparkline-container">
        <canvas data-url="${item.url}" data-type="toks" height="40"></canvas>
        <span class="sparkline-label">Tokens/sec</span>
      </div>
      <div class="timestamp">Updated: ${lastUpdated}</div>
    </div>
  `;
}

function renderSparkline(canvas, data, field, color) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const padding = 2;

  ctx.clearRect(0, 0, w, h);

  if (!data || data.length < 2) return;

  const values = data.map(d => d[field]).filter(v => v != null);
  if (values.length < 2) return;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const step = (w - padding * 2) / (values.length - 1);

  ctx.beginPath();
  ctx.moveTo(padding, h - padding - ((values[0] - min) / range) * (h - padding * 2));

  for (let i = 1; i < values.length; i++) {
    const x = padding + i * step;
    const y = h - padding - ((values[i] - min) / range) * (h - padding * 2);
    ctx.lineTo(x, y);
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color + '40');
  grad.addColorStop(1, color + '00');

  ctx.lineTo(padding + (values.length - 1) * step, h);
  ctx.lineTo(padding, h);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();
}

function initCanvases() {
  document.querySelectorAll('canvas[data-type="gpu"]').forEach(canvas => {
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width || 360;
    canvas.height = 40;
  });
  document.querySelectorAll('canvas[data-type="toks"]').forEach(canvas => {
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = canvas.closest('.sparkline-container').offsetWidth || 360;
    canvas.height = 40;
  });
}

function renderAllSparklines() {
  document.querySelectorAll('canvas[data-type="gpu"]').forEach(canvas => {
    const server = servers.find(s => s.url === canvas.dataset.url);
    if (server && server.history.length > 0) {
      renderSparkline(canvas, server.history, 'gpuCache', COLORS.blue);
    }
  });
  document.querySelectorAll('canvas[data-type="toks"]').forEach(canvas => {
    const server = servers.find(s => s.url === canvas.dataset.url);
    if (server && server.history.length > 0) {
      const vals = server.history.map(d => d.tokensPerSec).filter(v => v != null);
      if (vals.length >= 2) {
        renderSparkline(canvas, server.history, 'tokensPerSec', COLORS.green);
      }
    }
  });
}

function renderPoolStats(data) {
  const container = document.getElementById('pool-stats');
  if (!data.pool || !container) return;
  container.classList.add('visible');
  const p = data.pool;
  const healthyCount = servers.filter(s => s.status === 'healthy').length;

  container.innerHTML = `
    <div class="pool-stat">
      <span class="pool-stat-value">${formatTokSec(p.totalTokensPerSec)}</span>
      <span class="pool-stat-label">Tokens/sec</span>
    </div>
    <div class="pool-stat">
      <span class="pool-stat-value">${p.totalRunning}</span>
      <span class="pool-stat-label">Running</span>
    </div>
    <div class="pool-stat">
      <span class="pool-stat-value">${p.totalWaiting}</span>
      <span class="pool-stat-label">Waiting</span>
    </div>
    <div class="pool-stat">
      <span class="pool-stat-value ${healthyCount < p.serverCount ? 'pool-warning' : ''}">${healthyCount}/${p.serverCount}</span>
      <span class="pool-stat-label">Servers</span>
    </div>
  `;
}

function renderDashboard(data) {
  const grid = document.getElementById('server-grid');
  servers.length = 0;
  servers.push(...data.servers);

  renderPoolStats(data);
  grid.innerHTML = data.servers.map(item => renderCard(item)).join('');

  initCanvases();
  setTimeout(renderAllSparklines, 0);
}

function connectSSE() {
  let evtSource;
  let retryCount = 0;

  function connect() {
    evtSource = new EventSource('/stream');

    document.getElementById('status-text').textContent = 'Connecting...';
    const dot = document.querySelector('#connection-status .dot');
    dot.className = 'dot';

    evtSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (!data || !Array.isArray(data.servers)) return;
        const wasHealthy = servers.length > 0 &&
          servers.every(s => s.status === 'healthy');
        const isHealthy = data.servers.length > 0 &&
          data.servers.every(s => s.status === 'healthy');

        if (wasHealthy && !isHealthy) {
          dot.className = 'dot';
        } else if (isHealthy) {
          dot.className = 'dot connected';
          document.getElementById('status-text').textContent = 'Live';
        } else if (data.servers.length > 0) {
          dot.className = 'dot disconnected';
          document.getElementById('status-text').textContent = 'Some offline';
        }

        renderDashboard(data);
        retryCount = 0;
      } catch (e) {
        console.error('Parse error:', e);
      }
    };

    evtSource.onerror = () => {
      dot.className = 'dot disconnected';
      document.getElementById('status-text').textContent = 'Retrying...';
      evtSource.close();
      retryCount++;
      const delay = Math.min(1000 * Math.pow(2, retryCount), 30000);
      console.log(`Reconnecting in ${delay}ms...`);
      setTimeout(connect, delay);
    };
  }

  connect();
}

function closeModal() {
  document.getElementById('modal').classList.add('hidden');
}

document.getElementById('modal-close').addEventListener('click', closeModal);
document.getElementById('modal').addEventListener('click', (e) => {
  if (e.target === e.currentTarget) closeModal();
});

document.addEventListener('click', (e) => {
  const card = e.target.closest('.server-card[data-url]');
  if (!card) return;

  const item = servers.find(s => s.url === card.dataset.url);
  if (!item) return;

  const modal = document.getElementById('modal');
  document.getElementById('modal-title').textContent = item.url;

  fetch(`/api/history?url=${encodeURIComponent(item.url)}`)
    .then(r => r.json())
    .then(data => {
      const canvas = document.getElementById('modal-chart');
      canvas.width = canvas.parentElement.clientWidth - 48;
      canvas.height = 200;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const fields = [
        { key: 'running', color: COLORS.green, label: 'Running', max: 1 },
        { key: 'pending', color: COLORS.red, label: 'Waiting', max: 1 },
        { key: 'gpuCache', color: COLORS.blue, label: 'GPU Cache', max: 1 },
        { key: 'tokensPerSec', color: COLORS.purple, label: 'Tok/sec', max: null },
      ];

      const padding = 30;
      const w = canvas.width - padding * 2;
      const h = canvas.height - 10;

      fields.forEach(field => {
        const values = data.history.map(d => d[field.key]);
        if (!values || values.length < 2) return;

        const filtered = values.filter(v => v != null);
        if (filtered.length < 2) return;

        const min = field.max ? 0 : Math.min(...values);
        const max = field.max || Math.max(...values, 0.1);
        const range = max - min;

        ctx.beginPath();
        let started = false;
        values.forEach((v, i) => {
          if (v == null) return;
          const x = padding + i * step;
          const y = h - ((v - min) / range) * h;
          if (!started) { ctx.moveTo(x, y); started = true; }
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = field.color;
        ctx.lineWidth = 2;
        ctx.stroke();
      });

      // Recalculate with proper step
      step = w / (data.history.length - 1 || 1);

      fields.forEach(field => {
        const values = values.map(d => d[field.key]).filter(v => v != null);
        if (values.length < 2) return;

        const min = field.max ? 0 : Math.min(...values);
        const max = field.max || Math.max(...values, 0.1);
        const range = max - min || 1;

        ctx.beginPath();
        let started = false;
        data.history.forEach((d, i) => {
          if (d[field.key] == null) return;
          const x = padding + i * step;
          const y = h - ((d[field.key] - min) / range) * h;
          if (!started) { ctx.moveTo(x, y); started = true; }
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = field.color;
        ctx.lineWidth = 2;
        ctx.stroke();
      });

      const legend = document.createElement('div');
      legend.className = 'legend';
      legend.innerHTML = fields
        .filter(f => data.history.some(d => d[f.key] != null))
        .map(f => `
          <div class="legend-item">
            <div class="legend-dot" style="background:${f.color}"></div>
            <span>${f.label}</span>
          </div>
        `).join('');

      const existingLegend = document.querySelector('.legend');
      if (existingLegend) existingLegend.remove();
      document.querySelector('.modal-content').appendChild(legend);

      // Store fields and step for re-entry redraws
      window._modalFields = fields;
      window._modalStep = step;
    })
    .catch(() => {
      const canvas = document.getElementById('modal-chart');
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#8b949e';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No history available', canvas.width / 2, canvas.height / 2);
    });
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});

connectSSE();
