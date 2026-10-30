/* `?server=<name>`: THE PERFORMANCE AND SESSIONS PAGES ARE ONE SERVER'S (operator, 2026-10-17: "server should have
 * these pages too") — ONE helper both pages load before their own script.
 *
 * Every perf window and boot line names the server that recorded it (perflog `cluster`, 2026-10-20), and /lb/perf and
 * /lb/sessions take `server=<name>`: the API answers that server's minutes alone, whether one server runs or several
 * (a line from before the name is the one server's that may run its model, perflog.of_cluster). A name the fleet does
 * not declare is the API's 404, naming the declared ones — the page shows it, never another server's minutes. */

/** The `?server=` the page was opened with, or null. */
function serverName() {
  if (typeof location === 'undefined') return null;
  return new URLSearchParams(location.search).get('server') || null;
}

/** What the page appends to its /lb/perf or /lb/sessions query: `&server=<name>`, or nothing unscoped. */
function serverQuery() {
  const name = serverName();
  return name ? `&server=${encodeURIComponent(name)}` : '';
}

/** Names the page's server: its title and the `#server-only` line, with the way back to every server. Returns the
 *  name, or null unscoped. */
function serverScope(title) {
  const name = serverName();
  if (!name) return null;
  const esc = (x) => String(x).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  document.title = `${title} · ${name}`;
  const el = document.getElementById('server-only');
  if (el) {
    el.hidden = false;
    el.innerHTML = `<b>${esc(name)}</b> · this server's minutes · <a href="${esc(location.pathname)}">every server</a>`;
  }
  return name;
}
