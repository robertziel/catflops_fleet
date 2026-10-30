// The dashboard's vLLM servers come from VLLM_SERVERS (comma-separated URLs) and from nowhere else.
// There is no stand-in: an unset or blank list is an error that names the variable, so a dashboard that
// was never told which boxes to poll cannot show a localhost server as if it were one.
// No dependencies, so the test runs on a bare node (tests/config.test.js).

function parseServers(value) {
  const urls = (value || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);
  if (urls.length === 0) {
    throw new Error('VLLM_SERVERS is not set: name the vLLM servers to poll as comma-separated URLs, ' +
      'e.g. VLLM_SERVERS=http://<box>:8000,http://<box2>:8000');
  }
  return urls;
}

// A value the dashboard cannot guess: the LB it shows (LB_URL) and the model name it labels (CHAT_MODEL).
// Unset or blank is an error naming the variable — there is no localhost or model-name stand-in.
function requireEnv(name, value, hint) {
  const v = (value || '').trim();
  if (!v) throw new Error(`${name} is not set: ${hint}`);
  return v;
}

module.exports = { parseServers, requireEnv };
