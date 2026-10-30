# Security

Report a vulnerability privately: use the published repository's "Report a vulnerability" (GitHub
private security advisories). Do not open a public issue for it. Include the service (lb, power,
usage-mcp, deploy), the commit, and the smallest request or file that shows the problem.

What this project defends, and how, is in README.md, "Security model". In short: every service binds
127.0.0.1; the gateway has no authentication of its own and must sit behind one; every HTTP service
answers only to the Host names it is really reached by (its loopback names plus the required
`LB_ALLOWED_HOSTS` / `POWER_ALLOWED_HOSTS`: a DNS-rebound page is refused 421) and refuses a write from
an origin outside them; state-changing endpoints accept JSON only (no cross-site simple request reaches
them); the power panel fails closed when unconfigured; nothing ships a credential, a key or a default
secret.
