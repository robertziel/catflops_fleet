"""usage-mcp — a READ-ONLY MCP server over the fleet's own usage data.

The data is the vLLM/SGLang gateway's: what the two-box pair has served, how fast, under which
configuration, and what each cluster start and benchmark swarm did. Nothing here reaches an
Anthropic account: the fleet keeps no record of the operator's own Claude subscription (see
`inventory.py`), so no tool claims to report one.

Read-only is structural, not conventional (see `readonly.py`): the one outbound function issues the
literal verb "GET" against a frozen three-path allowlist, the one file function opens mode "r"
under a frozen root, and `tests/test_readonly.py` parses every module in this package to prove no
other shape exists.
"""
__version__ = "1.0.0-beta.2"
