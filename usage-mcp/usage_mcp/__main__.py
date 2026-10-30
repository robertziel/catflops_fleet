"""Run the server on stdio - the transport an MCP host launches a local server over.

    docker run --rm -i ... vllm-usage-mcp:latest

Nothing is listened on, nothing is published: the host owns the pipe. The server reads
USAGE_MCP_LB_URL, USAGE_MCP_STATE_DIR, USAGE_MCP_LIBRARY and USAGE_MCP_TIMEOUT_S from the
environment (see readonly.Sources) and nothing else - in particular no .env file and no credential.
"""
from __future__ import annotations

from .readonly import Sources
from .server import build


def main() -> None:
    build(Sources.from_env()).run("stdio")


if __name__ == "__main__":
    main()
