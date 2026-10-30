# vllm-usage-mcp — a READ-ONLY MCP server over the fleet's own usage data (README.md).
#
# An MCP host launches this over stdio (`docker run --rm -i`), so the image listens on nothing and
# publishes nothing. Its only outbound call is `GET` against three LB read endpoints; its only
# filesystem access is mode "r" under the state directory, which the compose file mounts `:ro`.
# pytest is in the image so the suite runs from it (repo convention, as in power/). The suite reads
# the fleet's REAL recorded data, so the full run needs the same two read-only mounts the server
# does — see README.md. Without them it skips what it cannot read rather than passing vacuously
# (26 passed / 65 skipped), which is the honest shape for a suite with no fixtures of its own.
FROM python:3.12-slim
# The official MCP Python SDK. Pinned: in 2.x FastMCP became MCPServer and the decorator surface
# changed, so an unpinned install would silently break the server's own imports.
RUN pip install --no-cache-dir "mcp==2.3.0" pytest
RUN useradd --uid 1000 --user-group --no-create-home --home-dir /nonexistent \
    --shell /usr/sbin/nologin usagemcp
COPY . /app
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    USAGE_MCP_LB_URL=http://127.0.0.1:8010 \
    USAGE_MCP_STATE_DIR=/data \
    USAGE_MCP_LIBRARY=/library
USER 1000:1000
CMD ["python", "-m", "usage_mcp"]
