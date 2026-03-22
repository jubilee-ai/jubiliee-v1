"""
Entry point: python -m mcp_server [--stdio] [--port PORT]

  --stdio   Use stdio transport (for local Cursor / Claude Desktop)
  --port N  Set HTTP port (default: $PORT env or 8080)

Railway / Docker: set START_COMMAND="python -m mcp_server" and the
server reads $PORT automatically.
"""

import os
import sys

from mcp_server.server import mcp

if __name__ == "__main__":
    transport = "stdio" if "--stdio" in sys.argv else "streamable-http"

    if transport == "streamable-http":
        port = int(os.environ.get("PORT", "8080"))
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        os.environ["PORT"] = str(port)

    mcp.run(transport=transport)
