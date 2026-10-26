"""GhostRigger KotorMCP server.

Provides an MCP-compatible server that exposes KotOR resource tools and
GhostRigger-specific rendering tools via:
  - stdio transport (for Claude Desktop / Claude Code integration)
  - HTTP-SSE transport (for programmatic access)
  - HTTP streamable transport

Usage:
    # stdio mode (default) — for Claude Desktop
    python -m kotormcp

    # HTTP mode on port 8765
    python -m kotormcp --mode http --port 8765

    # SSE mode
    python -m kotormcp --mode sse --port 8765

Environment variables:
    K1_PATH  — Path to KOTOR 1 installation
    K2_PATH  — Path to KOTOR 2 / TSL installation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Try to import MCP SDK ──────────────────────────────────────────────────────
_MCP_AVAILABLE = False
try:
    import mcp.server.stdio
    import mcp.server.sse
    from mcp.server.lowlevel import NotificationOptions, Server as MCPServer
    from mcp.server.models import InitializationOptions
    from mcp import types as mcp_types
    _MCP_AVAILABLE = True
except ImportError:
    MCPServer = None  # type: ignore[assignment, misc]
    mcp_types = None  # type: ignore[assignment, misc]

# ── Try uvicorn ────────────────────────────────────────────────────────────────
try:
    from uvicorn import Config as UvConfig, Server as UvServer
    _UVICORN_AVAILABLE = True
except ImportError:
    _UVICORN_AVAILABLE = False
    UvConfig = None  # type: ignore[assignment, misc]
    UvServer = None  # type: ignore[assignment, misc]

# ── Internal tool registry ────────────────────────────────────────────────────
# Add src to path so we can import GhostRigger modules
_SRC_DIR = str(Path(__file__).parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from kotormcp.tools import get_all_tools, handle_tool  # noqa: E402
from kotormcp import mcp_resources  # noqa: E402


# ── Fallback HTTP server (no MCP SDK required) ────────────────────────────────

class _FallbackHTTPServer:
    """
    A simple asyncio HTTP server that speaks a JSON RPC-like protocol
    compatible with the KotorMCP wire format.  Used when ``mcp`` is not installed.

    POST /tools/list     → {"tools": [...]}
    POST /tools/call     → {"name": "...", "arguments": {...}} → tool result
    GET  /health         → {"status": "ok", "version": "1.0.0"}
    GET  /resources/list → {"resources": [...]}
    POST /resources/read → {"uri": "kotor://..."} → resource content
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await reader.read(65536)
            request = raw.decode("utf-8", errors="replace")
            lines = request.split("\r\n")
            request_line = lines[0] if lines else ""
            parts = request_line.split(" ")
            if len(parts) < 2:
                writer.close()
                return
            method, path = parts[0], parts[1].split("?")[0]

            # Parse body
            body_start = request.find("\r\n\r\n")
            body_text = request[body_start + 4:] if body_start != -1 else ""
            body: Dict[str, Any] = {}
            if body_text.strip():
                try:
                    body = json.loads(body_text)
                except json.JSONDecodeError:
                    pass

            # Route
            status = "200 OK"
            content_type = "application/json"
            response_body: Any = {}

            if path == "/health":
                response_body = {"status": "ok", "version": "1.0.0", "server": "GhostRigger-KotorMCP"}

            elif path == "/tools/list":
                tools = get_all_tools()
                response_body = {"tools": tools}

            elif path == "/tools/call":
                tool_name = body.get("name", "")
                tool_args = body.get("arguments", {})
                try:
                    result = await handle_tool(tool_name, tool_args)
                    response_body = {"result": result}
                except ValueError as exc:
                    response_body = {"error": str(exc)}

            elif path == "/resources/list":
                resources = await mcp_resources.list_resources()
                templates = await mcp_resources.list_resource_templates()
                response_body = {"resources": resources, "resourceTemplates": templates}

            elif path == "/resources/read":
                uri = body.get("uri", "")
                try:
                    content = await mcp_resources.read_resource(uri)
                    response_body = {"content": content}
                except Exception as exc:
                    response_body = {"error": str(exc)}

            else:
                status = "404 Not Found"
                response_body = {"error": f"Unknown endpoint: {path}"}

            body_bytes = json.dumps(response_body, ensure_ascii=False, indent=2).encode("utf-8")
            headers = (
                f"HTTP/1.1 {status}\r\n"
                f"Content-Type: {content_type}; charset=utf-8\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
                f"Access-Control-Allow-Origin: *\r\n"
                "\r\n"
            )
            writer.write(headers.encode("ascii") + body_bytes)
            await writer.drain()
        except Exception as exc:
            log.error("HTTP handler error: %s", exc)
        finally:
            writer.close()

    async def serve(self) -> None:
        server = await asyncio.start_server(self.handle, self.host, self.port)
        addr = server.sockets[0].getsockname()
        print(f"GhostRigger KotorMCP HTTP server running on http://{addr[0]}:{addr[1]}", flush=True)
        print(f"  GET  http://{addr[0]}:{addr[1]}/health", flush=True)
        print(f"  POST http://{addr[0]}:{addr[1]}/tools/list", flush=True)
        print(f"  POST http://{addr[0]}:{addr[1]}/tools/call", flush=True)
        async with server:
            await server.serve_forever()


# ── MCP-SDK server ────────────────────────────────────────────────────────────

def _build_mcp_server():
    """Build and configure the MCP server (requires ``mcp`` package)."""
    SERVER = MCPServer("GhostRigger-KotorMCP")

    @SERVER.list_tools()
    async def list_tools() -> list:
        tools_raw = get_all_tools()
        result = []
        for t in tools_raw:
            result.append(mcp_types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
                annotations=t.get("annotations"),
            ))
        return result

    @SERVER.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]):
        raw = await handle_tool(name, arguments)
        text = raw.get("text", json.dumps(raw))
        content = [mcp_types.TextContent(type="text", text=text)]
        if raw.get("isError"):
            return mcp_types.CallToolResult(content=content, isError=True)
        return content

    @SERVER.list_resources()
    async def list_res():
        resources_raw = await mcp_resources.list_resources()
        return [
            mcp_types.Resource(
                uri=r["uri"],
                name=r["name"],
                description=r.get("description"),
                mimeType=r.get("mimeType"),
            )
            for r in resources_raw
        ]

    @SERVER.list_resource_templates()
    async def list_res_templates():
        templates_raw = await mcp_resources.list_resource_templates()
        return [
            mcp_types.ResourceTemplate(
                uriTemplate=r["uriTemplate"],
                name=r["name"],
                description=r.get("description"),
                mimeType=r.get("mimeType"),
            )
            for r in templates_raw
        ]

    @SERVER.read_resource()
    async def read_res(uri: str):
        return await mcp_resources.read_resource(uri)

    return SERVER


async def _run_stdio() -> None:
    SERVER = _build_mcp_server()
    async with mcp.server.stdio.stdio_server() as (r, w):
        await SERVER.run(
            r, w,
            InitializationOptions(
                server_name="GhostRigger-KotorMCP",
                server_version="1.0.0",
                capabilities=SERVER.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
                notification_options=NotificationOptions(),
            ),
        )


async def _run_sse(host: str, port: int) -> None:
    if not _UVICORN_AVAILABLE:
        raise ImportError("uvicorn required for SSE mode: pip install uvicorn[standard]")
    SERVER = _build_mcp_server()
    transport = mcp.server.sse.SseServerTransport(endpoint="/mcp")

    async def app(scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            method = scope.get("method", "")
            if method == "GET" and path == "/mcp":
                await transport.connect_sse(scope, receive, send)
            elif method == "POST" and path == "/mcp":
                await transport.handle_post_message(scope, receive, send)
            else:
                await send({"type": "http.response.start", "status": 404, "headers": []})
                await send({"type": "http.response.body", "body": b"Not Found"})

    uv = UvServer(UvConfig(app=app, host=host, port=port, log_level="info"))
    await uv.serve()


async def _run_http_fallback(host: str, port: int) -> None:
    """Run the built-in fallback HTTP server."""
    srv = _FallbackHTTPServer(host=host, port=port)
    await srv.serve()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m kotormcp",
        description="GhostRigger KotorMCP server",
    )
    parser.add_argument(
        "--mode",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport: stdio (Claude Desktop), sse (Server-Sent Events), http (simple JSON API)",
    )
    parser.add_argument("--host", default="localhost", help="Host for HTTP/SSE modes")
    parser.add_argument("--port", type=int, default=8765, help="Port for HTTP/SSE modes")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.mode == "stdio":
        if not _MCP_AVAILABLE:
            print(
                "ERROR: mcp package not installed. Install with: pip install mcp\n"
                "  For HTTP mode (no mcp required): python -m kotormcp --mode http",
                file=sys.stderr,
            )
            sys.exit(1)
        asyncio.run(_run_stdio())

    elif args.mode == "sse":
        if not _MCP_AVAILABLE:
            print("ERROR: mcp package required for SSE mode.", file=sys.stderr)
            sys.exit(1)
        asyncio.run(_run_sse(host=args.host, port=args.port))

    elif args.mode == "http":
        if _MCP_AVAILABLE:
            # Prefer real MCP SSE if available
            print(f"Running GhostRigger-KotorMCP HTTP server on {args.host}:{args.port}", flush=True)
        asyncio.run(_run_http_fallback(host=args.host, port=args.port))


if __name__ == "__main__":
    main()
