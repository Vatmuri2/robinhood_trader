#!/usr/bin/env python3
"""
Step 0: Robinhood MCP connectivity explorer.

Run this once (interactively, with a browser available if OAuth is needed)
to confirm what tools the server exposes and what their exact schemas are.

Usage:
  python test_mcp.py

Auth: reads email/password from env vars (same as .mcp.json):
  export email="your@email.com"
  export password="yourpassword"

The Robinhood MCP server at https://agent.robinhood.com/mcp/trading uses
X-Email / X-Password headers. If the server instead initiates an OAuth flow
(e.g. returns a 401 with WWW-Authenticate), this script will print what it
sees so you can handle it manually.

Requires: pip install mcp
"""

import asyncio
import json
import os
import sys

RH_MCP_URL = "https://agent.robinhood.com/mcp/trading"


async def explore(email: str, password: str) -> None:
    auth_headers = {
        "X-Email": email,
        "X-Password": password,
    }

    session = None

    # ── Try Streamable HTTP first (newer MCP transport) ────────────────────────
    try:
        from mcp.client.streamable_http import streamablehttp_client
        from mcp import ClientSession

        print(f"Trying Streamable HTTP → {RH_MCP_URL}")
        async with streamablehttp_client(RH_MCP_URL, headers=auth_headers) as (r, w, _):
            async with ClientSession(r, w) as sess:
                await sess.initialize()
                print("Connected via Streamable HTTP.\n")
                await print_server_info(sess)
        return
    except ImportError:
        print("streamablehttp_client not available in this mcp version; trying SSE...")
    except Exception as e:
        print(f"Streamable HTTP failed: {type(e).__name__}: {e}")
        print("Trying SSE transport...\n")

    # ── Fall back to SSE ───────────────────────────────────────────────────────
    try:
        from mcp.client.sse import sse_client
        from mcp import ClientSession

        print(f"Trying SSE → {RH_MCP_URL}")
        async with sse_client(RH_MCP_URL, headers=auth_headers) as (r, w):
            async with ClientSession(r, w) as sess:
                await sess.initialize()
                print("Connected via SSE.\n")
                await print_server_info(sess)
        return
    except ImportError:
        print("sse_client not available. Is the mcp package installed?  pip install mcp")
    except Exception as e:
        print(f"SSE failed: {type(e).__name__}: {e}")

    print("\nBoth transports failed. Check credentials and network access.")
    sys.exit(1)


async def print_server_info(session) -> None:
    # ── List tools ─────────────────────────────────────────────────────────────
    tools_result = await session.list_tools()
    tools = tools_result.tools

    print(f"{'='*60}")
    print(f"Available tools ({len(tools)}):")
    print(f"{'='*60}")
    for t in tools:
        print(f"\n  Tool: {t.name}")
        if t.description:
            # Wrap long descriptions
            desc = t.description.replace("\n", " ")
            print(f"  Desc: {desc[:120]}{'...' if len(desc) > 120 else ''}")
        schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
        if schema:
            print(f"  Schema: {json.dumps(schema, indent=4)}")

    # ── Read-only test calls ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Read-only test calls:")
    print(f"{'='*60}")

    tool_names = {t.name for t in tools}

    for tool_name, args in [
        ("get_accounts",          {}),
        ("get_portfolio",         {}),
        ("get_equity_positions",  {}),
        ("get_equity_quotes",     {"symbols": ["SOXL"]}),
        ("get_equity_tradability",{"symbol": "SOXL"}),
    ]:
        if tool_name not in tool_names:
            print(f"\n  {tool_name}: (not found on server)")
            continue
        print(f"\n  Calling {tool_name}({json.dumps(args) if args else ''})...")
        try:
            result = await session.call_tool(tool_name, args)
            for block in result.content:
                text = getattr(block, "text", None) or str(block)
                # Try to pretty-print JSON
                try:
                    parsed = json.loads(text)
                    print(json.dumps(parsed, indent=2))
                except Exception:
                    print(text[:800])
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print("Exploration complete.")
    print(f"{'='*60}")


def main() -> None:
    email = os.environ.get("email", "")
    password = os.environ.get("password", "")

    if not email or not password:
        print("ERROR: Set email and password environment variables.")
        print("  export email='your@email.com'")
        print("  export password='yourpassword'")
        sys.exit(1)

    # Verify mcp is installed
    try:
        import mcp
        print(f"mcp SDK version: {getattr(mcp, '__version__', 'unknown')}")
    except ImportError:
        print("ERROR: mcp package not installed. Run: pip install mcp")
        sys.exit(1)

    asyncio.run(explore(email, password))


if __name__ == "__main__":
    main()
