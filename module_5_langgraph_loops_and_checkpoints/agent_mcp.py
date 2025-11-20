from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import List, Any

_mcp_client: MultiServerMCPClient | None = None

def get_mcp_client() -> MultiServerMCPClient:
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MultiServerMCPClient(
            {
                "langchain-mcp-server": {
                    "transport": "streamable_http",
                    "url": "https://docs.langchain.com/mcp"
                }
            }
        )
    return _mcp_client

async def get_mcp_tools():
    try:
        return await get_mcp_client().get_tools()
    except Exception as e:
        print(f"Error getting MCP tools: {e}")
        return []