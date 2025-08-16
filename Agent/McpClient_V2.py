import json
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


class LangGraphMCPClient:
    """
    Wrapper for MultiServerMCPClient that:
      - Configures MCP servers from JSON/dict
      - Returns available tools for LangGraph
    """
    def __init__(self, mcp_config):
        """
        :param mcp_config: dict or JSON string defining MCP servers
        """
        if isinstance(mcp_config, str):
            mcp_config = json.loads(mcp_config)
        self.mcp_config = mcp_config
        self.client = MultiServerMCPClient(mcp_config)

    def get_tools(self):
        """
        Synchronous wrapper to retrieve MCP tools.
        """
        return asyncio.get_event_loop().run_until_complete(self.client.get_tools())


# ----------- Example usage -----------

def main():
    mcp_json_config = """
    {
      "jetbrains": {
        "command": "npx",
        "args": ["-y", "@jetbrains/mcp-proxy"],
        "transport": "stdio"
      }
    }
    """

    mcp_client = LangGraphMCPClient(mcp_json_config)

    try:
        tools = mcp_client.get_tools()
        print(f"Retrieved {len(tools)} tools from MCP servers")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        return tools
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        return []