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

    async def get_tools(self):
        """
        Return MCP tools for LangGraph usage.
        No explicit start/stop — handled by the adapter.
        """
        return await self.client.get_tools()


# ----------- Example usage -----------

async def main():
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

    # Get MCP tools (usable in LangGraph)
    return await mcp_client.get_tools()

# Run the async main
if __name__ == "__main__":
    asyncio.run(main())