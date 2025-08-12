# langgraph_agent.py
import asyncio
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

import McpClient  # your MCP client file

# Define conversation state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Function to load MCP tools synchronously
def load_mcp_tools():
    try:
        return asyncio.run(McpClient.main())
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(McpClient.main())

# Get tools from MCP servers
mcp_tools_list = load_mcp_tools()
print(f"Loaded MCP tools: {[t.name for t in mcp_tools_list]}")

# Initialize LLM with tools
llm = ChatOllama(model="llama3.1:8b")
llm_with_tools = llm.bind_tools(mcp_tools_list)

# Define LangGraph workflow
workflow = StateGraph(State)
memory = InMemorySaver()

# LLM node
def llm_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Add nodes
workflow.add_node("llm", llm_node)
workflow.add_node("tools", ToolNode(mcp_tools_list))

# Connect nodes
workflow.add_edge(START, "llm")
workflow.add_conditional_edges("llm", tools_condition)
workflow.add_edge("tools", "llm")
workflow.add_edge("llm", END)

# Compile the graph
graph = workflow.compile(checkpointer=memory)


# Run an example interaction
if __name__ == "__main__":

    system_prompt = """
    Your name is Scout and you are an software developer. You help customers manage their Java projects by leveraging the tools available to you.

    <filesystem>
    You have access to a set of tools that allow you to interact with the user's local filesystem. 
    You are only able to access files within the working directory `projects`. 
    The absolute path to this directory is: C:\\Users\\VaibhavJoshi\\Desktop\\Development
    If you try to access a file outside of this directory, you will receive an error.
    Always use absolute paths when specifying files.
    </filesystem>

    <tools>
    """
    mcp_tools_list
    """</tools>

    Assist the customer in all aspects of their data science workflow.
    """

    # Example input
    events = graph.stream(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Using Tool: create_new_file_with_text , create a Java class named DemoApp ")
            ]
        },
        config={"configurable": {"thread_id": "1"}},
        stream_mode="values"
    )

    for event in events:
        print(event)