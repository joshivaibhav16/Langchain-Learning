# McpAgent.py
import asyncio
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
import McpClient  # your MCP client file
import os

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

# LangGraph workflow
workflow = StateGraph(State)
memory = InMemorySaver()

# LLM node — just returns AIMessage with possible tool calls
def llm_node(state: State):
    response = llm_with_tools.invoke(state["messages"])

    # Ensure response is an AIMessage with tool_calls metadata
    if not isinstance(response, AIMessage):
        response = AIMessage(
            content=response.content if hasattr(response, "content") else str(response),
            tool_calls=getattr(response, "tool_calls", [])
        )

    return {"messages": [response]}

# Add nodes
workflow.add_node("llm", llm_node)
workflow.add_node("tools", ToolNode(mcp_tools_list))

# Connect nodes
workflow.add_edge(START, "llm")
workflow.add_conditional_edges("llm", tools_condition)
workflow.add_edge("tools", "llm")
workflow.add_edge("llm", END)

# Compile graph
graph = workflow.compile(checkpointer=memory)

state = {
    "messages": [
        SystemMessage(content="""
        You are Scout, a software developer assistant connected to IntelliJ via MCP.
        You have the following tools: {tool_list}.
        When the user requests an IDE action (creating files, opening files, executing actions, retrieving open files),
        you MUST call the relevant tool directly using the tool calling format — no explanations, no code samples.
        Do not describe how to do it manually; instead, call the tool with correct arguments.
        """.format(tool_list=", ".join([t.name for t in mcp_tools_list])))
    ]
}

while True:
    in_message = input("Provide input: ").strip()
    if in_message.lower() in {"quit", "exit"}:
        break

    state["messages"].append(HumanMessage(content=in_message))
    response_state = graph.invoke(state, config={"configurable": {"thread_id": "1"}})

    ai_msg = response_state["messages"][-1]
    state["messages"].append(ai_msg)

    print(ai_msg.content)



