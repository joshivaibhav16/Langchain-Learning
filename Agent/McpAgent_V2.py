# McpAgent_V2.py
import asyncio
from typing import TypedDict, Annotated

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
import McpClient_V2  # Updated import for sync client

# Define conversation state
class State(TypedDict):
    messages: Annotated[list, add_messages]


def wrap_tool_as_sync(tool):
    def sync_wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        # Use ainvoke for async-safe invocation (handles if tool is async)
        return loop.run_until_complete(tool.ainvoke(kwargs))
    # Create new tool with sync func
    return StructuredTool.from_function(
        func=sync_wrapper,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )

def load_mcp_tools():
    tools = McpClient_V2.main()  # Fetch raw tools
    wrapped_tools = []
    for t in tools:
        # Skip if tool lacks invocation capability
        if not hasattr(t, 'ainvoke') or t.ainvoke is None:
            print(f"Warning: Skipping tool {t.name} - No invocation method available")
            continue
        wrapped_tools.append(wrap_tool_as_sync(t))
    return wrapped_tools
# def wrap_tool_as_sync(tool):
#     def sync_wrapper(*args, **kwargs):
#         try:
#             # Try to get the current event loop (works in main thread)
#             loop = asyncio.get_event_loop()
#         except RuntimeError:
#             # If we're in a worker thread, no loop exists → make one
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#         return loop.run_until_complete(tool.func(*args, **kwargs))
#
#     return StructuredTool.from_function(
#         func=sync_wrapper,
#         name=tool.name,
#         description=tool.description,
#         args_schema=tool.args_schema
#     )
#
#
# #Load MCP tools and wrap them
# def load_mcp_tools():
#     tools = McpClient_V2.main()  # already sync
#     return [wrap_tool_as_sync(t) for t in tools]


# Get tools from MCP servers
mcp_tools_list = load_mcp_tools()
print(f"Loaded MCP tools: {[t.name for t in mcp_tools_list]}")

# Initialize LLM with tools
llm = ChatOllama(
   model="qwen3:8b",
   temperature=0,
)

# Print tool schemas for debugging
print("\nTool schemas:")
for tool in mcp_tools_list:
    print(f"- {tool.name}: {getattr(tool, 'args_schema', 'No schema')}")

llm_with_tools = llm.bind_tools(mcp_tools_list)

# LangGraph workflow
workflow = StateGraph(State)
memory = InMemorySaver()


def llm_node(state: State):
    messages = state["messages"]

    if messages and isinstance(messages[-1], HumanMessage):
        enhanced_messages = messages[:-1] + [
            HumanMessage(
                content=f"{messages[-1].content}\n\nIMPORTANT: You must call the appropriate tool directly. Do not provide explanations or code examples - just call the tool with the correct parameters."
            )
        ]
    else:
        enhanced_messages = messages

    response = asyncio.get_event_loop().run_until_complete(llm_with_tools.ainvoke(enhanced_messages))
    print(f"Response content from llm invoke: {response}")

    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"Tool calls detected: {response.tool_calls}")
    else:
        print("No tool calls detected in response")
        print(f"Response content: {response.content}")

    if not isinstance(response, AIMessage):
        response = AIMessage(
            content=response.content if hasattr(response, "content") else str(response),
            tool_calls=getattr(response, "tool_calls", [])
        )

    return {"messages": [response]}


workflow.add_node("llm", llm_node)
workflow.add_node("tools", ToolNode(mcp_tools_list))

workflow.add_edge(START, "llm")
workflow.add_conditional_edges(
    "llm",
    tools_condition,
    {
        "tools": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "llm")

graph = workflow.compile(checkpointer=memory)

system_message = SystemMessage(content=f"""
You are Scout, a software developer assistant connected to IntelliJ via MCP tools.

Available tools: {', '.join([t.name for t in mcp_tools_list])}

CRITICAL INSTRUCTIONS:
1. When the user requests IDE actions, you MUST use the available tools
2. For creating files, use the "create_new_file_with_text" tool with parameters: pathInProject and text
3. For opening files, use the "open_file_in_editor" tool
4. For getting file content, use "get_open_in_editor_file_text" tool
5. Always call the appropriate tool - do not provide manual instructions

Tool calling examples:
- To create Test.java: Use create_new_file_with_text with pathInProject="Test.java" and text="your code here"
- To open a file: Use open_file_in_editor with filePath="filename"
- To get current file: Use get_open_in_editor_file_text

You must call tools when requested, not provide explanations.
""")

state = {
    "messages": [system_message]
}

print("Scout AI Agent connected to IntelliJ. Type 'quit' or 'exit' to end.")
print("Available tools:", [t.name for t in mcp_tools_list])
print("-" * 50)

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"quit", "exit"}:
        break

    if not user_input:
        continue

    state["messages"].append(HumanMessage(content=user_input))

    try:
        response_state = graph.invoke(state, config={"configurable": {"thread_id": "1"}})
        ai_msg = response_state["messages"][-1]
        state = response_state

        if ai_msg.content:
            print(f"Scout: {ai_msg.content}")
        else:
            print("Scout: Task completed.")

    except Exception as e:
        print(f"Error: {e}")
        print("Please try again.")
