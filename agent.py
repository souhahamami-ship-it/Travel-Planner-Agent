import os

from dotenv import load_dotenv

# Must run BEFORE importing Tools.*, in case a tool reads an API key at import time.
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from prompts import build_system_prompt
from Tools.tools import tools_list, tools_map

MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")  # verify against Groq's model list
MAX_STEPS = 8            # max model calls per user request
MAX_RESULT_CHARS = 4000  # cap tool output so it can't flood the context window

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

llm = ChatGroq(model=MODEL)  # reads GROQ_API_KEY from the environment
llm_with_tools = llm.bind_tools(tools_list)


def new_conversation() -> list:
    """Start a conversation. Reuse the returned list across turns for memory."""
    return [SystemMessage(content=build_system_prompt())]


def execute_tool_call(tool_call: dict) -> str:
    name = tool_call["name"]
    args = tool_call["args"]
    print(f"[tool] {name}({args})")

    tool = tools_map.get(name)
    if tool is None:
        return f"Tool '{name}' does not exist."

    try:
        result = str(tool.invoke(args))
    except Exception as e:
        print(f"[tool error] {name}: {e}")
        return f"Error executing {name}: {e}"

    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS] + "\n...[truncated]"
    return result


def run_agent(user_request: str, messages: list) -> str:
    """Handle one user turn. `messages` is updated in place (conversation memory)."""
    messages.append(HumanMessage(content=user_request))

    for _ in range(MAX_STEPS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            messages.append(
                ToolMessage(
                    content=execute_tool_call(tool_call),
                    tool_call_id=tool_call["id"],
                )
            )

    return "Sorry, I couldn't finish planning within the step limit. Try a simpler request."