from odyssey.llm.client import generate, get_fast_model
from odyssey.tools.registry import get_all_tools, get_tool_schemas
from odyssey.core.state import AgentState


CLASSIFY_SYSTEM = """You are a classifier that routes user requests to the right agent.
Respond with exactly one word from: research, memory, journal, task, briefing, chat.

- research: The user wants to search the web, learn about a topic, research something
- memory: The user wants to save or retrieve information from memory ("remember", "recall")
- journal: The user wants to write a journal entry or see journal summaries
- task: The user wants to manage tasks (add, list, complete, etc.)
- briefing: The user wants a daily briefing or summary
- chat: Anything else - general conversation
"""


async def classify_intent(user_input: str) -> str:
    intent = await generate(
        system=CLASSIFY_SYSTEM,
        prompt=f"Classify this request: {user_input}",
        temperature=0.1,
    )
    intent = intent.strip().lower()
    valid = {"research", "memory", "journal", "task", "briefing", "chat"}
    return intent if intent in valid else "chat"


async def run_agent(user_input: str, state: AgentState | None = None) -> AgentState:
    if state is None:
        state = AgentState(user_input=user_input)
    state.user_input = user_input

    intent = await classify_intent(user_input)
    state.active_agent = intent
    state.messages.append({"role": "user", "content": user_input})

    tools = get_tool_schemas()
    tools_description = "\n".join(
        f"- {t['function']['name']}: {t['function']['description']}"
        for t in tools
    )

    system = f"""You are Odyssey, a helpful personal AI assistant that runs locally.
Your current role is: {intent}.

Available tools:
{tools_description}

Respond helpfully and concisely. Use tools when appropriate to fulfill the user's request.
If you use a tool, explain what you found briefly."""

    response = await generate(
        system=system,
        prompt=user_input,
        model=get_fast_model(),
        temperature=0.7,
    )

    state.final_output = response
    state.messages.append({"role": "assistant", "content": response})
    return state
