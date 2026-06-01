async def task(description: str) -> str:
    from odyssey.core.state import Conversation
    from odyssey.core.tool_loop import run_tool_loop
    sub_conv = Conversation(max_tool_rounds=10)
    result = await run_tool_loop(description, sub_conv)
    return result
