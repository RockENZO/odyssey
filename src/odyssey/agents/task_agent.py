from odyssey.llm.client import generate, get_fast_model
from odyssey.tools.registry import get_tool
from odyssey.storage.db import list_tasks, add_task


SYSTEM_PROMPT = """You are a task management assistant. Help users organize their work by adding,
listing, completing, and prioritizing tasks. Parse natural language to extract task details."""


async def handle_task(user_input: str) -> str:
    add_tool = get_tool("add_task")
    list_tool = get_tool("list_tasks")
    complete_tool = get_tool("complete_task")

    add_keywords = ["add", "create", "new task", "remind", "todo"]
    list_keywords = ["list", "show", "what", "pending", "all tasks"]
    done_keywords = ["done", "complete", "finish", "mark"]

    if any(kw in user_input.lower() for kw in add_keywords):
        parsed = await generate(
            system="""Extract task details from the user's request. Respond in this format:
TITLE: <task title>
PRIORITY: <high|medium|low>
DUE: <YYYY-MM-DD or empty>
""",
            prompt=user_input,
            temperature=0.1,
        )
        lines = parsed.strip().split("\n")
        title = ""
        priority = "medium"
        due = ""
        for line in lines:
            if line.startswith("TITLE:"):
                title = line[6:].strip()
            elif line.startswith("PRIORITY:"):
                priority = line[9:].strip()
            elif line.startswith("DUE:"):
                due = line[4:].strip()
        if title:
            return await add_tool.run(title=title, priority=priority, due_date=due)
        return "Could not parse task from your request."

    if any(kw in user_input.lower() for kw in done_keywords):
        import re
        match = re.search(r'(\d+)', user_input)
        if match:
            return await complete_tool.run(task_id=int(match.group(1)))
        tasks = list_tasks(status="pending")
        if not tasks:
            return "No pending tasks to complete."
        prompt = f"User wants to mark a task done: {user_input}\n\nPending tasks: {tasks}\nAsk which task ID."
        return await generate(system=SYSTEM_PROMPT, prompt=prompt)

    return await list_tool.run(status="all")
