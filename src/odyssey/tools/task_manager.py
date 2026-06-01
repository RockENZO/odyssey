from odyssey.storage.db import add_task, list_tasks, complete_task, delete_task, get_task


async def add_task_tool(title: str, priority: str = "medium", due_date: str = "") -> str:
    try:
        task_id = add_task(title=title, priority=priority, due_date=due_date or None)
        return f"Task added (id={task_id}): {title} [priority={priority}]"
    except Exception as e:
        return f"Failed to add task: {e}"


async def list_tasks_tool(status: str = "pending") -> str:
    try:
        s = None if status == "all" else status
        tasks = list_tasks(status=s)
        if not tasks:
            return "No tasks found."
        lines = ["Your tasks:\n"]
        for t in tasks:
            due = f" due: {t['due_date']}" if t["due_date"] else ""
            tags = f" [{t['tags']}]" if t["tags"] else ""
            lines.append(f"  [{t['id']}] {'✅' if t['status'] == 'done' else '⬜'} {t['title']} ({t['priority']}){due}{tags}")
        return "\n".join(lines)
    except Exception as e:
        return f"Failed to list tasks: {e}"


async def complete_task_tool(task_id: int) -> str:
    try:
        if complete_task(task_id):
            return f"Task {task_id} marked as done."
        return f"Task {task_id} not found."
    except Exception as e:
        return f"Failed to complete task: {e}"
