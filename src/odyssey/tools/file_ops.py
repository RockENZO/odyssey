import os
from pathlib import Path


def _resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


async def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
    try:
        fpath = _resolve_path(path)
        if not os.path.isfile(fpath):
            return f"Error: file not found: {path}"
        with open(fpath, "r", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        start = offset
        end = offset + limit if limit else total
        selected = lines[start:end]
        result = []
        for i, line in enumerate(selected, start + 1):
            result.append(f"{i}:{line}")
        output = "".join(result)
        if end < total:
            output += f"\n... ({total - end} more lines)"
        return output
    except Exception as e:
        return f"Error reading file: {e}"


async def write_file(path: str, content: str) -> str:
    try:
        fpath = _resolve_path(path)
        parent = os.path.dirname(fpath)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with open(fpath, "w") as f:
            f.write(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


async def edit_file(path: str, old_string: str, new_string: str) -> str:
    try:
        fpath = _resolve_path(path)
        if not os.path.isfile(fpath):
            return f"Error: file not found: {path}"
        with open(fpath, "r", errors="replace") as f:
            content = f.read()
        if old_string not in content:
            return f"Error: could not find the specified text in {path}"
        count = content.count(old_string)
        if count > 1:
            return f"Error: found {count} matches. Provide more context to identify the correct match."
        content = content.replace(old_string, new_string, 1)
        with open(fpath, "w") as f:
            f.write(content)
        return f"Applied edit to {path} ({len(old_string)} chars → {len(new_string)} chars)"
    except Exception as e:
        return f"Error editing file: {e}"
