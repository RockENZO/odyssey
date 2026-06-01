import os
import subprocess
from pathlib import Path


async def glob(pattern: str, path: str | None = None) -> str:
    try:
        import glob as glob_mod
        search_path = path or "."
        full_pattern = os.path.join(search_path, pattern)
        matches = glob_mod.glob(full_pattern, recursive=True)
        if not matches:
            return "No files found matching that pattern."
        matches.sort()
        result = "\n".join(matches)
        if len(matches) > 100:
            result = result + f"\n... and {len(matches) - 100} more matches"
        return result
    except Exception as e:
        return f"Error in glob: {e}"


async def grep(pattern: str, path: str | None = None, include: str | None = None) -> str:
    try:
        search_path = path or os.getcwd()
        cmd = ["rg", "-n", "--no-heading", pattern, search_path]
        if include:
            cmd.extend(["--type", include] if include == "py" or include == "js" else ["--glob", f"*.{include}"])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            lines = output.split("\n")
            if len(lines) > 50:
                output = "\n".join(lines[:50]) + f"\n... and {len(lines) - 50} more results"
            return output or "No matches found."
        elif result.returncode == 1:
            return "No matches found."
        else:
            return f"Error running rg: {result.stderr.strip()}"
    except FileNotFoundError:
        return "Error: ripgrep (rg) not found. Install it with 'brew install ripgrep'"
    except Exception as e:
        return f"Error in grep: {e}"
