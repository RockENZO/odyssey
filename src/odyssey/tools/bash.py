import asyncio
import os
import subprocess


_working_dir: str = os.getcwd()
_lock = asyncio.Lock()


async def bash(command: str, timeout: int = 120) -> str:
    async with _lock:
        try:
            global _working_dir
            loop = asyncio.get_event_loop()

            def _run():
                return subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=_working_dir,
                    env={**os.environ, "TERM": "dumb"},
                )

            result = await loop.run_in_executor(None, _run)
            output = (result.stdout or "") + (result.stderr or "")
            output = output.strip()
            if not output:
                if result.returncode == 0:
                    return "[command completed successfully with no output]"
                return f"[exit code {result.returncode}]"
            if len(output) > 30000:
                output = output[:30000] + "\n... [output truncated at 30000 characters]"
            return output
        except subprocess.TimeoutExpired:
            return f"[command timed out after {timeout}s]"
        except Exception as e:
            return f"Error executing command: {e}"
