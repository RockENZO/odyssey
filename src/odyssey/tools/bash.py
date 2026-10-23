import asyncio
import os
import math
import signal


_working_dir: str = os.getcwd()
_lock = asyncio.Lock()


async def bash(command: str, timeout: int = 120) -> str:
    async with _lock:
        try:
            global _working_dir
            if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                    or not math.isfinite(timeout) or timeout <= 0):
                return "Error executing command: timeout must be finite and positive"
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=_working_dir,
                env={**os.environ, "TERM": "dumb"},
                start_new_session=(os.name == "posix"),
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # Killing the shell alone leaves its children running. Each POSIX
                # invocation owns a session, so stop that group before unlocking.
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                await process.communicate()
                raise
            output = (stdout.decode(errors="replace") + stderr.decode(errors="replace"))
            output = output.replace("\r\n", "\n").replace("\r", "\n")
            output = output.strip()
            if not output:
                if process.returncode == 0:
                    return "[command completed successfully with no output]"
                return f"[exit code {process.returncode}]"
            if len(output) > 30000:
                output = output[:30000] + "\n... [output truncated at 30000 characters]"
            return output
        except asyncio.TimeoutError:
            return f"[command timed out after {timeout}s]"
        except Exception as e:
            return f"Error executing command: {e}"
