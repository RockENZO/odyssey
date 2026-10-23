import asyncio
import locale
import math
import os
import signal
import subprocess
import tempfile


_working_dir: str = os.getcwd()
_lock = asyncio.Lock()


async def _wait(process):
    # No asyncio subprocess transport: inherited pipes in a detached descendant
    # must not prevent reaping the direct child.
    while process.poll() is None:
        await asyncio.sleep(0.02)


async def _stop(process):
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.kill()
    try:
        await asyncio.wait_for(_wait(process), 2)
    except asyncio.TimeoutError as error:
        raise RuntimeError("command cleanup could not reap the direct process within 2s") from error


async def _finish_cleanup(process, interrupted):
    cleanup = asyncio.create_task(_stop(process))
    cancelled = isinstance(interrupted, asyncio.CancelledError)
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break  # Re-raise cleanup failure below, preserving cancellation.
    try:
        cleanup.result()
    finally:
        if cancelled:
            raise asyncio.CancelledError()
    raise interrupted


async def bash(command: str, timeout: int = 120) -> str:
    async with _lock:
        try:
            if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                    or not math.isfinite(timeout) or timeout <= 0):
                return "Error executing command: timeout must be finite and positive"
            # Regular temporary files avoid pipe EOF waits on descendants that
            # escape the group. Creation is synchronous, with no cancellation
            # point between spawning a session and taking ownership of it.
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                process = subprocess.Popen(
                    command, shell=True, stdout=stdout, stderr=stderr,
                    cwd=_working_dir, env={**os.environ, "TERM": "dumb"},
                    start_new_session=(os.name == "posix"),
                )
                try:
                    await asyncio.wait_for(_wait(process), timeout)
                except (asyncio.TimeoutError, asyncio.CancelledError) as interrupted:
                    await _finish_cleanup(process, interrupted)
                stdout.seek(0)
                stderr.seek(0)
                encoding = locale.getpreferredencoding(False)
                output = (stdout.read(os.fstat(stdout.fileno()).st_size).decode(encoding, errors="replace")
                          + stderr.read(os.fstat(stderr.fileno()).st_size).decode(encoding, errors="replace"))
            output = output.replace("\r\n", "\n").replace("\r", "\n").strip()
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
