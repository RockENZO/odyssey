import asyncio
import locale
import math
import os
import signal
import subprocess
import tempfile


_working_dir: str = os.getcwd()
_lock = asyncio.Lock()

MAX_OUTPUT_CHARS = 30_000
# Four bytes is the widest UTF-8 encoding of one character, so this many bytes
# can always produce the visible cap without loading unbounded output at once.
MAX_OUTPUT_READ_BYTES = MAX_OUTPUT_CHARS * 4


def _read_captured_output(stream, encoding):
    """Read at most enough captured bytes for the display cap.

    Commands may write arbitrarily large output to the temporary files. Reading
    the whole file would copy all of it into memory (and decode it) before the
    display cap is applied, so bound the read and report whether more exists.
    The temporary files can still consume disk while the command runs.
    """
    size = os.fstat(stream.fileno()).st_size
    stream.seek(0)
    data = stream.read(min(size, MAX_OUTPUT_READ_BYTES))
    return data.decode(encoding, errors="replace"), size > MAX_OUTPUT_READ_BYTES


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
    return (await run_bash(command, timeout))[0]


async def run_bash(command: str, timeout: int = 120) -> tuple[str, bool]:
    """Run a command and return (output text, succeeded). Succeeded means exit code 0."""
    async with _lock:
        try:
            if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                    or not math.isfinite(timeout) or timeout <= 0):
                return "Error executing command: timeout must be finite and positive", False
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
                stdout_text, stdout_cut = _read_captured_output(stdout, encoding)
                stderr_text, stderr_cut = _read_captured_output(stderr, encoding)
                output = stdout_text + stderr_text
                truncated = stdout_cut or stderr_cut
            output = output.replace("\r\n", "\n").replace("\r", "\n").strip()
            if not output:
                if process.returncode == 0:
                    return "[command completed successfully with no output]", True
                return f"[exit code {process.returncode}]", False
            if truncated or len(output) > MAX_OUTPUT_CHARS:
                output = output[:MAX_OUTPUT_CHARS] + "\n... [output truncated at 30000 characters]"
            return output, process.returncode == 0
        except asyncio.TimeoutError:
            return f"[command timed out after {timeout}s]", False
        except Exception as e:
            return f"Error executing command: {e}", False
