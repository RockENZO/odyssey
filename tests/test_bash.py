"""Exercise real shell grandchildren, without modifying anything outside tmpdirs."""
import asyncio
import os
import shlex
import sys
import tempfile
import time
import signal
import importlib
from unittest.mock import patch
import unittest
from pathlib import Path

from odyssey.tools.bash import bash
shell_module = importlib.import_module("odyssey.tools.bash")


def python_command(source):
    return f'{shlex.quote(sys.executable)} -c {shlex.quote(source)}'


@unittest.skipUnless(os.name == 'posix', 'POSIX process-group lifecycle')
class BashLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def descendant_command(self, root):
        child = (f'from pathlib import Path; import time; '
                 f'Path({str(root / "started")!r}).touch(); time.sleep(0.8); '
                 f'Path({str(root / "escaped")!r}).touch()')
        parent = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(5)'
        return python_command(parent)

    async def test_timeout_stops_descendants_before_returning(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = await bash(self.descendant_command(root), timeout=0.4)
            self.assertIn('timed out', result)
            self.assertTrue((root / 'started').exists(), 'Child must start to exercise the timeout race')
            await asyncio.sleep(0.9)
            self.assertFalse((root / 'escaped').exists(), 'Timed-out child must not continue writing')
            self.assertEqual(await bash('printf recovered'), 'recovered')

    async def test_cancellation_stops_descendants_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = asyncio.create_task(bash(self.descendant_command(root)))
            try:
                for _ in range(200):
                    if (root / 'started').exists():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue((root / 'started').exists())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.sleep(0.9)
                self.assertFalse((root / 'escaped').exists(), 'Cancelled command must not outlive its tool call')
                self.assertEqual(await bash('printf recovered'), 'recovered')
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_escaped_session_cannot_hold_timeout_open_via_output_handles(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            child = 'import time; time.sleep(3)'
            parent = (f'import subprocess,sys,time; from pathlib import Path; '
                      f'p=subprocess.Popen([sys.executable,"-c",{child!r}],start_new_session=True); '
                      f'Path({str(root / "pid")!r}).write_text(str(p.pid)); time.sleep(5)')
            start = time.monotonic()
            try:
                result = await bash(python_command(parent), timeout=0.4)
                self.assertIn('timed out', result)
                self.assertLess(time.monotonic() - start, 2, 'Detached stdout owner must not delay cleanup')
                self.assertEqual(await bash('printf recovered'), 'recovered')
            finally:
                if (root / 'pid').exists():
                    try:
                        os.kill(int((root / 'pid').read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    async def test_repeated_cancellation_waits_for_reap_before_unlocking(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original_stop = shell_module._stop
        async def delayed_stop(process):
            entered.set()
            await release.wait()
            await original_stop(process)
        with patch.object(shell_module, '_stop', side_effect=delayed_stop):
            task = asyncio.create_task(bash(python_command('import time; time.sleep(5)')))
            try:
                await asyncio.sleep(0.1)
                task.cancel()
                await asyncio.wait_for(entered.wait(), 2)
                task.cancel()
                await asyncio.sleep(0.02)
                self.assertFalse(task.done())
                self.assertTrue(shell_module._lock.locked())
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        self.assertEqual(await bash('printf recovered'), 'recovered')

    async def test_cancellation_queued_at_spawn_is_cleaned_up(self):
        original_spawn = shell_module.subprocess.Popen
        captured = []
        task = None
        def spawn_then_cancel(*args, **kwargs):
            process = original_spawn(*args, **kwargs)
            captured.append(process)
            task.cancel()
            return process
        with patch.object(shell_module.subprocess, 'Popen', side_effect=spawn_then_cancel):
            task = asyncio.create_task(bash(python_command('import time; time.sleep(5)')))
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIsNotNone(captured[0].poll(), 'Spawned child must be reaped before cancellation completes')
        self.assertEqual(await bash('printf recovered'), 'recovered')

    async def test_locale_output_is_not_assumed_to_be_utf8(self):
        with patch.object(shell_module.locale, 'getpreferredencoding', return_value='cp1252'):
            result = await bash(python_command('import sys; sys.stdout.buffer.write(bytes([99,97,102,233]))'))
        self.assertEqual(result, 'caf\u00e9')

    async def test_invalid_timeouts_do_not_launch_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'unexpected'
            command = python_command(f'from pathlib import Path; Path({str(target)!r}).touch()')
            for timeout in [None, '1', float('nan'), float('inf'), 0, -1, True]:
                with self.subTest(timeout=timeout):
                    self.assertIn('timeout must be finite and positive', await bash(command, timeout))
            self.assertFalse(target.exists())

    async def test_output_and_exit_reporting(self):
        self.assertEqual(await bash('printf stdout; printf stderr >&2'), 'stdoutstderr')
        self.assertEqual(await bash(python_command('print("a\\r\\nb\\r\\nc")')), 'a\nb\nc')
        self.assertEqual(await bash('exit 7'), '[exit code 7]')
        self.assertEqual(await bash('true'), '[command completed successfully with no output]')
        result = await bash(python_command('print("x"*31000)'))
        self.assertTrue(result.endswith('[output truncated at 30000 characters]'))
