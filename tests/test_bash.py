"""Exercise real shell grandchildren, without modifying anything outside tmpdirs."""
import asyncio
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

from odyssey.tools.bash import bash


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
