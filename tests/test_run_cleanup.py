import asyncio
import importlib.util
from pathlib import Path
import threading
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('run_cleanup_test', Path(__file__).resolve().parents[1] / 'run_cleanup.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RunCleanupTests(unittest.TestCase):
    def test_success_error_interrupt_and_cancel_preserve_models(self):
        for failure in (None, RuntimeError('node failed'), KeyboardInterrupt(), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                reader = types.SimpleNamespace()
                class Reader:
                    lock = threading.RLock()
                    closed = False
                    def close(self): self.closed = True
                reader = Reader()
                model = object()
                class Executor:
                    cached_model = model
                    async def execute_async(self):
                        module.track_reader(reader)
                        if failure is not None: raise failure
                        return 42
                module.install(Executor)
                wrapper = Executor.execute_async
                module.install(Executor)
                self.assertIs(wrapper, Executor.execute_async)
                mm = types.ModuleType('comfy.model_management')
                mm.soft_empty_cache = unittest.mock.Mock()
                comfy = types.ModuleType('comfy'); comfy.model_management = mm
                with patch.dict('sys.modules', {'comfy': comfy, 'comfy.model_management': mm}), patch.object(module.gc, 'collect') as collect:
                    if failure is None:
                        self.assertEqual(asyncio.run(Executor().execute_async()), 42)
                    else:
                        with self.assertRaises(type(failure)):
                            asyncio.run(Executor().execute_async())
                    self.assertTrue(reader.closed)
                    collect.assert_called_once()
                    mm.soft_empty_cache.assert_called_once()
                    self.assertIs(Executor.cached_model, model)
                    self.assertIsNone(module._readers.get())

    def test_cleanup_failure_does_not_replace_original_error(self):
        class Reader:
            lock = threading.RLock()
            def close(self): raise ValueError('close failed')
        reader = Reader()
        class Executor:
            async def execute_async(self):
                module.track_reader(reader)
                raise RuntimeError('original')
        module.install(Executor)
        with patch.object(module.gc, 'collect', side_effect=ValueError('gc failed')), self.assertLogs(level='WARNING'):
            with self.assertRaisesRegex(RuntimeError, 'original'):
                asyncio.run(Executor().execute_async())

    def test_concurrent_runs_keep_separate_readers(self):
        class Reader:
            lock = threading.RLock()
            closed = False
            def close(self): self.closed = True
        a, b = Reader(), Reader()
        async def scenario():
            ready, released = asyncio.Event(), asyncio.Event()
            class Executor:
                async def execute_async(self, reader, wait):
                    module.track_reader(reader)
                    if wait:
                        ready.set()
                        await released.wait()
                    else:
                        await ready.wait()
            module.install(Executor)
            first = asyncio.create_task(Executor().execute_async(a, True))
            await Executor().execute_async(b, False)
            self.assertTrue(b.closed)
            self.assertFalse(a.closed)
            released.set()
            await first
            self.assertTrue(a.closed)
        with patch.object(module, 'cleanup', side_effect=lambda readers: [r.close() for r in readers]):
            asyncio.run(scenario())

if __name__ == '__main__': unittest.main()
