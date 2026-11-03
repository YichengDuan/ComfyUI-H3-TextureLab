"""Prompt-finally GC without evicting model or node-output caches."""
from contextvars import ContextVar
from functools import wraps
import gc
import logging
import weakref

_readers = ContextVar('h3_texture_run_readers', default=None)


def track_reader(reader):
    """Register on use, including readers reused from a cached node output."""
    readers = _readers.get()
    if readers is not None:
        readers.add(reader)


def cleanup(readers):
    for reader in list(readers):
        try:
            with reader.lock:
                reader.close()
        except Exception:
            logging.warning('[H3 Texture Lab] Reader cleanup failed', exc_info=True)
    readers.clear()
    try:
        gc.collect()
    except Exception:
        logging.warning('[H3 Texture Lab] Python GC failed', exc_info=True)
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        logging.warning('[H3 Texture Lab] Accelerator cache cleanup failed', exc_info=True)
    logging.info('[H3 Texture Lab] End-of-run GC complete (model cache retained)')


def install(executor_class=None):
    if executor_class is None:
        from execution import PromptExecutor
        executor_class = PromptExecutor
    original = executor_class.execute_async
    if getattr(original, '_h3_texture_run_cleanup', False):
        return

    @wraps(original)
    async def execute_with_cleanup(self, *args, **kwargs):
        readers = weakref.WeakSet()
        token = _readers.set(readers)
        try:
            return await original(self, *args, **kwargs)
        finally:
            try:
                cleanup(readers)
            finally:
                _readers.reset(token)

    execute_with_cleanup._h3_texture_run_cleanup = True
    executor_class.execute_async = execute_with_cleanup
