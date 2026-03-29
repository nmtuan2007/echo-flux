import logging
import threading
from typing import Callable, Optional
from tqdm.auto import tqdm as std_tqdm

logger = logging.getLogger("echoflux.core.progress")

# Global callback to receive progress updates: callback(model_name: str, percent: int)
_progress_callback: Optional[Callable[[str, int], None]] = None
_lock = threading.Lock()

def set_progress_callback(callback: Optional[Callable[[str, int], None]]):
    global _progress_callback
    with _lock:
        _progress_callback = callback


class HijackedTqdm(std_tqdm):
    """
    A custom tqdm class that intercepts progress updates and forwards them to a global callback.
    """
    def __init__(self, *args, **kwargs):
        # Always enable the progress bar so we get updates
        if "disable" in kwargs:
            kwargs["disable"] = False
            
        # huggingface_hub passes 'name' which isn't accepted by std_tqdm
        kwargs.pop("name", None)
            
        super().__init__(*args, **kwargs)
        self._last_percent = -1
        # Extract the model name from the description (desc) or use a generic one
        self._model_name = kwargs.get("desc", "AI Model")

    def update(self, n=1):
        super().update(n)
        self._report_progress()
        
    def close(self):
        try:
            super().close()
        except AttributeError:
            pass
            
        # Ensure we always send 100% when done
        if hasattr(self, "total") and hasattr(self, "n"):
            if self.total and self.n >= self.total:
                self._emit(100)

    def _report_progress(self):
        if not hasattr(self, "total") or not hasattr(self, "n") or not self.total:
            return
            
        percent = int((self.n / self.total) * 100)
        # Update only when percent changes by a reasonable step, e.g., 2% to avoid flooding
        if percent - self._last_percent >= 2 or percent == 100:
            self._last_percent = percent
            self._emit(percent)

    def _emit(self, percent: int):
        global _progress_callback
        cb = None
        with _lock:
            cb = _progress_callback
            
        if cb:
            try:
                cb(self._model_name, percent)
            except Exception as e:
                logger.error("Progress callback failed: %s", e)


def install_progress_hijack():
    """
    Monkey-patch common tqdm classes used by huggingface_hub, transformers, and faster_whisper.
    Call this early in the initialization.
    """
    logger.info("Installing tqdm progress hijack...")
    
    # 1. Patch tqdm.auto.tqdm (used by many HF libraries)
    import tqdm
    import tqdm.auto
    tqdm.tqdm = HijackedTqdm
    tqdm.auto.tqdm = HijackedTqdm

    # 2. Patch huggingface_hub utils if it uses a local alias
    try:
        import huggingface_hub.utils
        if hasattr(huggingface_hub.utils, "_tqdm"):
            huggingface_hub.utils._tqdm.tqdm = HijackedTqdm
        elif hasattr(huggingface_hub.utils, "tqdm"):
            huggingface_hub.utils.tqdm = HijackedTqdm
    except Exception:
        pass

    # 3. Patch faster_whisper's explicit disabled_tqdm
    try:
        import faster_whisper.utils
        faster_whisper.utils.disabled_tqdm = HijackedTqdm
    except Exception:
        pass
        
    # 4. Patch transformers utils natively if available
    try:
        import transformers.utils.logging
        import transformers.utils.hub
        # Transformers might alias tqdm
        if hasattr(transformers.utils.logging, "tqdm"):
            transformers.utils.logging.tqdm = HijackedTqdm
    except Exception:
        pass
    
    logger.debug("Tqdm hijacked for progress reporting.")

