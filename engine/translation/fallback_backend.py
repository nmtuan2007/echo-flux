import time
from typing import List, Tuple, Optional

from engine.translation.base import TranslationBackend, TranslationResult
from engine.translation.online_backend import OnlineBackend
from engine.translation.marian_backend import MarianBackend
from engine.core.logging import get_logger
from engine.core.exceptions import TranslationError, ModelLoadError

logger = get_logger("translation.fallback")

# How often to retry online backend when fallen back to marian (seconds)
ONLINE_RETRY_INTERVAL = 60.0


class FallbackTranslationBackend(TranslationBackend):
    """
    Translation backend with automatic fallback.
    Primary: OnlineBackend (Google Translate)
    Fallback: MarianBackend (local model)
    """

    def __init__(self):
        self._online: Optional[OnlineBackend] = None
        self._marian: Optional[MarianBackend] = None
        self._config: dict = {}

        self._online_enabled = True
        self._marian_enabled = True
        self._online_loaded = False
        self._marian_loaded = False

        # Which backend is currently active
        self._active: Optional[str] = None  # "online" | "marian"
        self._fallen_back = False
        self._last_online_retry = 0.0

    def load_model(self, config: dict) -> None:
        self._config = config
        preferred = config.get("translation.backend", config.get("backend", "online"))

        # Determine which backends to initialize
        if preferred == "online":
            self._online_enabled = True
            self._marian_enabled = True  # Always prepare marian as fallback
        elif preferred == "marian":
            self._online_enabled = False
            self._marian_enabled = True
        else:
            self._online_enabled = True
            self._marian_enabled = True

        # Load online backend
        if self._online_enabled:
            try:
                self._online = OnlineBackend()
                self._online.load_model(config)
                self._online_loaded = True
                logger.info("Online translation backend ready")
            except Exception as e:
                logger.warning("Failed to initialize online backend: %s", e)
                self._online_loaded = False

        # Load marian backend
        if self._marian_enabled:
            try:
                self._marian = MarianBackend()
                self._marian.load_model(config)
                self._marian_loaded = True
                logger.info("Marian translation backend ready (fallback)")
            except Exception as e:
                logger.warning("Failed to initialize Marian backend: %s", e)
                self._marian_loaded = False

        # Set initial active backend
        if preferred == "marian" and self._marian_loaded:
            self._active = "marian"
        elif self._online_loaded:
            self._active = "online"
        elif self._marian_loaded:
            self._active = "marian"
            self._fallen_back = True
            logger.warning("Online backend unavailable, starting with Marian")
        else:
            raise ModelLoadError("No translation backend could be loaded")

        logger.info("Active translation backend: %s", self._active)

    def translate_raw(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        if not text.strip():
            return TranslationResult(text, "", source_lang, target_lang)

        # If fallen back to marian, periodically try online again
        if self._fallen_back and self._online_loaded:
            self._maybe_retry_online(text, source_lang, target_lang)

        # Try active backend
        if self._active == "online":
            return self._try_online(text, source_lang, target_lang)
        else:
            return self._try_marian(text, source_lang, target_lang)

    def _try_online(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Try online backend, fall back to marian on failure."""
        try:
            if not self._online or not self._online.is_available:
                raise TranslationError("Online backend not available")

            result = self._online.translate(text, source_lang, target_lang)

            # Validate result is not empty
            if not result.translated_text.strip():
                raise TranslationError("Empty translation result")

            return result

        except (TranslationError, Exception) as e:
            logger.warning("Online translation failed: %s", e)

            # Check if we should fall back
            if self._online and self._online.consecutive_failures >= 3:
                self._switch_to_marian()

            # Try marian as fallback for this request
            if self._marian_loaded:
                return self._try_marian(text, source_lang, target_lang)

            # Nothing works — return empty
            logger.error("All translation backends failed")
            return TranslationResult(text, "", source_lang, target_lang)

    def _try_marian(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Try marian backend."""
        if not self._marian or not self._marian.is_loaded:
            return TranslationResult(text, "", source_lang, target_lang)

        try:
            return self._marian.translate(text, source_lang, target_lang)
        except Exception as e:
            logger.error("Marian translation failed: %s", e)
            return TranslationResult(text, "", source_lang, target_lang)

    def _switch_to_marian(self):
        """Switch active backend to marian."""
        if not self._marian_loaded:
            logger.error("Cannot fall back to Marian — not loaded")
            return

        if self._active == "marian":
            return

        self._active = "marian"
        self._fallen_back = True
        self._last_online_retry = time.time()
        logger.warning(
            "Switched to Marian backend (online failed %d times). "
            "Will retry online in %ds.",
            self._online.consecutive_failures if self._online else 0,
            int(ONLINE_RETRY_INTERVAL),
        )

    def _switch_to_online(self):
        """Switch back to online backend."""
        self._active = "online"
        self._fallen_back = False
        if self._online:
            self._online.reset_failures()
        logger.info("Switched back to online translation backend")

    def _maybe_retry_online(self, text: str, source_lang: str, target_lang: str):
        """Periodically retry online backend when fallen back to marian."""
        now = time.time()
        if now - self._last_online_retry < ONLINE_RETRY_INTERVAL:
            return

        self._last_online_retry = now
        logger.info("Retrying online translation backend...")

        if self._online:
            self._online.reset_failures()

        try:
            if not self._online or not self._online.is_loaded:
                return

            # Test with a short probe
            probe_text = text[:50] if len(text) > 50 else text
            result = self._online.translate(probe_text, source_lang, target_lang)

            if result.translated_text.strip():
                self._switch_to_online()
            else:
                logger.info("Online retry returned empty result, staying on Marian")
        except Exception as e:
            logger.info("Online retry failed: %s. Staying on Marian.", e)

    def unload_model(self) -> None:
        if self._online:
            self._online.unload_model()
            self._online = None
            self._online_loaded = False

        if self._marian:
            self._marian.unload_model()
            self._marian = None
            self._marian_loaded = False

        self._active = None
        self._fallen_back = False
        logger.info("Fallback translation backend unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._online_loaded or self._marian_loaded

    @property
    def active_backend(self) -> Optional[str]:
        """Return which backend is currently active: 'online' or 'marian'."""
        return self._active

    @property
    def is_fallen_back(self) -> bool:
        return self._fallen_back

    @property
    def supported_pairs(self) -> List[Tuple[str, str]]:
        if self._active == "online" and self._online:
            return self._online.supported_pairs
        if self._marian:
            return self._marian.supported_pairs
        return []
