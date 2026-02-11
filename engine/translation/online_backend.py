import json
import time
import urllib.parse
import urllib.request
import urllib.error
from collections import OrderedDict
from threading import Lock
from typing import List, Tuple

from engine.translation.base import TranslationBackend, TranslationResult
from engine.core.logging import get_logger
from engine.core.exceptions import TranslationError

logger = get_logger("translation.online")

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 30
REQUEST_TIMEOUT_SECONDS = 5

# Cache
MAX_CACHE_SIZE = 500

# Backoff
INITIAL_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
BACKOFF_MULTIPLIER = 2.0

# Consecutive failures before considered "down"
MAX_CONSECUTIVE_FAILURES = 3


class OnlineBackend(TranslationBackend):

    def __init__(self):
        self._loaded = False
        self._base_url = "https://translate.googleapis.com/translate_a/single"
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }

        # Rate limiting state
        self._request_timestamps: list = []
        self._rate_lock = Lock()

        # Backoff state
        self._backoff_until = 0.0
        self._current_backoff = INITIAL_BACKOFF_SECONDS

        # Cache: (source_lang, target_lang, text) -> translated_text
        self._cache: OrderedDict = OrderedDict()
        self._cache_lock = Lock()

        # Failure tracking
        self._consecutive_failures = 0

    def load_model(self, config: dict) -> None:
        self._loaded = True
        self._consecutive_failures = 0
        self._backoff_until = 0.0
        self._current_backoff = INITIAL_BACKOFF_SECONDS
        logger.info("Online translation backend initialized")

    def translate_raw(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        if not text.strip():
            return TranslationResult(text, "", source_lang, target_lang)

        # Check backoff
        now = time.time()
        if now < self._backoff_until:
            wait = self._backoff_until - now
            logger.debug("Backoff active, %.1fs remaining", wait)
            raise TranslationError(f"Online backend in backoff ({wait:.0f}s remaining)")

        # Split into sentences for long text
        sentences = self.split_sentences(text, max_length=300)
        translated_parts = []

        for sentence in sentences:
            translated = self._translate_single(sentence, source_lang, target_lang)
            translated_parts.append(translated)

        translated_text = " ".join(translated_parts)

        return TranslationResult(
            source_text=text,
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def _translate_single(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single sentence with caching and rate limiting."""
        if not text.strip():
            return ""

        # Check cache
        cache_key = (source_lang, target_lang, text.strip())
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Rate limit
        self._wait_for_rate_limit()

        # Make request
        try:
            result = self._do_request(text, source_lang, target_lang)
            self._on_success()
            self._cache_put(cache_key, result)
            return result

        except TranslationError:
            raise
        except Exception as e:
            self._on_failure(e)
            raise TranslationError(f"Online translation failed: {e}") from e

    def _do_request(self, text: str, source_lang: str, target_lang: str) -> str:
        """Execute HTTP request to Google Translate."""
        src = "" if source_lang == "auto" else source_lang

        params = {
            "client": "gtx",
            "sl": src,
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }

        url = f"{self._base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=self._headers)

        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status = response.status
                if status != 200:
                    raise TranslationError(f"HTTP {status}")

                data = json.loads(response.read().decode("utf-8"))

                translated_parts = []
                if data and isinstance(data, list) and len(data) > 0:
                    for part in data[0]:
                        if part and isinstance(part, list) and len(part) > 0 and part[0]:
                            translated_parts.append(part[0])

                return "".join(translated_parts)

        except urllib.error.HTTPError as e:
            if e.code in (429, 403):
                self._activate_backoff()
                raise TranslationError(f"Rate limited (HTTP {e.code})")
            elif e.code >= 500:
                self._activate_backoff()
                raise TranslationError(f"Server error (HTTP {e.code})")
            raise TranslationError(f"HTTP error {e.code}") from e

        except urllib.error.URLError as e:
            raise TranslationError(f"Network error: {e.reason}") from e

        except TimeoutError:
            raise TranslationError("Request timed out")

    def _wait_for_rate_limit(self):
        """Block until we're within rate limits."""
        with self._rate_lock:
            now = time.time()
            cutoff = now - 60.0

            # Remove timestamps older than 1 minute
            self._request_timestamps = [
                t for t in self._request_timestamps if t > cutoff
            ]

            if len(self._request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
                oldest = self._request_timestamps[0]
                wait = 60.0 - (now - oldest) + 0.1
                if wait > 0:
                    logger.debug("Rate limit reached, waiting %.1fs", wait)
                    time.sleep(wait)

            self._request_timestamps.append(time.time())

    def _activate_backoff(self):
        """Activate exponential backoff."""
        self._backoff_until = time.time() + self._current_backoff
        logger.warning(
            "Online backend: activating backoff for %.1fs",
            self._current_backoff,
        )
        self._current_backoff = min(
            self._current_backoff * BACKOFF_MULTIPLIER,
            MAX_BACKOFF_SECONDS,
        )

    def _on_success(self):
        """Reset failure tracking on successful request."""
        self._consecutive_failures = 0
        self._current_backoff = INITIAL_BACKOFF_SECONDS

    def _on_failure(self, error: Exception):
        """Track consecutive failures."""
        self._consecutive_failures += 1
        logger.warning(
            "Online translation failure %d/%d: %s",
            self._consecutive_failures, MAX_CONSECUTIVE_FAILURES, error,
        )

    def _cache_get(self, key: tuple) -> str | None:
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def _cache_put(self, key: tuple, value: str):
        with self._cache_lock:
            self._cache[key] = value
            if len(self._cache) > MAX_CACHE_SIZE:
                self._cache.popitem(last=False)

    @property
    def is_available(self) -> bool:
        """Whether the backend is currently usable (not in backoff, not too many failures)."""
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            return False
        if time.time() < self._backoff_until:
            return False
        return True

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def reset_failures(self):
        """Reset failure counter — called by fallback backend when retrying."""
        self._consecutive_failures = 0
        self._current_backoff = INITIAL_BACKOFF_SECONDS
        self._backoff_until = 0.0

    def unload_model(self) -> None:
        self._loaded = False
        with self._cache_lock:
            self._cache.clear()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def supported_pairs(self) -> List[Tuple[str, str]]:
        return [("any", "any")]
