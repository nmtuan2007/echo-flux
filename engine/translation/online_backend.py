import json
import urllib.parse
import urllib.request
from typing import List, Tuple

from engine.translation.base import TranslationBackend, TranslationResult
from engine.core.logging import get_logger
from engine.core.exceptions import TranslationError

logger = get_logger("translation.online")


class OnlineBackend(TranslationBackend):
    """
    Lightweight online translation backend using public APIs.
    Does not require local GPU/CPU heavy models.
    """

    def __init__(self):
        self._loaded = False
        self._base_url = "https://translate.googleapis.com/translate_a/single"
        # Common user agent to avoid basic blocking
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

    def load_model(self, config: dict) -> None:
        # No model to load, just marking as ready
        self._loaded = True
        logger.info("Online translation backend initialized (Google Translate API)")

    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        if not text.strip():
            return TranslationResult(text, "", source_lang, target_lang)

        try:
            # Map 'auto' to empty string for API
            src = "" if source_lang == "auto" else source_lang
            
            params = {
                "client": "gtx",
                "sl": src,
                "tl": target_lang,
                "dt": "t",
                "q": text
            }
            
            url = f"{self._base_url}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers=self._headers)
            
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    raise TranslationError(f"HTTP error {response.status}")
                
                data = json.loads(response.read().decode("utf-8"))
                
                # Parse structure: [[[translated_text, source_text, ...], ...], ...]
                translated_parts = []
                if data and isinstance(data, list) and len(data) > 0:
                    for part in data[0]:
                        if part and isinstance(part, list) and len(part) > 0:
                            translated_parts.append(part[0])
                
                translated_text = "".join(translated_parts)
                
                return TranslationResult(
                    source_text=text,
                    translated_text=translated_text,
                    source_lang=source_lang,
                    target_lang=target_lang
                )

        except Exception as e:
            logger.error("Online translation failed: %s", e)
            # Fallback: return original text to avoid crashing pipeline
            return TranslationResult(text, "", source_lang, target_lang)

    def unload_model(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def supported_pairs(self) -> List[Tuple[str, str]]:
        # Online supports almost everything, returning a wildcard
        return [("any", "any")]
