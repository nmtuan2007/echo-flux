import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from engine.core.logging import get_logger

logger = get_logger("translation.base")


@dataclass
class TranslationResult:
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence: Optional[float] = None


class TranslationBackend(ABC):

    @abstractmethod
    def load_model(self, config: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def translate_raw(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Raw translation without post-processing. Implemented by each backend."""
        raise NotImplementedError

    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Translate with automatic post-processing."""
        if not text.strip():
            return TranslationResult(text, "", source_lang, target_lang)

        result = self.translate_raw(text, source_lang, target_lang)

        if result.translated_text:
            result.translated_text = self.post_process(result.translated_text)

        return result

    @abstractmethod
    def unload_model(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_pairs(self) -> list:
        raise NotImplementedError

    @staticmethod
    def split_sentences(text: str, max_length: int = 200) -> List[str]:
        """Split text into sentence-sized chunks for better translation quality."""
        if len(text) <= max_length:
            return [text]

        # Split on sentence boundaries
        parts = re.split(r'(?<=[.!?;])\s+', text)

        sentences = []
        current = ""

        for part in parts:
            if not part.strip():
                continue

            if current and len(current) + len(part) + 1 > max_length:
                sentences.append(current.strip())
                current = part
            else:
                current = f"{current} {part}".strip() if current else part

        if current.strip():
            sentences.append(current.strip())

        # If any sentence is still too long, split on commas
        final = []
        for s in sentences:
            if len(s) <= max_length:
                final.append(s)
            else:
                comma_parts = re.split(r',\s*', s)
                chunk = ""
                for cp in comma_parts:
                    if chunk and len(chunk) + len(cp) + 2 > max_length:
                        final.append(chunk.strip())
                        chunk = cp
                    else:
                        chunk = f"{chunk}, {cp}".strip(", ") if chunk else cp
                if chunk.strip():
                    final.append(chunk.strip())

        return final if final else [text]

    @staticmethod
    def post_process(text: str) -> str:
        """Remove repetitions and clean up translated text."""
        if not text or len(text) < 5:
            return text

        words = text.split()
        if len(words) < 3:
            return text

        # Pass 1: Single word consecutive repeats (keep max 1)
        deduplicated = [words[0]]
        for i in range(1, len(words)):
            if words[i].lower() != words[i - 1].lower():
                deduplicated.append(words[i])

        words = deduplicated

        if len(words) < 4:
            return " ".join(words)

        # Pass 2: N-gram consecutive repeats (n = 2..8)
        result = list(words)
        found = True
        while found:
            found = False
            for n in range(2, min(9, len(result) // 2 + 1)):
                i = 0
                new_result = []
                while i < len(result):
                    if i + n * 2 <= len(result):
                        pattern = [w.lower() for w in result[i:i + n]]
                        next_block = [w.lower() for w in result[i + n:i + n * 2]]

                        if pattern == next_block:
                            new_result.extend(result[i:i + n])
                            pos = i + n
                            while pos + n <= len(result):
                                candidate = [w.lower() for w in result[pos:pos + n]]
                                if candidate == pattern:
                                    pos += n
                                else:
                                    break
                            new_result.extend(result[pos:])
                            found = True
                            result = new_result
                            break
                        else:
                            new_result.append(result[i])
                            i += 1
                    else:
                        new_result.append(result[i])
                        i += 1
                else:
                    result = new_result
                if found:
                    break

        if len(result) < 2:
            return " ".join(result)

        # Pass 3: Dominant word check
        word_counts = Counter(w.lower() for w in result)
        total = len(result)
        for word, count in word_counts.most_common(1):
            if count > total * 0.4 and total > 5:
                trimmed = []
                seen = 0
                for w in result:
                    if w.lower() == word:
                        seen += 1
                    if seen > 3:
                        break
                    trimmed.append(w)
                logger.debug("Dominant word '%s' (%d/%d). Trimmed.", word, count, total)
                result = trimmed
                break

        return " ".join(result)
