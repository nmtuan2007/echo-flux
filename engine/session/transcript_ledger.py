"""Provider-neutral transcript revision control for one session.

The ledger decides what a normalized ``STTTranscriptResult`` means for the visible
transcript: show a partial, commit a final, or drop it as stale, late, or duplicate. It
holds no provider, socket, or UI knowledge.

Streams: every stream id (``mic`` / ``system``) has its own ``StreamTranscriptLedger``.
Streams never share items, finals, boundaries, or warnings, so two streams hearing the same
words at the same time never reconcile against each other.

Epochs: an epoch is one provider connection on one stream. Provider timelines restart per
connection, so each epoch carries an explicit ``offset_ms`` and every result interval is
mapped onto the stream's session timeline as ``provider_ms + offset_ms``. Only the newest
epoch of a stream is live; results from an older epoch are rejected, which is what stops a
dead connection's late events from reappearing after a reconnect or fallback.

Revision rules, per item (``epoch`` + ``provider_event_id``):

- a revision lower than or equal to the last accepted one is stale;
- nothing is accepted for an item after its final, so a partial never overwrites a final;
- an empty final retracts the item and commits nothing.

Duplicate reconciliation for a result in epoch *E* only considers finals committed by
epochs older than *E* (a single provider connection never duplicates itself):

1. **Interval first.** Prior finals whose session interval overlaps the result are
   candidates. None means the result is new; text is never compared.
2. If the union of candidate intervals covers at least ``covered_ratio`` of the result, it
   is a duplicate by interval alone.
3. Otherwise text is compared **only** when the result touches *E*'s overlap window (the
   replay region before the last final boundary). First, if the candidates' joined text
   ends with the result's first ``min_trim_tokens``+ tokens, that prefix is trimmed and the
   rest is committed (nothing left means duplicate). Otherwise the normalized result is a
   duplicate if its similarity ratio with the joined text reaches ``text_similarity``, or
   if it is contained in that text at ``containment_ratio``.
4. Any overlap that neither rule resolves is committed with an explicit
   ``unresolved_overlap`` continuity warning rather than being silently accepted.

Thread ownership: a stream ledger is mutated by the thread that consumes that stream's STT
results (today the ``Process-{stream}`` thread). Each stream ledger has its own lock, and
``TranscriptLedger`` guards its stream map, so readers on other threads see consistent
state. No asyncio loop is involved and no call blocks on I/O.
"""

import difflib
import threading
import unicodedata
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

from engine.core.logging import get_logger
from engine.providers.types import STTTranscriptResult

logger = get_logger("session.transcript_ledger")


class LedgerAction(str, Enum):
    """Outcome of ingesting one transcript result."""

    PARTIAL = "partial"
    FINAL = "final"
    RETRACTED = "retracted"
    SUPPRESSED_PARTIAL = "suppressed_partial"
    DUPLICATE_FINAL = "duplicate_final"
    STALE = "stale"
    AFTER_FINAL = "after_final"
    RETIRED_EPOCH = "retired_epoch"
    STREAM_MISMATCH = "stream_mismatch"


# Actions that change what the transcript shows.
_VISIBLE_ACTIONS = frozenset({LedgerAction.PARTIAL, LedgerAction.FINAL, LedgerAction.RETRACTED})


class ContinuityReason(str, Enum):
    """Why continuity of a stream's transcript could not be proven."""

    AUDIO_GAP = "audio_gap"
    UNRESOLVED_OVERLAP = "unresolved_overlap"


@dataclass(frozen=True)
class LedgerThresholds:
    """Reconciliation thresholds and memory bounds for one stream ledger."""

    interval_tolerance_ms: float = 120.0
    covered_ratio: float = 0.8
    text_similarity: float = 0.85
    containment_ratio: float = 0.9
    min_containment_chars: int = 4
    min_trim_tokens: int = 2
    max_items: int = 512
    max_committed_finals: int = 256
    max_warnings: int = 256


@dataclass(frozen=True)
class ContinuityWarning:
    """An interval of the session timeline whose transcript may be missing or repeated."""

    stream_id: str
    epoch_id: int
    start_ms: float
    end_ms: float
    reason: ContinuityReason


@dataclass(frozen=True)
class EpochHandle:
    """One provider connection on one stream, with its timeline mapping."""

    stream_id: str
    epoch_id: int
    provider_id: str
    offset_ms: float = 0.0
    overlap_start_ms: Optional[float] = None
    overlap_end_ms: Optional[float] = None

    @property
    def has_overlap_window(self) -> bool:
        return self.overlap_start_ms is not None and self.overlap_end_ms is not None


@dataclass(frozen=True)
class EpochStart:
    """Result of opening an epoch: its handle and the prior epoch's abandoned partials."""

    handle: EpochHandle
    abandoned_entry_keys: Tuple[str, ...] = ()
    gap_warning: Optional[ContinuityWarning] = None


@dataclass(frozen=True)
class LedgerDecision:
    """What the transcript should do with one result. ``text`` is empty for rejections."""

    action: LedgerAction
    stream_id: str
    epoch_id: int
    entry_key: str
    revision: int
    text: str = ""
    start_ms: float = 0.0
    end_ms: float = 0.0
    duplicate_of: Optional[str] = None
    trimmed_prefix_tokens: int = 0
    warning: Optional[ContinuityWarning] = None

    @property
    def is_visible(self) -> bool:
        return self.action in _VISIBLE_ACTIONS


def normalize_text(text: str) -> str:
    """Casefold, NFKC-normalize, drop punctuation/symbols, and collapse whitespace."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    cleaned = "".join(
        " " if unicodedata.category(char)[0] in ("P", "S") else char for char in folded
    )
    return " ".join(cleaned.split())


@dataclass
class _ItemState:
    revision: int
    is_final: bool = False


@dataclass(frozen=True)
class _CommittedFinal:
    entry_key: str
    epoch_id: int
    start_ms: float
    end_ms: float
    normalized: str


class _Verdict(str, Enum):
    NEW = "new"
    DUPLICATE = "duplicate"
    TRIMMED = "trimmed"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class _Reconciliation:
    verdict: _Verdict
    text: str
    start_ms: float
    duplicate_of: Optional[str] = None
    trimmed_tokens: int = 0


class StreamTranscriptLedger:
    """Revision and duplicate control for a single audio stream."""

    def __init__(self, stream_id: str, thresholds: Optional[LedgerThresholds] = None) -> None:
        self.stream_id = stream_id
        self.thresholds = thresholds or LedgerThresholds()
        self._lock = threading.Lock()
        self._epoch: Optional[EpochHandle] = None
        self._next_epoch_id = 1
        self._items: OrderedDict[str, _ItemState] = OrderedDict()
        self._finals: Deque[_CommittedFinal] = deque(maxlen=self.thresholds.max_committed_finals)
        self._warnings: Deque[ContinuityWarning] = deque(maxlen=self.thresholds.max_warnings)
        self._warning_count = 0
        self._duplicate_count = 0
        self._final_boundary_ms = 0.0

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def active_epoch(self) -> Optional[EpochHandle]:
        with self._lock:
            return self._epoch

    @property
    def last_final_boundary_ms(self) -> float:
        """End of the latest committed final on the session timeline (0 before any)."""
        with self._lock:
            return self._final_boundary_ms

    @property
    def warning_count(self) -> int:
        """Total warnings raised, including any that aged out of ``warnings()``."""
        with self._lock:
            return self._warning_count

    @property
    def duplicate_count(self) -> int:
        with self._lock:
            return self._duplicate_count

    @property
    def retained_item_count(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def retained_final_count(self) -> int:
        with self._lock:
            return len(self._finals)

    def warnings(self) -> List[ContinuityWarning]:
        with self._lock:
            return list(self._warnings)

    # ── Epochs ───────────────────────────────────────────────────────────────

    def begin_epoch(
        self,
        provider_id: str,
        offset_ms: float = 0.0,
        overlap_window: Optional[Tuple[float, float]] = None,
        gap: Optional[Tuple[float, float]] = None,
    ) -> EpochStart:
        """Retire the live epoch and open a new one.

        ``overlap_window`` is the replayed region already covered by committed finals;
        ``gap`` is audio that could not be replayed and is recorded as a warning.
        """
        with self._lock:
            abandoned = tuple(key for key, item in self._items.items() if not item.is_final)
            epoch_id = self._next_epoch_id
            self._next_epoch_id += 1
            overlap_start, overlap_end = (None, None)
            if overlap_window is not None and overlap_window[1] > overlap_window[0]:
                overlap_start, overlap_end = overlap_window
            self._epoch = EpochHandle(
                stream_id=self.stream_id,
                epoch_id=epoch_id,
                provider_id=provider_id,
                offset_ms=offset_ms,
                overlap_start_ms=overlap_start,
                overlap_end_ms=overlap_end,
            )
            # Items of retired epochs can never be accepted again, so their state is dead.
            self._items.clear()
            gap_warning = None
            if gap is not None and gap[1] > gap[0]:
                gap_warning = self._warn(epoch_id, gap[0], gap[1], ContinuityReason.AUDIO_GAP)
            handle = self._epoch
        logger.info(
            "Stream %s opened transcript epoch %d (provider=%s, offset_ms=%.0f, abandoned=%d)",
            self.stream_id,
            epoch_id,
            provider_id,
            offset_ms,
            len(abandoned),
        )
        return EpochStart(handle=handle, abandoned_entry_keys=abandoned, gap_warning=gap_warning)

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest(self, result: STTTranscriptResult, epoch: EpochHandle) -> LedgerDecision:
        key = f"{result.stream_id}:{epoch.epoch_id}:{result.provider_event_id}"
        if result.stream_id != self.stream_id or epoch.stream_id != self.stream_id:
            return self._reject(LedgerAction.STREAM_MISMATCH, result, epoch, key)

        with self._lock:
            if self._epoch is None or epoch.epoch_id != self._epoch.epoch_id:
                return self._reject(LedgerAction.RETIRED_EPOCH, result, epoch, key)
            live = self._epoch
            start_ms = result.start_time_ms + live.offset_ms
            end_ms = max(start_ms, result.end_time_ms + live.offset_ms)

            item = self._items.get(key)
            if item is not None:
                if item.is_final:
                    return self._reject(LedgerAction.AFTER_FINAL, result, epoch, key)
                if result.revision <= item.revision:
                    return self._reject(LedgerAction.STALE, result, epoch, key)
                item.revision = result.revision
                self._items.move_to_end(key)
            else:
                item = _ItemState(revision=result.revision)
                self._items[key] = item
                self._evict_items()

            if not result.is_final:
                return self._partial(result, live, key, start_ms, end_ms)
            item.is_final = True
            return self._final(result, live, key, start_ms, end_ms)

    # ── Internals (lock held) ────────────────────────────────────────────────

    def _partial(
        self,
        result: STTTranscriptResult,
        epoch: EpochHandle,
        key: str,
        start_ms: float,
        end_ms: float,
    ) -> LedgerDecision:
        rec = self._reconcile(result.text.strip(), start_ms, end_ms, epoch)
        if rec.verdict is _Verdict.DUPLICATE:
            action = LedgerAction.SUPPRESSED_PARTIAL
        else:
            action = LedgerAction.PARTIAL
        return LedgerDecision(
            action=action,
            stream_id=self.stream_id,
            epoch_id=epoch.epoch_id,
            entry_key=key,
            revision=result.revision,
            text=rec.text if action is LedgerAction.PARTIAL else "",
            start_ms=rec.start_ms,
            end_ms=end_ms,
            duplicate_of=rec.duplicate_of,
            trimmed_prefix_tokens=rec.trimmed_tokens,
        )

    def _final(
        self,
        result: STTTranscriptResult,
        epoch: EpochHandle,
        key: str,
        start_ms: float,
        end_ms: float,
    ) -> LedgerDecision:
        text = result.text.strip()
        if not normalize_text(text):
            return LedgerDecision(
                action=LedgerAction.RETRACTED,
                stream_id=self.stream_id,
                epoch_id=epoch.epoch_id,
                entry_key=key,
                revision=result.revision,
                start_ms=start_ms,
                end_ms=end_ms,
            )

        rec = self._reconcile(text, start_ms, end_ms, epoch)
        if rec.verdict is _Verdict.DUPLICATE:
            self._duplicate_count += 1
            logger.debug(
                "Stream %s dropped duplicate final %s (duplicate of %s)",
                self.stream_id,
                key,
                rec.duplicate_of,
            )
            return LedgerDecision(
                action=LedgerAction.DUPLICATE_FINAL,
                stream_id=self.stream_id,
                epoch_id=epoch.epoch_id,
                entry_key=key,
                revision=result.revision,
                start_ms=start_ms,
                end_ms=end_ms,
                duplicate_of=rec.duplicate_of,
            )

        warning = None
        if rec.verdict is _Verdict.UNRESOLVED:
            warning = self._warn(
                epoch.epoch_id, start_ms, end_ms, ContinuityReason.UNRESOLVED_OVERLAP
            )
        self._finals.append(
            _CommittedFinal(
                entry_key=key,
                epoch_id=epoch.epoch_id,
                start_ms=rec.start_ms,
                end_ms=end_ms,
                normalized=normalize_text(rec.text),
            )
        )
        self._final_boundary_ms = max(self._final_boundary_ms, end_ms)
        return LedgerDecision(
            action=LedgerAction.FINAL,
            stream_id=self.stream_id,
            epoch_id=epoch.epoch_id,
            entry_key=key,
            revision=result.revision,
            text=rec.text,
            start_ms=rec.start_ms,
            end_ms=end_ms,
            trimmed_prefix_tokens=rec.trimmed_tokens,
            warning=warning,
        )

    def _reconcile(
        self, text: str, start_ms: float, end_ms: float, epoch: EpochHandle
    ) -> _Reconciliation:
        tol = self.thresholds.interval_tolerance_ms
        candidates = [
            final
            for final in self._finals
            if final.epoch_id < epoch.epoch_id and self._intersects(final, start_ms, end_ms)
        ]
        if not candidates:
            return _Reconciliation(_Verdict.NEW, text, start_ms)

        candidates.sort(key=lambda final: final.start_ms)
        best = max(candidates, key=lambda f: min(end_ms, f.end_ms) - max(start_ms, f.start_ms))
        if self._coverage(candidates, start_ms, end_ms) >= self.thresholds.covered_ratio:
            return _Reconciliation(_Verdict.DUPLICATE, text, start_ms, best.entry_key)

        in_window = (
            epoch.has_overlap_window
            and start_ms <= epoch.overlap_end_ms + tol
            and end_ms >= epoch.overlap_start_ms - tol
        )
        if not in_window:
            return _Reconciliation(_Verdict.UNRESOLVED, text, start_ms)

        prior = " ".join(final.normalized for final in candidates)
        # Trim before fuzzy matching: a replayed tail plus genuinely new words can score a
        # high similarity ratio, and dropping it would lose the new words.
        trimmed = self._trim_repeated_prefix(text, prior)
        if trimmed is not None:
            remaining, tokens = trimmed
            if not remaining:
                return _Reconciliation(_Verdict.DUPLICATE, text, start_ms, best.entry_key)
            covered_end = max(final.end_ms for final in candidates)
            new_start = min(end_ms, max(start_ms, covered_end))
            return _Reconciliation(_Verdict.TRIMMED, remaining, new_start, None, tokens)

        if self._is_similar(normalize_text(text), prior):
            return _Reconciliation(_Verdict.DUPLICATE, text, start_ms, best.entry_key)
        return _Reconciliation(_Verdict.UNRESOLVED, text, start_ms)

    def _intersects(self, final: _CommittedFinal, start_ms: float, end_ms: float) -> bool:
        tol = self.thresholds.interval_tolerance_ms
        duration = end_ms - start_ms
        if duration <= 0:
            return final.start_ms - tol <= start_ms <= final.end_ms + tol
        overlap = min(end_ms, final.end_ms) - max(start_ms, final.start_ms)
        # Ignore boundary jitter, but never let the tolerance hide a short result entirely.
        return overlap > min(tol, duration * 0.5)

    def _coverage(self, candidates: List[_CommittedFinal], start_ms: float, end_ms: float) -> float:
        duration = end_ms - start_ms
        if duration <= 0:
            tol = self.thresholds.interval_tolerance_ms
            inside = any(f.start_ms - tol <= start_ms <= f.end_ms + tol for f in candidates)
            return 1.0 if inside else 0.0
        covered = 0.0
        cursor = start_ms
        for final in candidates:  # sorted by start
            lo = max(cursor, final.start_ms)
            hi = min(end_ms, final.end_ms)
            if hi > lo:
                covered += hi - lo
                cursor = hi
        return covered / duration

    def _is_similar(self, candidate: str, prior: str) -> bool:
        if not candidate:
            return True
        if not prior:
            return False
        ratio = difflib.SequenceMatcher(None, candidate, prior, autojunk=False).ratio()
        if ratio >= self.thresholds.text_similarity:
            return True
        if len(candidate) < self.thresholds.min_containment_chars or len(candidate) > len(prior):
            return False
        cand_tokens = candidate.split()
        prior_tokens = prior.split()
        if len(cand_tokens) >= 2:
            a, b = cand_tokens, prior_tokens
        else:
            # Unspaced scripts (and single words) are compared by character.
            a, b = list(candidate), list(prior)
        matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
        longest = matcher.find_longest_match(0, len(a), 0, len(b)).size
        return longest / len(a) >= self.thresholds.containment_ratio

    def _trim_repeated_prefix(self, text: str, prior: str) -> Optional[Tuple[str, int]]:
        """Drop leading tokens of ``text`` that repeat the end of ``prior``."""
        raw_tokens = text.split()
        pairs: List[Tuple[int, str]] = [
            (index, token)
            for index, raw in enumerate(raw_tokens)
            for token in normalize_text(raw).split()
        ]
        prior_tokens = prior.split()
        cand_tokens = [token for _, token in pairs]
        longest = min(len(prior_tokens), len(cand_tokens))
        for size in range(longest, self.thresholds.min_trim_tokens - 1, -1):
            if size <= 0 or prior_tokens[-size:] != cand_tokens[:size]:
                continue
            raw_index = pairs[size - 1][0]
            if size < len(pairs) and pairs[size][0] == raw_index:
                return None  # match ends inside a raw token; cannot trim cleanly
            return " ".join(raw_tokens[raw_index + 1 :]), size
        return None

    def _warn(
        self, epoch_id: int, start_ms: float, end_ms: float, reason: ContinuityReason
    ) -> ContinuityWarning:
        warning = ContinuityWarning(
            stream_id=self.stream_id,
            epoch_id=epoch_id,
            start_ms=start_ms,
            end_ms=end_ms,
            reason=reason,
        )
        self._warnings.append(warning)
        self._warning_count += 1
        logger.info(
            "Stream %s continuity warning (%s) over %.0f-%.0f ms",
            self.stream_id,
            reason.value,
            start_ms,
            end_ms,
        )
        return warning

    def _evict_items(self) -> None:
        # Oldest items go first; a late event for an evicted item is treated as a new item.
        while len(self._items) > self.thresholds.max_items:
            self._items.popitem(last=False)

    def _reject(
        self,
        action: LedgerAction,
        result: STTTranscriptResult,
        epoch: EpochHandle,
        key: str,
    ) -> LedgerDecision:
        logger.debug(
            "Stream %s rejected %s revision %d as %s",
            self.stream_id,
            key,
            result.revision,
            action.value,
        )
        return LedgerDecision(
            action=action,
            stream_id=result.stream_id,
            epoch_id=epoch.epoch_id,
            entry_key=key,
            revision=result.revision,
        )


class TranscriptLedger:
    """Routes results to independent per-stream ledgers."""

    def __init__(self, thresholds: Optional[LedgerThresholds] = None) -> None:
        self.thresholds = thresholds or LedgerThresholds()
        self._lock = threading.Lock()
        self._streams: Dict[str, StreamTranscriptLedger] = {}

    def stream(self, stream_id: str) -> StreamTranscriptLedger:
        with self._lock:
            ledger = self._streams.get(stream_id)
            if ledger is None:
                ledger = StreamTranscriptLedger(stream_id, self.thresholds)
                self._streams[stream_id] = ledger
            return ledger

    def stream_ids(self) -> List[str]:
        with self._lock:
            return list(self._streams)

    def begin_epoch(
        self,
        stream_id: str,
        provider_id: str,
        offset_ms: float = 0.0,
        overlap_window: Optional[Tuple[float, float]] = None,
        gap: Optional[Tuple[float, float]] = None,
    ) -> EpochStart:
        return self.stream(stream_id).begin_epoch(provider_id, offset_ms, overlap_window, gap)

    def ingest(self, result: STTTranscriptResult, epoch: EpochHandle) -> LedgerDecision:
        return self.stream(epoch.stream_id).ingest(result, epoch)

    def last_final_boundary_ms(self, stream_id: str) -> float:
        return self.stream(stream_id).last_final_boundary_ms

    def warnings(self, stream_id: Optional[str] = None) -> List[ContinuityWarning]:
        if stream_id is not None:
            return self.stream(stream_id).warnings()
        with self._lock:
            ledgers = list(self._streams.values())
        return [warning for ledger in ledgers for warning in ledger.warnings()]
