"""Bounded in-memory PCM ring buffer used to replay audio after an STT failure.

One buffer per audio stream (``mic`` / ``system``). Audio is never written to disk.

Timeline: every appended frame advances a per-stream *session timeline* measured in
milliseconds of audio captured since the buffer was created. Provider timelines restart on
every new connection, so replay plans and the transcript ledger work on this timeline and
map provider-relative times onto it with an explicit offset.

Memory: the backing store is a single ``bytearray`` preallocated at construction and never
resized. Its size is ``capacity_ms / 1000 * sample_rate * channels * sample_width_bytes``;
at the defaults (15 s, 16 kHz, mono, int16) that is 480 000 bytes per stream.

Thread ownership: the stream's ``Capture-{stream}`` thread calls ``append``; the thread that
owns the stream's failover controller calls ``read``. Both go through an internal lock, so
no other synchronisation is required.
"""

import math
import threading
from dataclasses import dataclass
from typing import List

from engine.core.logging import get_logger
from engine.providers.types import AudioFrame

logger = get_logger("session.audio_ring_buffer")

DEFAULT_CAPACITY_MS = 15000.0

# Float ms -> frame conversion must not lose a whole frame to rounding noise.
_FRAME_EPSILON = 1e-6


@dataclass(frozen=True)
class AudioSpan:
    """Half-open interval ``[start_ms, end_ms)`` on a stream's session timeline."""

    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


@dataclass(frozen=True)
class ReplaySlice:
    """A copy of retained audio, with how much of the requested range was already evicted."""

    stream_id: str
    pcm_data: bytes
    sample_rate: int
    channels: int
    sample_width_bytes: int
    requested_start_ms: float
    start_ms: float
    end_ms: float

    @property
    def missing_ms(self) -> float:
        """Audio between the requested start and the oldest retained frame."""
        return max(0.0, self.start_ms - self.requested_start_ms)

    @property
    def is_complete(self) -> bool:
        return self.missing_ms <= 0.0

    def to_frames(
        self,
        frame_ms: float = 100.0,
        first_sequence: int = 0,
        monotonic_timestamp: float = 0.0,
    ) -> List[AudioFrame]:
        """Split the slice into ``AudioFrame`` chunks of at most ``frame_ms``."""
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        frame_bytes = self.channels * self.sample_width_bytes
        samples_per_chunk = max(1, math.floor(frame_ms * self.sample_rate / 1000.0))
        chunk_bytes = samples_per_chunk * frame_bytes
        frames: List[AudioFrame] = []
        view = memoryview(self.pcm_data)
        for index, offset in enumerate(range(0, len(view), chunk_bytes)):
            chunk = bytes(view[offset : offset + chunk_bytes])
            frames.append(
                AudioFrame(
                    stream_id=self.stream_id,
                    sequence=first_sequence + index,
                    pcm_data=chunk,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    monotonic_timestamp=monotonic_timestamp,
                    duration_ms=(len(chunk) // frame_bytes) * 1000.0 / self.sample_rate,
                )
            )
        return frames


class AudioRingBuffer:
    """Fixed-size circular store of the most recent PCM audio for one stream."""

    def __init__(
        self,
        stream_id: str,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width_bytes: int = 2,
        capacity_ms: float = DEFAULT_CAPACITY_MS,
    ) -> None:
        if sample_rate <= 0 or channels <= 0 or sample_width_bytes <= 0:
            raise ValueError("sample_rate, channels and sample_width_bytes must be positive")
        capacity_frames = math.floor(capacity_ms * sample_rate / 1000.0 + _FRAME_EPSILON)
        if capacity_frames <= 0:
            raise ValueError("capacity_ms must hold at least one sample")
        self.stream_id = stream_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes
        self._frame_bytes = channels * sample_width_bytes
        self._capacity_frames = capacity_frames
        self._buffer = bytearray(capacity_frames * self._frame_bytes)
        self._total_frames = 0
        self._discard_before_frames = 0
        self._lock = threading.Lock()

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def capacity_bytes(self) -> int:
        """Size of the preallocated backing store; constant for the buffer's lifetime."""
        return len(self._buffer)

    @property
    def capacity_ms(self) -> float:
        return self._frames_to_ms(self._capacity_frames)

    @property
    def end_ms(self) -> float:
        """Session-timeline position just after the newest captured frame."""
        with self._lock:
            return self._frames_to_ms(self._total_frames)

    def retained_span(self) -> AudioSpan:
        with self._lock:
            return AudioSpan(
                self._frames_to_ms(self._oldest_frame()),
                self._frames_to_ms(self._total_frames),
            )

    # ── Capture side ─────────────────────────────────────────────────────────

    def append(self, pcm_data: bytes) -> AudioSpan:
        """Store frame-aligned PCM and return the timeline interval it occupies."""
        if len(pcm_data) % self._frame_bytes:
            raise ValueError(
                f"PCM length {len(pcm_data)} is not a multiple of frame size {self._frame_bytes}"
            )
        frames = len(pcm_data) // self._frame_bytes
        view = memoryview(pcm_data)
        with self._lock:
            start_frame = self._total_frames
            skip = max(0, frames - self._capacity_frames)
            if skip:
                view = view[skip * self._frame_bytes :]
            position = ((start_frame + skip) % self._capacity_frames) * self._frame_bytes
            first = min(len(view), len(self._buffer) - position)
            self._buffer[position : position + first] = view[:first]
            if len(view) > first:
                self._buffer[: len(view) - first] = view[first:]
            self._total_frames += frames
            return AudioSpan(
                self._frames_to_ms(start_frame), self._frames_to_ms(self._total_frames)
            )

    # ── Replay side ──────────────────────────────────────────────────────────

    def read(self, start_ms: float) -> ReplaySlice:
        """Copy retained audio from ``start_ms`` to the newest frame.

        A start older than the retained window is clamped; the shortfall is reported by
        ``ReplaySlice.missing_ms`` instead of being hidden.
        """
        with self._lock:
            requested = min(max(0, self._ms_to_frames(start_ms)), self._total_frames)
            start = max(requested, self._oldest_frame())
            end = self._total_frames
            pcm = self._copy(start, end)
            return ReplaySlice(
                stream_id=self.stream_id,
                pcm_data=pcm,
                sample_rate=self.sample_rate,
                channels=self.channels,
                sample_width_bytes=self.sample_width_bytes,
                requested_start_ms=self._frames_to_ms(requested),
                start_ms=self._frames_to_ms(start),
                end_ms=self._frames_to_ms(end),
            )

    def clear(self) -> None:
        """Forget retained audio (zero-filled). The session timeline keeps advancing."""
        with self._lock:
            self._buffer[:] = bytes(len(self._buffer))
            self._discard_before_frames = self._total_frames
        logger.debug("Cleared audio ring buffer for stream %s", self.stream_id)

    # ── Internals (lock held) ────────────────────────────────────────────────

    def _oldest_frame(self) -> int:
        return max(self._total_frames - self._capacity_frames, self._discard_before_frames, 0)

    def _copy(self, start_frame: int, end_frame: int) -> bytes:
        count = end_frame - start_frame
        if count <= 0:
            return b""
        position = (start_frame % self._capacity_frames) * self._frame_bytes
        length = count * self._frame_bytes
        first = min(length, len(self._buffer) - position)
        data = bytes(self._buffer[position : position + first])
        if length > first:
            data += bytes(self._buffer[: length - first])
        return data

    def _frames_to_ms(self, frames: int) -> float:
        return frames * 1000.0 / self.sample_rate

    def _ms_to_frames(self, ms: float) -> int:
        return math.floor(ms * self.sample_rate / 1000.0 + _FRAME_EPSILON)
