import threading
from queue import Queue, Empty
from typing import List

import numpy as np

from engine.audio.input_manager import AudioInput, AudioDevice
from engine.audio.microphone import MicrophoneInput
from engine.audio.system_audio import SystemAudioInput
from engine.core.logging import get_logger

logger = get_logger("audio.mixer")


class MixerInput(AudioInput):
    """Mixes a MicrophoneInput and a SystemAudioInput into a single stream.

    Both sources are started on the calling thread (they each use their own
    internal PyAudio callback threads).  A dedicated mixer thread continuously
    drains both queues, averages the two int16 arrays at 50 % gain each, and
    places the result into the output queue consumed by read_chunk().

    Signal math:
        mixed = clip((mic_float * 0.5) + (sys_float * 0.5), -32768, 32767)

    Both inputs are configured with the same target sample-rate and chunk_ms,
    so downstream lengths should be identical in the common case.  If they
    differ (e.g. at startup) the shorter buffer is zero-padded to match.
    """

    def __init__(
        self,
        mic_input: MicrophoneInput,
        sys_input: SystemAudioInput,
        config: dict,
    ) -> None:
        self._mic = mic_input
        self._sys = sys_input
        self._config = config

        self._active = False
        self._out_queue: Queue = Queue(maxsize=300)
        self._mixer_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Source dominance tracking (updated by the mixer thread)
        self._mic_energy_accum: float = 0.0
        self._sys_energy_accum: float = 0.0
        self._source_lock = threading.Lock()

    # ------------------------------------------------------------------
    # AudioInput interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._active:
            return

        logger.info("MixerInput: starting microphone sub-input …")
        self._mic.start()

        logger.info("MixerInput: starting system-audio sub-input …")
        self._sys.start()

        self._stop_event.clear()
        self._active = True

        self._mixer_thread = threading.Thread(
            target=self._mix_loop,
            name="MixerThread",
            daemon=True,
        )
        self._mixer_thread.start()
        logger.info("MixerInput started (mic + system audio combined).")

    def stop(self) -> None:
        if not self._active:
            return

        logger.info("MixerInput: stopping …")
        self._active = False
        self._stop_event.set()

        if self._mixer_thread:
            self._mixer_thread.join(timeout=2.0)
            self._mixer_thread = None

        self._mic.stop()
        self._sys.stop()

        # Drain output queue
        while not self._out_queue.empty():
            try:
                self._out_queue.get_nowait()
            except Empty:
                break

        logger.info("MixerInput stopped.")

    def read_chunk(self) -> bytes:
        try:
            return self._out_queue.get(timeout=0.1)
        except Empty:
            return b""

    def is_active(self) -> bool:
        return self._active and self._mic.is_active() and self._sys.is_active()

    def list_devices(self) -> List[AudioDevice]:
        # Devices are configured individually on each sub-input.
        return []

    def get_dominant_source(self) -> str:
        """Return the dominant audio source integrated over the recent processing window.
        Returns one of: "mic", "system", "both".
        """
        with self._source_lock:
            mic = self._mic_energy_accum
            sys = self._sys_energy_accum
            
            if mic == 0.0 and sys == 0.0:
                return "both"
                
            _DOMINANCE_RATIO = 3.0
            if mic > sys * _DOMINANCE_RATIO:
                return "mic"
            elif sys > mic * _DOMINANCE_RATIO:
                return "system"
            else:
                return "both"

    def reset_source_tracking(self) -> None:
        """Reset the integrated energy counters."""
        with self._source_lock:
            self._mic_energy_accum = 0.0
            self._sys_energy_accum = 0.0

    # ------------------------------------------------------------------
    # Internal mixer thread
    # ------------------------------------------------------------------

    def _mix_loop(self) -> None:
        """Continuously drain both source queues and mix them.

        Bug that this fixes
        -------------------
        The naive approach calls read_chunk() on both sources sequentially.
        Each read_chunk() blocks for up to 0.1 s, so one iteration can
        take up to 0.2 s.  Meanwhile, the mic PyAudio callback enqueues a
        new chunk every 20 ms.  After ~100 ms of the remote speaker being
        silent (sys queue empty), the mic queue has 5 unread chunks.
        Once the mic queue (maxsize=200) fills up (~4 s), the PyAudio
        callback silently drops frames → words are missed.

        Fix: use non-blocking get_nowait() on both *internal* queues so
        neither source ever blocks the other.  When both queues are empty
        we sleep for half a chunk duration before retrying.
        """
        logger.info("Mixer thread started.")
        chunk_duration_s = self._config.get("chunk_ms", 20) / 1000.0

        try:
            while not self._stop_event.is_set():
                # Non-blocking reads — never let one source stall the other.
                # We access _queue directly because the public read_chunk()
                # API has a 0.1 s blocking timeout that causes the race above.
                mic_chunk = b""
                sys_chunk = b""

                try:
                    mic_chunk = self._mic._queue.get_nowait()  # type: ignore[attr-defined]
                except (Empty, AttributeError):
                    pass

                try:
                    sys_chunk = self._sys._queue.get_nowait()  # type: ignore[attr-defined]
                except (Empty, AttributeError):
                    pass

                if not mic_chunk and not sys_chunk:
                    # Both empty — yield for half a chunk period.
                    self._stop_event.wait(timeout=chunk_duration_s * 0.5)
                    continue

                # Track energy continuously over the entire ASR segment
                mic_rms = self._rms(mic_chunk)
                sys_rms = self._rms(sys_chunk)
                
                with self._source_lock:
                    self._mic_energy_accum += mic_rms
                    self._sys_energy_accum += sys_rms

                mixed = self._mix_chunks(mic_chunk, sys_chunk)
                if mixed and not self._out_queue.full():
                    try:
                        self._out_queue.put_nowait(mixed)
                    except Exception:
                        pass  # Drop if output queue is full

        except Exception as e:
            logger.error("Mixer thread error: %s", e, exc_info=True)
        finally:
            logger.info("Mixer thread ended.")

    @staticmethod
    def _mix_chunks(a: bytes, b: bytes) -> bytes:
        """Mix two int16 byte buffers.

        When BOTH sources have data: mix at 50 % gain each to prevent
        clipping when mic and speaker overlap.

        When only ONE source has data (the other is between words / silent):
        pass through at full gain so Whisper receives a normal-volume signal.
        Zero-padding handles the rare case where chunk lengths differ.
        """
        has_a = bool(a)
        has_b = bool(b)

        if not has_a and not has_b:
            return b""

        # Fast path: only one source active — full pass-through, no numpy needed.
        if has_a and not has_b:
            return a
        if has_b and not has_a:
            return b

        # Both have data — sum them and clip.
        a_arr = np.frombuffer(a, dtype=np.int16).astype(np.float32)
        b_arr = np.frombuffer(b, dtype=np.int16).astype(np.float32)

        # Zero-pad the shorter array (rare at startup)
        length = max(len(a_arr), len(b_arr))
        if len(a_arr) < length:
            a_arr = np.pad(a_arr, (0, length - len(a_arr)))
        if len(b_arr) < length:
            b_arr = np.pad(b_arr, (0, length - len(b_arr)))

        # Standard addition mixing. Using `*0.5` causes volume drops!
        mixed = np.clip(a_arr + b_arr, -32768, 32767).astype(np.int16)
        return mixed.tobytes()

    @staticmethod
    def _rms(chunk: bytes) -> float:
        """Compute RMS energy of an int16 PCM byte buffer.  Returns 0.0 for empty input."""
        if not chunk:
            return 0.0
        arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(arr ** 2)))
