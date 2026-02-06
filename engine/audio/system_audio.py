import sys
import threading
from typing import List, Optional
from queue import Queue, Empty

from engine.audio.input_manager import AudioInput, AudioDevice, AudioSourceType
from engine.core.logging import get_logger
from engine.core.exceptions import AudioError

logger = get_logger("audio.system")


class SystemAudioInput(AudioInput):
    """
    System audio loopback capture.
    Uses soundcard library for cross-platform loopback support.
    """

    def __init__(self, config: dict, device_id: Optional[str] = None):
        self._sample_rate = config.get("sample_rate", 16000)
        self._channels = config.get("channels", 1)
        self._chunk_ms = config.get("chunk_ms", 20)
        self._device_id = device_id

        self._active = False
        self._queue: Queue = Queue(maxsize=200)
        self._capture_thread: Optional[threading.Thread] = None
        self._loopback_mic = None

    def start(self) -> None:
        if self._active:
            return

        try:
            import soundcard
        except ImportError:
            raise AudioError("soundcard library is not installed")

        self._loopback_mic = self._resolve_loopback(self._device_id)
        if not self._loopback_mic:
            raise AudioError("No loopback device found")

        self._active = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="system-audio-capture",
        )
        self._capture_thread.start()

        logger.info(
            "System audio capture started (rate=%d, channels=%d, device=%s)",
            self._sample_rate, self._channels,
            getattr(self._loopback_mic, "name", "unknown"),
        )

    def stop(self) -> None:
        if not self._active:
            return

        self._active = False

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)
        self._capture_thread = None
        self._loopback_mic = None

        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

        logger.info("System audio capture stopped")

    def read_chunk(self) -> bytes:
        try:
            return self._queue.get(timeout=0.1)
        except Empty:
            return b""

    def is_active(self) -> bool:
        return self._active

    def list_devices(self) -> List[AudioDevice]:
        try:
            import soundcard
        except ImportError:
            raise AudioError("soundcard library is not installed")

        devices = []
        try:
            speakers = soundcard.all_speakers()
            for speaker in speakers:
                devices.append(AudioDevice(
                    id=speaker.id,
                    name=f"{speaker.name} (Loopback)",
                    source_type=AudioSourceType.SYSTEM,
                    sample_rate=self._sample_rate,
                    channels=self._channels,
                ))
        except Exception as e:
            logger.error("Failed to enumerate system audio devices: %s", e)

        return devices

    def _capture_loop(self):
        import numpy as np

        frames_per_chunk = int(self._sample_rate * self._chunk_ms / 1000)

        try:
            recorder = self._loopback_mic.recorder(
                samplerate=self._sample_rate,
                channels=self._channels,
            )
            recorder.__enter__()
        except Exception as e:
            logger.error("Failed to open loopback recorder: %s", e)
            self._active = False
            return

        try:
            while self._active:
                try:
                    data = recorder.record(numframes=frames_per_chunk)
                    # Convert float32 [-1.0, 1.0] to int16 bytes
                    audio_int16 = (data * 32767).astype(np.int16)
                    if self._channels == 1 and audio_int16.ndim > 1:
                        audio_int16 = audio_int16[:, 0]
                    chunk_bytes = audio_int16.tobytes()

                    try:
                        self._queue.put_nowait(chunk_bytes)
                    except Exception:
                        pass  # Drop frame if queue full
                except Exception as e:
                    if self._active:
                        logger.error("System audio read error: %s", e)
                    break
        finally:
            try:
                recorder.__exit__(None, None, None)
            except Exception as e:
                logger.debug("Recorder cleanup error: %s", e)

    def _resolve_loopback(self, device_id: Optional[str]):
        import soundcard

        if device_id:
            try:
                speakers = soundcard.all_speakers()
                for speaker in speakers:
                    if speaker.id == device_id:
                        return soundcard.get_microphone(
                            speaker.id, include_loopback=True
                        )
            except Exception as e:
                logger.warning("Failed to resolve device %s: %s", device_id, e)

        # Fall back to default speaker loopback
        try:
            default_speaker = soundcard.default_speaker()
            loopback = soundcard.get_microphone(
                default_speaker.id, include_loopback=True
            )
            logger.info("Using default loopback: %s", default_speaker.name)
            return loopback
        except Exception as e:
            logger.error("Failed to get default loopback device: %s", e)
            return None
