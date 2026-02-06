import threading
from typing import List, Optional
from queue import Queue, Empty

from engine.audio.input_manager import AudioInput, AudioDevice, AudioSourceType
from engine.core.logging import get_logger
from engine.core.exceptions import AudioError

logger = get_logger("audio.microphone")


class MicrophoneInput(AudioInput):

    def __init__(self, config: dict, device_id: Optional[int] = None):
        self._sample_rate = config.get("sample_rate", 16000)
        self._channels = config.get("channels", 1)
        self._chunk_ms = config.get("chunk_ms", 20)
        self._format_width = 2  # int16
        self._device_id = device_id

        self._pa = None
        self._stream = None
        self._active = False
        self._queue: Queue = Queue(maxsize=200)

    def start(self) -> None:
        if self._active:
            return

        try:
            import pyaudio
        except ImportError:
            raise AudioError("pyaudio is not installed")

        self._pa = pyaudio.PyAudio()

        frames_per_chunk = int(self._sample_rate * self._chunk_ms / 1000)

        kwargs = {
            "format": pyaudio.paInt16,
            "channels": self._channels,
            "rate": self._sample_rate,
            "input": True,
            "frames_per_buffer": frames_per_chunk,
            "stream_callback": self._audio_callback,
        }

        if self._device_id is not None:
            kwargs["input_device_index"] = self._device_id

        try:
            self._stream = self._pa.open(**kwargs)
            self._active = True
            self._stream.start_stream()
            logger.info(
                "Microphone started (rate=%d, channels=%d, chunk_ms=%d, device=%s)",
                self._sample_rate, self._channels, self._chunk_ms,
                self._device_id if self._device_id is not None else "default",
            )
        except Exception as e:
            self._cleanup()
            raise AudioError(f"Failed to open microphone: {e}") from e

    def stop(self) -> None:
        if not self._active:
            return

        self._active = False
        self._cleanup()

        # Drain remaining items
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

        logger.info("Microphone stopped")

    def read_chunk(self) -> bytes:
        try:
            return self._queue.get(timeout=0.1)
        except Empty:
            return b""

    def is_active(self) -> bool:
        return self._active

    def list_devices(self) -> List[AudioDevice]:
        try:
            import pyaudio
        except ImportError:
            raise AudioError("pyaudio is not installed")

        pa = pyaudio.PyAudio()
        devices = []

        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    devices.append(AudioDevice(
                        id=str(i),
                        name=info.get("name", f"Device {i}"),
                        source_type=AudioSourceType.MICROPHONE,
                        sample_rate=int(info.get("defaultSampleRate", 16000)),
                        channels=min(int(info.get("maxInputChannels", 1)), 2),
                    ))
        finally:
            pa.terminate()

        return devices

    def _audio_callback(self, in_data, frame_count, time_info, status):
        import pyaudio

        if status:
            logger.debug("Audio callback status: %s", status)

        if self._active and in_data:
            try:
                self._queue.put_nowait(in_data)
            except Exception:
                pass  # Drop frame if queue is full

        return (None, pyaudio.paContinue)

    def _cleanup(self):
        if self._stream:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.debug("Stream cleanup error: %s", e)
            self._stream = None

        if self._pa:
            try:
                self._pa.terminate()
            except Exception as e:
                logger.debug("PyAudio cleanup error: %s", e)
            self._pa = None
