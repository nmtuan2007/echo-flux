import contextlib
import sys
from queue import Empty, Queue
from typing import List, Optional

import numpy as np

from engine.audio.input_manager import AudioDevice, AudioInput, AudioSourceType
from engine.core.exceptions import AudioError
from engine.core.logging import get_logger

logger = get_logger("audio.system")

_VIRTUAL_LOOPBACK_TOKENS = ("blackhole", "loopback", "soundflower", "monitor", "stereo mix")


def is_virtual_loopback_name(name: str) -> bool:
    lowered = (name or "").lower()
    return any(tok in lowered for tok in _VIRTUAL_LOOPBACK_TOKENS)


class SystemAudioInput(AudioInput):

    def __init__(
        self,
        config: dict,
        device_id: Optional[str] = None,
        device_name: Optional[str] = None,
    ):
        self._target_sample_rate = config.get("sample_rate", 16000)
        self._channels = config.get("channels", 1)
        self._chunk_ms = config.get("chunk_ms", 20)
        self._device_id = device_id
        self._device_name = device_name

        self._pa = None
        self._stream = None
        self._active = False
        self._queue: Queue = Queue(maxsize=200)
        self._native_sample_rate: Optional[int] = None
        self._native_channels: Optional[int] = None

    def start(self) -> None:
        if self._active:
            return

        if sys.platform != "win32":
            self._start_virtual_loopback()
            return

        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            raise AudioError(
                "pyaudiowpatch is not installed. Install with: pip install pyaudiowpatch"
            ) from None

        self._pa = pyaudio.PyAudio()

        loopback_device = self._find_loopback_device(self._device_id)
        if not loopback_device:
            self._pa.terminate()
            self._pa = None
            raise AudioError("No WASAPI loopback device found")

        self._native_sample_rate = int(loopback_device["defaultSampleRate"])
        self._native_channels = loopback_device["maxInputChannels"]

        frames_per_chunk = int(self._native_sample_rate * self._chunk_ms / 1000)

        logger.info(
            "Opening WASAPI loopback: device='%s', native_rate=%d, native_channels=%d, "
            "target_rate=%d, target_channels=%d, frames_per_chunk=%d",
            loopback_device["name"], self._native_sample_rate, self._native_channels,
            self._target_sample_rate, self._channels, frames_per_chunk,
        )

        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self._native_channels,
                rate=self._native_sample_rate,
                input=True,
                input_device_index=loopback_device["index"],
                frames_per_buffer=frames_per_chunk,
                stream_callback=self._audio_callback,
            )
            self._active = True
            self._stream.start_stream()

            logger.info(
                "System audio capture started via WASAPI loopback: %s",
                loopback_device["name"],
            )
        except Exception as e:
            self._cleanup()
            raise AudioError(f"Failed to open WASAPI loopback: {e}") from e

    def _start_virtual_loopback(self) -> None:
        """Fallback for macOS/Linux using virtual audio loopback/monitor input devices (e.g. BlackHole)."""
        try:
            import pyaudio
        except ImportError:
            raise AudioError("pyaudio is not installed") from None

        self._pa = pyaudio.PyAudio()
        candidates = []
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                candidates.append(info)

        selected = None
        if self._device_id:
            selected = next((d for d in candidates if str(d["index"]) == str(self._device_id)), None)
            if selected and self._device_name:
                needle = self._device_name.replace(" (Loopback)", "").lower()
                if needle and needle not in selected.get("name", "").lower():
                    selected = None
        if selected is None and self._device_name:
            needle = self._device_name.replace(" (Loopback)", "").lower()
            selected = next(
                (d for d in candidates if needle and needle in d.get("name", "").lower()),
                None,
            )
        if selected is None:
            selected = next((d for d in candidates if is_virtual_loopback_name(d.get("name", ""))), None)
        if selected is None:
            self._pa.terminate()
            self._pa = None
            raise AudioError("No virtual loopback device found (install BlackHole or configure a monitor input)")

        self._native_sample_rate = int(selected.get("defaultSampleRate", self._target_sample_rate))
        self._native_channels = max(1, min(int(selected.get("maxInputChannels", 1)), 2))
        frames_per_chunk = int(self._native_sample_rate * self._chunk_ms / 1000)

        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self._native_channels,
                rate=self._native_sample_rate,
                input=True,
                input_device_index=selected["index"],
                frames_per_buffer=frames_per_chunk,
                stream_callback=self._audio_callback,
            )
            self._active = True
            self._stream.start_stream()
            logger.info("System audio capture started via virtual loopback: %s", selected.get("name"))
        except Exception as e:
            self._cleanup()
            raise AudioError(f"Failed to open virtual loopback: {e}") from e

    def stop(self) -> None:
        if not self._active:
            return

        self._active = False
        self._cleanup()

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
        if sys.platform != "win32":
            try:
                import pyaudio
            except ImportError:
                raise AudioError("pyaudio is not installed") from None
            pa = pyaudio.PyAudio()
            devices: List[AudioDevice] = []
            try:
                for i in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(i)
                    if info.get("maxInputChannels", 0) > 0 and is_virtual_loopback_name(
                        info.get("name", "")
                    ):
                        devices.append(
                            AudioDevice(
                                id=str(info["index"]),
                                name=f"{info['name']} (Loopback)",
                                source_type=AudioSourceType.SYSTEM,
                                sample_rate=int(info.get("defaultSampleRate", 16000)),
                                channels=int(info.get("maxInputChannels", 1)),
                            )
                        )
            finally:
                pa.terminate()
            return devices

        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            raise AudioError("pyaudiowpatch is not installed") from None

        pa = pyaudio.PyAudio()
        devices = []

        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info["hostApi"] != wasapi_info["index"]:
                    continue
                if not info.get("isLoopbackDevice", False):
                    continue
                devices.append(AudioDevice(
                    id=str(info["index"]),
                    name=f"{info['name']} (Loopback)",
                    source_type=AudioSourceType.SYSTEM,
                    sample_rate=int(info["defaultSampleRate"]),
                    channels=info["maxInputChannels"],
                ))
        except Exception as e:
            logger.error("Failed to enumerate loopback devices: %s", e)
        finally:
            pa.terminate()

        return devices

    def _find_loopback_device(self, device_id: Optional[str]) -> Optional[dict]:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            return None

        try:
            wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError as e:
            logger.error("WASAPI not available: %s", e)
            return None

        logger.info("Scanning WASAPI devices (count=%d)...", self._pa.get_device_count())

        loopback_devices = []
        default_output_name = None

        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)

            if info["hostApi"] != wasapi_info["index"]:
                continue

            is_loopback = info.get("isLoopbackDevice", False)

            if info["maxOutputChannels"] > 0 and not is_loopback:
                if i == wasapi_info.get("defaultOutputDevice", -1):
                    default_output_name = info["name"]
                    logger.info(
                        "  [%d] OUTPUT (default): '%s' rate=%d ch=%d",
                        i, info["name"], int(info["defaultSampleRate"]),
                        info["maxOutputChannels"],
                    )
                else:
                    logger.info(
                        "  [%d] OUTPUT: '%s' rate=%d ch=%d",
                        i, info["name"], int(info["defaultSampleRate"]),
                        info["maxOutputChannels"],
                    )

            if is_loopback:
                loopback_devices.append(info)
                logger.info(
                    "  [%d] LOOPBACK: '%s' rate=%d ch=%d",
                    i, info["name"], int(info["defaultSampleRate"]),
                    info["maxInputChannels"],
                )

        if not loopback_devices:
            logger.error("No WASAPI loopback devices found")
            return None

        # If a specific device was requested, verify index/name or fall back by name
        if device_id:
            for dev in loopback_devices:
                if str(dev["index"]) == device_id and (
                    not self._device_name
                    or self._device_name.replace(" (Loopback)", "").lower() in dev["name"].lower()
                ):
                    logger.info("Using requested loopback device: %s", dev["name"])
                    return dev
            logger.warning("Requested device_id=%s not found or mismatched; checking name", device_id)

        if self._device_name:
            needle = self._device_name.replace(" (Loopback)", "").lower()
            for dev in loopback_devices:
                if needle and needle in dev["name"].lower():
                    logger.info("Matched shifted loopback device by name: %s", dev["name"])
                    return dev

        # Try to match the default output device's loopback
        if default_output_name:
            for dev in loopback_devices:
                if default_output_name.lower() in dev["name"].lower():
                    logger.info(
                        "Matched default output loopback: %s", dev["name"]
                    )
                    return dev

        # Fall back to the first loopback device
        selected = loopback_devices[0]
        logger.info("Using first available loopback: %s", selected["name"])
        return selected

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            logger.debug("Loopback audio callback status: %s", status)

        if not self._active or not in_data:
            return (None, 0)  # paContinue == 0 in both pyaudio and pyaudiowpatch

        try:
            processed = self._process_audio(in_data)
            if processed:
                with contextlib.suppress(Exception):
                    self._queue.put_nowait(processed)
        except Exception as e:
            logger.error("Audio processing error: %s", e)

        return (None, 0)

    def _process_audio(self, raw_data: bytes) -> bytes:
        # Use frombuffer to avoid "fromstring" binary mode error in newer numpy versions
        audio = np.frombuffer(raw_data, dtype=np.int16)

        # Downmix to mono if needed
        if self._native_channels and self._native_channels > 1:
            audio = audio.reshape(-1, self._native_channels)
            audio = audio.mean(axis=1).astype(np.int16)

        # Resample if needed
        if self._native_sample_rate and self._native_sample_rate != self._target_sample_rate:
            audio = self._resample_int16(audio, self._native_sample_rate, self._target_sample_rate)

        return audio.tobytes()

    @staticmethod
    def _resample_int16(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        if from_rate == to_rate or len(audio) == 0:
            return audio

        # Exact integer decimation (e.g. 48k -> 16k) with boxcar averaging prevents aliasing
        # and chunk-boundary phase jumps that degrade speech recognition.
        if from_rate > to_rate and from_rate % to_rate == 0:
            factor = from_rate // to_rate
            trimmed = (len(audio) // factor) * factor
            if trimmed > 0:
                return audio[:trimmed].reshape(-1, factor).mean(axis=1).astype(np.int16)

        ratio = to_rate / from_rate
        n_in = len(audio)
        n_out = max(1, int(n_in * ratio))

        indices = np.linspace(0, n_in, n_out, endpoint=False)
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.minimum(idx_floor + 1, n_in - 1)
        frac = (indices - idx_floor).astype(np.float32)

        resampled = audio[idx_floor].astype(np.float32) * (1 - frac) + \
                    audio[idx_ceil].astype(np.float32) * frac

        return resampled.astype(np.int16)

    def _cleanup(self):
        if self._stream:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.info("Stream cleanup: %s", e)
            self._stream = None

        if self._pa:
            try:
                self._pa.terminate()
            except Exception as e:
                logger.info("PyAudio cleanup: %s", e)
            self._pa = None
