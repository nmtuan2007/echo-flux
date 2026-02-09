import threading
from typing import List, Optional
from queue import Queue, Empty
import numpy as np

from engine.audio.input_manager import AudioInput, AudioDevice, AudioSourceType
from engine.core.logging import get_logger
from engine.core.exceptions import AudioError

logger = get_logger("audio.system")


class SystemAudioInput(AudioInput):

    def __init__(self, config: dict, device_id: Optional[str] = None):
        self._target_sample_rate = config.get("sample_rate", 16000)
        self._channels = config.get("channels", 1)
        self._chunk_ms = config.get("chunk_ms", 20)
        self._device_id = device_id

        self._pa = None
        self._stream = None
        self._active = False
        self._queue: Queue = Queue(maxsize=200)
        self._native_sample_rate: Optional[int] = None
        self._native_channels: Optional[int] = None

    def start(self) -> None:
        if self._active:
            return

        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            raise AudioError(
                "pyaudiowpatch is not installed. Install with: pip install pyaudiowpatch"
            )

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
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            raise AudioError("pyaudiowpatch is not installed")

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

        # If a specific device was requested
        if device_id:
            for dev in loopback_devices:
                if str(dev["index"]) == device_id:
                    logger.info("Using requested loopback device: %s", dev["name"])
                    return dev
            logger.warning("Requested device_id=%s not found among loopback devices", device_id)

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
        import pyaudiowpatch as pyaudio

        if status:
            logger.debug("WASAPI callback status: %s", status)

        if not self._active or not in_data:
            return (None, pyaudio.paContinue)

        try:
            processed = self._process_audio(in_data)
            if processed:
                try:
                    self._queue.put_nowait(processed)
                except Exception:
                    pass
        except Exception as e:
            logger.error("Audio processing error: %s", e)

        return (None, pyaudio.paContinue)

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
        if from_rate == to_rate:
            return audio

        ratio = to_rate / from_rate
        n_in = len(audio)
        n_out = int(n_in * ratio)

        indices = np.linspace(0, n_in - 1, n_out)
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
