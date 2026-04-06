import logging
import threading
import time
import os
from collections import deque
from queue import Queue, Empty

from engine.audio.vad import VAD
from engine.audio.microphone import MicrophoneInput
from engine.audio.system_audio import SystemAudioInput

logger = logging.getLogger("echoflux.audio_pipeline")

SILENCE_FINALIZE_DELAY = 0.8

class AudioPipeline:
    """
    Manages capturing hardware audio streams and VAD processing.
    Emits speech chunks and finalization signals to the assigned callbacks.
    """
    def __init__(self, settings: dict, on_speech_chunk, on_finalize):
        self._settings = settings
        self.on_speech_chunk = on_speech_chunk
        self.on_finalize = on_finalize

        self._inputs = {}
        self._vads = {}
        self._audio_queues = {}
        self._running = False

        self._capture_threads = []
        self._process_threads = []

        self._initialize_devices()

    def _initialize_devices(self):
        sample_rate = int(os.getenv("ECHOFLUX_SAMPLE_RATE", "16000"))
        chunk_ms = int(os.getenv("ECHOFLUX_CHUNK_MS", "20"))
        
        audio_config = {
            "sample_rate": sample_rate,
            "channels": 1,
            "chunk_ms": chunk_ms,
        }

        audio_source = self._settings.get("audio.source", os.getenv("ECHOFLUX_AUDIO_SOURCE", "microphone"))
        mic_id_str = self._settings.get("audio.mic_device_id") or os.getenv("ECHOFLUX_MIC_DEVICE_ID")
        spk_id_str = self._settings.get("audio.speaker_device_id") or os.getenv("ECHOFLUX_SPEAKER_DEVICE_ID")
        legacy_device_id = os.getenv("ECHOFLUX_AUDIO_DEVICE_ID")

        if audio_source == "both":
            mic_dev = int(mic_id_str) if mic_id_str else None
            self._inputs["mic"] = MicrophoneInput(audio_config, device_id=mic_dev)
            self._inputs["system"] = SystemAudioInput(audio_config, device_id=spk_id_str)
            logger.info("AudioPipeline: Dual Stream enabled")
        elif audio_source == "system":
            device_id = spk_id_str or legacy_device_id
            self._inputs["system"] = SystemAudioInput(audio_config, device_id=device_id)
        else:
            raw_id = mic_id_str or legacy_device_id
            dev_id = int(raw_id) if raw_id else None
            self._inputs["mic"] = MicrophoneInput(audio_config, device_id=dev_id)

        for stream_id in self._inputs:
            self._audio_queues[stream_id] = Queue(maxsize=500)
            self._vads[stream_id] = VAD({
                "enabled": self._settings.get("vad.enabled", True),
                "threshold": self._settings.get("vad.threshold", 0.5),
                "sample_rate": sample_rate,
            })

    def start(self):
        self._running = True
        
        # Start hardware inputs
        for stream_id, audio_input in self._inputs.items():
            audio_input.start()

        # Start capture and process threads
        for stream_id, audio_input in self._inputs.items():
            ct = threading.Thread(
                target=self._capture_loop, 
                args=(stream_id, audio_input, self._audio_queues[stream_id]),
                name=f"Capture-{stream_id}",
                daemon=True
            )
            ct.start()
            self._capture_threads.append(ct)

            pt = threading.Thread(
                target=self._process_loop,
                args=(stream_id, self._audio_queues[stream_id], self._vads[stream_id]),
                name=f"Process-{stream_id}",
                daemon=True
            )
            pt.start()
            self._process_threads.append(pt)

    def stop(self):
        self._running = False

        for inp in self._inputs.values():
            inp.stop()

        for ct in self._capture_threads:
            ct.join(timeout=1.0)
            
        for pt in self._process_threads:
            pt.join(timeout=2.0)

        self._inputs.clear()
        self._vads.clear()
        self._audio_queues.clear()

    @property
    def stream_ids(self) -> list:
        return list(self._inputs.keys())

    def _capture_loop(self, stream_id: str, audio_input, audio_queue: Queue):
        chunk_count = 0
        try:
            while self._running:
                chunk = audio_input.read_chunk()
                if chunk:
                    chunk_count += 1
                    if not audio_queue.full():
                        audio_queue.put(chunk)
                else:
                    time.sleep(0.005)
        except Exception as e:
            logger.error("Capture thread error: %s", e)

    def _process_loop(self, stream_id: str, audio_queue: Queue, vad):
        was_speech = False
        silence_start_time = None
        has_pending_audio = False
        pre_speech_buffer = deque(maxlen=3)

        try:
            while self._running:
                chunks = []
                try:
                    first = audio_queue.get(timeout=0.1)
                    chunks.append(first)
                    while not audio_queue.empty() and len(chunks) < 10:
                        try:
                            chunks.append(audio_queue.get_nowait())
                        except Empty:
                            break
                except Empty:
                    if has_pending_audio and silence_start_time is not None:
                        if time.time() - silence_start_time >= SILENCE_FINALIZE_DELAY:
                            self.on_finalize(stream_id)
                            has_pending_audio = False
                            silence_start_time = None
                            was_speech = False
                    continue

                combined_audio = b"".join(chunks)
                is_speech = vad.process(combined_audio)

                if is_speech:
                    if not was_speech:
                        if pre_speech_buffer:
                            combined_audio = b"".join(pre_speech_buffer) + combined_audio
                            pre_speech_buffer.clear()
                    silence_start_time = None
                    has_pending_audio = True
                    was_speech = True
                    
                    self.on_speech_chunk(stream_id, combined_audio)
                else:
                    pre_speech_buffer.append(combined_audio)
                    if was_speech and silence_start_time is None:
                        silence_start_time = time.time()
                    
                    if silence_start_time is not None:
                        if time.time() - silence_start_time >= SILENCE_FINALIZE_DELAY and has_pending_audio:
                            self.on_finalize(stream_id)
                            has_pending_audio = False
                            silence_start_time = None
                            was_speech = False
                            vad.reset()

        except Exception as e:
            logger.error("Process thread error: %s", e)
