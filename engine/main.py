import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from engine.core.progress import install_progress_hijack, set_progress_callback
from engine.server.websocket_server import WebSocketServer
from engine.core.event_bus import EventBus
from engine.services.audio_pipeline import AudioPipeline
from engine.services.inference_controller import InferenceController

logger = logging.getLogger("echoflux.engine")


def _setup_cuda_paths():
    if sys.platform != "win32":
        return

    # Use sys.prefix to find the current venv root dynamically
    venv_base = Path(sys.prefix)
    nvidia_path = venv_base / "Lib" / "site-packages" / "nvidia"

    targets = [
        nvidia_path / "cublas" / "bin",
        nvidia_path / "cudnn" / "bin",
        nvidia_path / "cuda_runtime" / "bin",
    ]

    for target in targets:
        if target.exists():
            abs_path = str(target.resolve())
            try:
                os.add_dll_directory(abs_path)
            except Exception:
                pass
            os.environ["PATH"] = abs_path + os.pathsep + os.environ["PATH"]

    zlib_found = False
    for p in os.environ["PATH"].split(os.pathsep):
        if p and (Path(p) / "zlibwapi.dll").exists():
            zlib_found = True
            break

    if not zlib_found and (Path.cwd() / "zlibwapi.dll").exists():
        zlib_found = True

    if not zlib_found:
        logger.warning("zlibwapi.dll not found. CUDA might fail. Check System32.")


def _load_dotenv():
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def _setup_logging():
    log_dir = Path.home() / ".echoflux" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"session_{ts}.log"

    level_str = os.getenv("ECHOFLUX_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger.info("Logging initialized – file: %s", log_file)


class EchoFluxEngine:
    def __init__(self):
        self._ws_server = WebSocketServer(
            host=os.getenv("ECHOFLUX_HOST", "127.0.0.1"),
            port=int(os.getenv("ECHOFLUX_PORT", "8765")),
        )
        self._event_bus: Optional[EventBus] = None
        self._audio_pipeline: Optional[AudioPipeline] = None
        self._inference_controller: Optional[InferenceController] = None
        
        self._running = False
        self._result_task: Optional[asyncio.Task] = None
        
        self._client = None
        self._llm_assistant = None
        self._current_config = {}

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._event_bus = EventBus(self._loop)
        
        def _progress_cb(model_name: str, percent: int):
            msg = {"type": "download_progress", "model": model_name, "percent": percent}
            self._event_bus.emit(msg)
                
        set_progress_callback(_progress_cb)

        self._ws_server.on_start(self._on_client_start)
        self._ws_server.on_stop(self._on_client_stop)
        self._ws_server.on_suggestion(self._on_suggestion_request)
        self._ws_server.on_summary(self._on_summary_request)
        self._ws_server.on_request_models_list(self._on_request_models_list)
        self._ws_server.on_download_model(self._on_download_model)
        self._ws_server.on_delete_model(self._on_delete_model)
        self._ws_server.on_search_hub(self._on_search_hub)

        logger.info("EchoFlux Engine starting...")
        self._result_task = asyncio.create_task(self._broadcast_loop())
        await self._ws_server.start()

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await self._shutdown()

    async def _on_client_start(self, config: dict):
        logger.info("Starting pipeline with client config: %s", config)
        self._client = list(self._ws_server._clients)[0] if self._ws_server._clients else None
        self._current_config = config

        if self._running:
            await self._stop_pipeline()

        try:
            if config.get("llm.enabled", False):
                from engine.llm.assistant import LLMAssistant
                api_key = config.get("llm.api_key", "")
                model = config.get("llm.model", "openai/gpt-4o-mini")
                provider_url = config.get("llm.provider_url") or None
                if api_key and model:
                    self._llm_assistant = LLMAssistant(api_key=api_key, model=model, base_url=provider_url)
                else:
                    self._llm_assistant = None
            else:
                self._llm_assistant = None

            def _init_and_start():
                self._inference_controller = InferenceController(config, self._event_bus)
                self._audio_pipeline = AudioPipeline(
                    config, 
                    on_speech_chunk=self._inference_controller.handle_speech_chunk,
                    on_finalize=self._inference_controller.handle_finalize
                )
                self._inference_controller.start()
                self._audio_pipeline.start()

            await asyncio.to_thread(_init_and_start)
            
            self._running = True
            self._ws_server.is_capturing = True
            
        except Exception as e:
            logger.error("Failed to start pipeline: %s", e, exc_info=True)
            if self._client:
                await self._client.send(json.dumps({"type": "error", "message": str(e)}))

    async def _on_client_stop(self):
        await self._stop_pipeline()

    async def _stop_pipeline(self):
        if not self._running:
            return
        
        logger.info("Stopping pipeline...")
        self._running = False
        self._ws_server.is_capturing = False

        if self._audio_pipeline:
            self._audio_pipeline.stop()
        
        if self._inference_controller:
            stream_ids = self._audio_pipeline.stream_ids if self._audio_pipeline else []
            self._inference_controller.stop(stream_ids)
            
        self._audio_pipeline = None
        self._inference_controller = None
        logger.info("Pipeline stopped.")

    async def _shutdown(self):
        await self._stop_pipeline()
        if self._result_task:
            self._result_task.cancel()
        await self._ws_server.stop()

    async def _broadcast_loop(self):
        async for msg_dict in self._event_bus.consume():
            if self._ws_server:
                await self._ws_server.broadcast(msg_dict)

    def _ensure_llm_assistant(self, message: dict) -> bool:
        if self._llm_assistant and self._llm_assistant.is_available():
            return True

        llm_cfg = message.get("llm_config", {})
        api_key = llm_cfg.get("api_key", "")
        model = llm_cfg.get("model", "")
        provider_url = llm_cfg.get("provider_url") or None

        if not api_key or not model:
            return False

        try:
            from engine.llm.assistant import LLMAssistant
            self._llm_assistant = LLMAssistant(
                api_key=api_key,
                model=model,
                base_url=provider_url,
            )
            logger.info("LLM assistant initialized on-demand: model=%s", model)
            return self._llm_assistant.is_available()
        except Exception as e:
            logger.error("Failed to initialize LLM assistant on-demand: %s", e)
            return False

    async def _on_suggestion_request(self, message: dict, websocket):
        if not self._ensure_llm_assistant(message):
            await websocket.send(json.dumps({
                "type": "suggestion_result",
                "entry_id": message.get("entry_id", ""),
                "error": "AI Assistant is not configured. Please add your API key in Settings.",
            }))
            return

        entry_id = message.get("entry_id", "")
        target_text = message.get("target_text", "")
        context = message.get("context", [])

        def _callback(result: dict):
            msg = {"type": "suggestion_result", **result}
            self._event_bus.emit(msg)

        self._llm_assistant.request_suggestion(
            entry_id=entry_id,
            target_text=target_text,
            context=context,
            callback=_callback,
        )

    async def _on_summary_request(self, message: dict, websocket):
        if not self._ensure_llm_assistant(message):
            await websocket.send(json.dumps({
                "type": "llm_chunk",
                "text": "**Error:** AI Assistant is not configured. Please add your API key in Settings.",
            }))
            await websocket.send(json.dumps({"type": "llm_done"}))
            return

        entries = message.get("entries", [])
        transcript_text = "\n".join(
            f"[{e.get('source', 'speaker').upper()}] {e.get('text', '')}"
            for e in entries
            if e.get("text", "").strip()
        )

        if not transcript_text.strip():
            await websocket.send(json.dumps({
                "type": "llm_chunk",
                "text": "**Error:** Transcript is empty — nothing to summarize.",
            }))
            await websocket.send(json.dumps({"type": "llm_done"}))
            return

        def _chunk_cb(text: str):
            self._event_bus.emit({"type": "llm_chunk", "text": text})

        def _done_cb():
            self._event_bus.emit({"type": "llm_done"})

        self._llm_assistant.request_summary(
            transcript_text=transcript_text,
            chunk_callback=_chunk_cb,
            done_callback=_done_cb,
        )

    async def _on_request_models_list(self, message: dict, websocket):
        from engine.core.model_manager import get_models_list
        from engine.core.config import Config
        try:
            config = Config()
            models = get_models_list(config)
            await websocket.send(json.dumps({
                "type": "models_list",
                "asr": models["asr"],
                "translation": models["translation"]
            }))
        except Exception as e:
            logger.error("Error getting models list: %s", e)

    async def _on_delete_model(self, message: dict, websocket):
        from engine.core.model_manager import delete_model
        from engine.core.config import Config
        model_id = message.get("model_id")
        model_type = message.get("model_type")
        success = False
        error = None
        try:
            config = Config()
            delete_model(model_id, model_type, config)
            success = True
        except Exception as e:
            logger.error("Error deleting model: %s", e)
            error = str(e)
            
        await websocket.send(json.dumps({
            "type": "model_action_result",
            "action": "delete",
            "model_id": model_id,
            "success": success,
            "error": error
        }))

    async def _on_download_model(self, message: dict, websocket):
        model_id = message.get("model_id")
        model_type = message.get("model_type")
        hf_token = message.get("hf_token")

        def _download_task():
            from engine.core.model_manager import download_model
            from engine.core.config import Config
            success = False
            error = None
            try:
                config = Config()
                download_model(model_id, model_type, config, hf_token=hf_token)
                success = True
            except Exception as e:
                logger.error("Error downloading model: %s", e)
                error = str(e)

            result_msg = {
                "type": "model_action_result",
                "action": "download",
                "model_id": model_id,
                "success": success,
                "error": error
            }
            self._event_bus.emit(result_msg)

        t = threading.Thread(target=_download_task, daemon=True)
        t.start()

    async def _on_search_hub(self, message: dict, websocket):
        query = message.get("query", "")
        task = message.get("task", "asr")
        hf_token = message.get("hf_token")
        
        def _search_task():
            from engine.core.model_manager import search_hub
            try:
                results = search_hub(query, task, hf_token=hf_token)
                result_msg = {
                    "type": "hub_search_results",
                    "results": results
                }
                self._event_bus.emit(result_msg)
            except Exception as e:
                logger.error("Error searching hub: %s", e)
                
        t = threading.Thread(target=_search_task, daemon=True)
        t.start()


def _generate_ws_token():
    token_path = Path.home() / ".echoflux" / "ws_token.txt"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    import uuid
    token = str(uuid.uuid4())
    token_path.write_text(token, encoding="utf-8")
    logger.info("Generated WebSocket Auth Token.")

if __name__ == "__main__":
    _load_dotenv()
    _setup_logging()
    _setup_cuda_paths()
    install_progress_hijack()
    _generate_ws_token()

    engine = EchoFluxEngine()
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        logger.info("Engine interrupted by user.")
    except Exception as e:
        logger.error("Fatal engine error: %s", e, exc_info=True)
