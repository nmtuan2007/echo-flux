import json
from pathlib import Path
from engine.core.config import TranscriptionConfig
from engine.core.logging import get_logger
from engine.core.config import Config
from engine.asr.base import ASRBackend, TranscriptResult
from engine.asr.faster_whisper_backend import FasterWhisperBackend

logger = get_logger("core.asr_auto")

class AutoModelAdapter:
    """
    Factory pattern for Universal ASR model instantiation.
    Detects the downloaded model's runtime and returns the correct pipeline adapter.
    """

    @staticmethod
    def load(config: TranscriptionConfig) -> ASRBackend:
        model_id = config.model_path or config.model_size
        models_dir = Config().models_dir
        
        # Default behavior for legacy Faster-Whisper
        runtime = "ctranslate2" 
        
        # 1. Search for the local path of the downloaded model
        sanitized_id = model_id.replace("/", "--")
        hub_path = models_dir / f"models--{sanitized_id}"
        legacy_id = model_id.split("/")[-1]
        
        if legacy_id.startswith("faster-whisper-"):
            legacy_id = legacy_id.replace("faster-whisper-", "")
        legacy_path = models_dir / f"models--Systran--faster-whisper-{legacy_id}"
        
        local_dir = None
        if hub_path.exists():
            local_dir = hub_path
        elif legacy_path.exists():
            local_dir = legacy_path
            
        # 2. Read runtime from manifest if available
        manifest_path = None
        if local_dir:
            if (local_dir / "manifest.json").exists():
                manifest_path = local_dir / "manifest.json"
            else:
                # Check snapshots directory for Hugging Face cache structure
                snapshots_dir = local_dir / "snapshots"
                if snapshots_dir.exists() and snapshots_dir.is_dir():
                    for snap in snapshots_dir.iterdir():
                        if (snap / "manifest.json").exists():
                            manifest_path = snap / "manifest.json"
                            break

        if manifest_path:
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    runtime = manifest.get("runtime", "ctranslate2")
                    config.model_path = manifest.get("local_path", str(manifest_path.parent))
            except Exception as e:
                logger.warning(f"Failed to read manifest for {model_id}, defaulting to {runtime}. Error: {e}")

        logger.info(f"AutoModelAdapter routing ASR model {model_id} to [{runtime}] runtime")

        # 3. Instantiate the proper adapter
        if runtime == "ctranslate2":
            return FasterWhisperBackend()
        elif runtime == "transformers":
            from engine.asr.adapters.transformers_asr import TransformersASRAdapter
            return TransformersASRAdapter()
        elif runtime == "onnx":
            logger.error("ONNX ASR adapter not fully implemented yet!")
            raise NotImplementedError("ONNX is experimental and currently disabled.")
        else:
            raise ValueError(f"Unknown runtime defined in model manifest: {runtime}")
