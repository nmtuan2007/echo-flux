import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from huggingface_hub import HfApi, repo_info, snapshot_download

from engine.core.config import Config
from engine.core.logging import get_logger

logger = get_logger("core.model_manager")

def _get_manifest_path(model_dir: Path) -> Path:
    return model_dir / "manifest.json"

def _auto_detect_runtime(files: List[str]) -> str:
    """Detect runtime based on repository files."""
    # CT2 strict check: must be at the root of the repository
    has_model_bin = "model.bin" in files
    has_config_json = "config.json" in files
    
    has_onnx = any(f.endswith(".onnx") for f in files)
    has_safetensors = any(f.endswith(".safetensors") for f in files)
    has_pytorch_bin = any(f.endswith("pytorch_model.bin") for f in files)

    if has_model_bin and has_config_json:
        return "ctranslate2"
    elif has_onnx:
        return "onnx"
    elif has_safetensors or has_pytorch_bin:
        return "transformers"
    
    # Fallback to transformers if we can't tell
    return "transformers"

def search_hub(query: str, task: str, hf_token: str = None) -> List[Dict[str, Any]]:
    """Search Hugging Face Hub for models."""
    api = HfApi(token=hf_token or os.getenv("HF_TOKEN"))
    
    # Map our internal tasks to HF pipeline tags
    hf_task = "automatic-speech-recognition" if task == "asr" else "translation"
    
    try:
        models = api.list_models(
            search=query,
            filter=hf_task,
            sort="downloads",
            direction=-1,
            limit=20
        )
        
        results = []
        for m in models:
            results.append({
                "id": m.id,
                "downloads": getattr(m, "downloads", 0),
                "tags": getattr(m, "tags", []),
                "task": task
            })
        return results
    except Exception as e:
        logger.error(f"Error searching hub for '{query}': {e}")
        return []

def get_models_list(config: Config) -> dict:
    models_dir = config.models_dir
    result = {"asr": [], "translation": []}

    if not models_dir.exists():
        return result

    for item in models_dir.iterdir():
        if not item.is_dir():
            continue

        # Check for manifest first (New Universal format)
        manifest_path = _get_manifest_path(item)
        if not manifest_path.exists():
            snapshots_dir = item / "snapshots"
            if snapshots_dir.exists() and snapshots_dir.is_dir():
                for snap in snapshots_dir.iterdir():
                    if _get_manifest_path(snap).exists():
                        manifest_path = _get_manifest_path(snap)
                        break

        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                
                task = manifest.get("task", "asr")
                if task in result:
                    result[task].append({
                        "id": manifest.get("id"),
                        "name": manifest.get("id"),
                        "runtime": manifest.get("runtime"),
                        "size_mb": 0, # Size calculation can be added if needed
                        "is_downloaded": True
                    })
            except Exception as e:
                logger.warning(f"Failed to read manifest for {item.name}: {e}")
        
        # Legacy Faster Whisper models detection (no manifest)
        elif item.name.startswith("models--Systran--faster-whisper-"):
            model_id = item.name.replace("models--Systran--faster-whisper-", "")
            result["asr"].append({
                "id": f"Systran/faster-whisper-{model_id}",
                "name": f"Whisper {model_id.capitalize()} (Legacy)",
                "runtime": "ctranslate2",
                "size_mb": 0,
                "is_downloaded": True
            })

    # Scan for legacy CT2 translation models
    ct2_dir = models_dir / "ct2"
    if ct2_dir.exists():
        for item in ct2_dir.iterdir():
            if item.is_dir() and (item / "model.bin").exists():
                model_id = item.name.replace("_", "/")
                result["translation"].append({
                    "id": model_id,
                    "name": model_id,
                    "runtime": "ctranslate2",
                    "size_mb": 0,
                    "is_downloaded": True
                })

    return result

def delete_model(model_id: str, model_type: str, config: Config) -> None:
    models_dir = config.models_dir
    paths_to_remove = []
    
    # Check all directories for manifest with matching ID
    if models_dir.exists():
        for item in models_dir.iterdir():
            if not item.is_dir(): continue
            manifest_path = _get_manifest_path(item)
            if not manifest_path.exists():
                snapshots_dir = item / "snapshots"
                if snapshots_dir.exists() and snapshots_dir.is_dir():
                    for snap in snapshots_dir.iterdir():
                        if _get_manifest_path(snap).exists():
                            manifest_path = _get_manifest_path(snap)
                            break

            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    if manifest.get("id") == model_id:
                        paths_to_remove.append(item)
                except Exception:
                    pass

    # Handle legacy forms
    if model_type == "asr":
        legacy_id = model_id.split("/")[-1]
        if legacy_id.startswith("faster-whisper-"):
            legacy_id = legacy_id.replace("faster-whisper-", "")
        legacy_path = models_dir / f"models--Systran--faster-whisper-{legacy_id}"
        if legacy_path.exists():
            paths_to_remove.append(legacy_path)
    elif model_type == "translation":
        legacy_path = models_dir / "ct2" / model_id.replace("/", "_")
        if legacy_path.exists():
            paths_to_remove.append(legacy_path)
    
    for path in paths_to_remove:
        logger.info(f"Deleting model directory: {path}")
        shutil.rmtree(path, ignore_errors=True)
    
    if not paths_to_remove:
        logger.warning(f"Model ID {model_id} not found for deletion.")

def _convert_marian_model(model_id: str, output_path: Path):
    import ctranslate2.converters
    logger.info("Converting '%s' to CTranslate2 INT8 format...", model_id)
    try:
        converter = ctranslate2.converters.TransformersConverter(model_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        converter.convert(str(output_path), quantization="int8", force=True)
        logger.info("Conversion complete.")
    except Exception as e:
        if output_path.exists():
            shutil.rmtree(output_path, ignore_errors=True)
        raise e

def download_model(model_id: str, model_type: str, config: Config, hf_token: str = None) -> None:
    models_dir = config.models_dir
    
    # 1. Provide a quick fallback hook for Marian CT2 conversion if asked for Helsinki translations
    if model_type == "translation" and "Helsinki-NLP" in model_id and "preferred_runtime" not in config.__dict__:
        output_path = models_dir / "ct2" / model_id.replace("/", "_")
        if output_path.exists() and (output_path / "model.bin").exists():
            logger.info(f"Summary: Legacy Translation model {model_id} already downloaded.")
            return

        logger.info(f"Downloading and converting Translation model {model_id}...")
        _convert_marian_model(model_id, output_path)
        logger.info(f"Successfully downloaded Translation model {model_id}.")
        return

    hf_token = hf_token or os.getenv("HF_TOKEN")
    
    logger.info(f"Checking repo info for {model_id}...")
    
    # 2. Fetch repo info (files) from Hugging Face
    try:
        info = repo_info(model_id, token=hf_token)
        files = [f.rfilename for f in info.siblings]
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e) or "Gated" in str(e) or "403" in str(e):
            raise ValueError(f"Model '{model_id}' is restricted (403/401). Please visit its HuggingFace page to accept the terms, and ensure your Token in Settings has 'Read' permissions.")
        logger.error(f"Failed to fetch repo info for {model_id}: {e}")
        raise ValueError(f"Model {model_id} not found or inaccessible on Hub.")

    # 3. Auto-detect runtime
    runtime = _auto_detect_runtime(files)
    logger.info(f"Auto-detected runtime for {model_id}: {runtime}")

    # 4. Filter out heavy files that aren't needed by the runtime (to save bandwidth)
    ignore_patterns = ["*.msgpack", "*.h5", "*.tflite", ".DS_Store", "*.safetensors.index.json"] 
    # Notice we removed raw safetensors and pt here. We only add them if we ALREADY HAVE CT2 equivalents!
    
    if runtime == "ctranslate2":
        # Ignore heavy PyTorch/safetensors files if CT2 format is available
        ignore_patterns.extend(["*.safetensors", "pytorch_model.bin", "*.pt", "model.safetensors"])
    elif runtime == "transformers":
        ignore_patterns.extend(["*.onnx", "onnx/*", "model.bin"])
    elif runtime == "onnx":
        ignore_patterns.extend(["*.safetensors", "pytorch_model.bin", "*.pt"])
    
    logger.info(f"Downloading model {model_id} to {models_dir} (runtime: {runtime})...")
    
    def _download_progress(t): 
        from engine.core.progress import _hijack_progress
        # The snapshot_download internally uses tqdm. We hijack it in engine/core/progress.py
        pass

    # Ensure cache dir exists
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(models_dir),
            ignore_patterns=ignore_patterns,
            token=hf_token
        )
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "GatedRepoError" in str(type(e).__name__):
            raise ValueError(f"Model '{model_id}' is restricted (403/401). Please visit its HuggingFace page to accept the terms, and ensure your Token in Settings has 'Read' permissions.")
        raise e
    
    # 5. Save manifest
    manifest_info = {
        "id": model_id,
        "task": model_type,
        "runtime": runtime,
        "local_path": local_path,
        "downloaded_at": time.time()
    }
    
    manifest_path = _get_manifest_path(Path(local_path))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_info, f, indent=2)
        
    logger.info(f"Successfully downloaded {model_id} and saved manifest to {manifest_path}.")
