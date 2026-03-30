import os
import shutil
import threading
from pathlib import Path

from engine.core.config import Config
from engine.core.logging import get_logger

logger = get_logger("core.model_manager")

MODEL_CATALOG = {
    "asr": [
        {"id": "tiny", "name": "Whisper Tiny (Fastest, low accuracy)", "size_mb": 75},
        {"id": "base", "name": "Whisper Base", "size_mb": 140},
        {"id": "small", "name": "Whisper Small (Recommended for daily use)", "size_mb": 460},
        {"id": "medium", "name": "Whisper Medium", "size_mb": 1500},
        {"id": "large-v3", "name": "Whisper Large V3 (Most accurate, heavy)", "size_mb": 2900},
        {"id": "tiny.en", "name": "Whisper Tiny (English only)", "size_mb": 75},
        {"id": "base.en", "name": "Whisper Base (English only)", "size_mb": 140},
        {"id": "small.en", "name": "Whisper Small (English only)", "size_mb": 460},
    ],
    "translation": [
        # English to Others
        {"id": "Helsinki-NLP/opus-mt-en-vi", "name": "English → Vietnamese", "size_mb": 310},
        {"id": "Helsinki-NLP/opus-mt-en-zh", "name": "English → Chinese", "size_mb": 310},
        {"id": "Helsinki-NLP/opus-mt-en-jap", "name": "English → Japanese", "size_mb": 340},
        {"id": "Helsinki-NLP/opus-mt-tc-big-en-ko", "name": "English → Korean", "size_mb": 450},
        {"id": "Helsinki-NLP/opus-mt-en-fr", "name": "English → French", "size_mb": 300},
        {"id": "Helsinki-NLP/opus-mt-en-de", "name": "English → German", "size_mb": 300},
        {"id": "Helsinki-NLP/opus-mt-en-es", "name": "English → Spanish", "size_mb": 300},
        {"id": "Helsinki-NLP/opus-mt-en-ru", "name": "English → Russian", "size_mb": 300},
        # Others to English
        {"id": "Helsinki-NLP/opus-mt-vi-en", "name": "Vietnamese → English", "size_mb": 310},
        {"id": "Helsinki-NLP/opus-mt-zh-en", "name": "Chinese → English", "size_mb": 310},
        {"id": "Helsinki-NLP/opus-mt-ja-en", "name": "Japanese → English", "size_mb": 340},
        {"id": "Helsinki-NLP/opus-mt-fr-en", "name": "French → English", "size_mb": 300},
        {"id": "Helsinki-NLP/opus-mt-es-en", "name": "Spanish → English", "size_mb": 300},
    ]
}


def get_models_list(config: Config) -> dict:
    models_dir = config.models_dir
    result = {"asr": [], "translation": []}

    for model in MODEL_CATALOG["asr"]:
        path = models_dir / f"models--Systran--faster-whisper-{model['id']}"
        item = dict(model)
        item["is_downloaded"] = path.exists()
        result["asr"].append(item)

    ct2_dir = models_dir / "ct2"
    for model in MODEL_CATALOG["translation"]:
        path = ct2_dir / model["id"].replace("/", "_")
        item = dict(model)
        item["is_downloaded"] = path.exists()
        result["translation"].append(item)

    return result


def delete_model(model_id: str, model_type: str, config: Config) -> None:
    models_dir = config.models_dir
    path = None
    if model_type == "asr":
        path = models_dir / f"models--Systran--faster-whisper-{model_id}"
    elif model_type == "translation":
        path = models_dir / "ct2" / model_id.replace("/", "_")
    
    if path and path.exists():
        logger.info(f"Deleting model directory: {path}")
        shutil.rmtree(path, ignore_errors=True)
    else:
        logger.warning(f"Model path {path} not found for deletion.")


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


def download_model(model_id: str, model_type: str, config: Config) -> None:
    models_dir = config.models_dir
    if model_type == "asr":
        import faster_whisper
        logger.info(f"Downloading ASR model {model_id}...")
        faster_whisper.download_model(model_id, cache_dir=str(models_dir))
        logger.info(f"Successfully downloaded ASR model {model_id}.")
    elif model_type == "translation":
        output_path = models_dir / "ct2" / model_id.replace("/", "_")
        if output_path.exists() and (output_path / "model.bin").exists():
            logger.info(f"Translation model {model_id} is already downloaded.")
            return

        logger.info(f"Downloading and converting Translation model {model_id}...")
        _convert_marian_model(model_id, output_path)
        logger.info(f"Successfully downloaded Translation model {model_id}.")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
