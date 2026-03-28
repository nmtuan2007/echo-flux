---
name: add-engine-backend
description: Add a new ASR or Translation AI backend to the Python Engine.
---

# Add Engine Backend Skill

To add a new AI backend to the EchoFlux engine:

## Step 1: Create the Backend Class

- Create a new file in `engine/asr/` or `engine/translation/`.
- Inherit from `ASRBackend` (in `engine.asr.base`) or `TranslationBackend` (in `engine.translation.base`).

## Step 2: Implement Abstract Methods

For ASR:

```python
def load_model(self, config: TranscriptionConfig) -> None: ...
def transcribe_stream(self, audio_chunk: bytes) -> Optional[TranscriptResult]: ...
def reset_stream(self) -> None: ...
def unload_model(self) -> None: ...
@property
def is_loaded(self) -> bool: ...
```

For Translation:

```python
def load_model(self, config: dict) -> None: ...
def translate_raw(self, text: str, source_lang: str, target_lang: str) -> TranslationResult: ...
def unload_model(self) -> None: ...
@property
def is_loaded(self) -> bool: ...
@property
def supported_pairs(self) -> list: ...
```

## Step 3: Register the Backend

- Add the new backend to the appropriate `__init__.py`.
- Update `engine/main.py` in the `_initialize_pipeline` method to instantiate the new backend if requested by the client's configuration.

## Step 4: Graceful Degradation

- Ensure the `load_model` method attempts GPU initialization first, but catches `Exception` and falls back to CPU if necessary.
- Ensure any large model files are downloaded to `Config().models_dir`.
