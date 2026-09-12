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

Implement EVERY abstract member, or the class cannot be instantiated.
`tests/test_backend_contract.py` compares these signatures against the real ABCs.

For ASR (`engine.asr.base.ASRBackend`) — 6 members:

```python
def load_model(self, config: dict) -> None: ...
def transcribe_stream(self, audio_chunk: bytes, stream_id: str = "default") -> TranscriptResult: ...
def finalize_current(self, stream_id: str = "default") -> Optional[TranscriptResult]: ...
def reset_stream(self, stream_id: str = "default") -> None: ...
def unload_model(self) -> None: ...
@property
def is_loaded(self) -> bool: ...
```

`stream_id` is not optional in practice — the engine runs dual streams (`mic` and
`system`) concurrently, so a backend MUST keep per-stream state. `finalize_current`
flushes the buffered audio for one stream and returns its last segment, or `None`.

For Translation (`engine.translation.base.TranslationBackend`) — 5 members:

```python
def load_model(self, config: dict) -> None: ...
def translate_raw(self, text: str, source_lang: str, target_lang: str) -> TranslationResult: ...
def unload_model(self) -> None: ...
@property
def is_loaded(self) -> bool: ...
@property
def supported_pairs(self) -> list: ...
```

Implement `translate_raw`, never `translate` — the base class's `translate()` wraps it
with post-processing and empty-input handling.

## Step 3: Register the Backend

- Add the new backend to the appropriate `__init__.py`.
- Update `engine/main.py` in the `_initialize_pipeline` method to instantiate the new backend if requested by the client's configuration.

## Step 4: Graceful Degradation

- Ensure the `load_model` method attempts GPU initialization first, but catches `Exception` and falls back to CPU if necessary.
- Ensure any large model files are downloaded to `Config().models_dir`.
