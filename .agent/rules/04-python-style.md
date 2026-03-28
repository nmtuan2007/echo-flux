---
description: Python formatting, typing, and linting rules
globs: "**/*.py"
---

# Python Code Style

## Formatting & Linting

- The project uses `ruff`. All code MUST comply with `ruff` rules specified in `pyproject.toml`.
- Line length is 100 characters.
- Use double quotes `"` for strings.
- 4 spaces for indentation.

## Type Hinting

- ALWAYS use type hints for function signatures (`def translate(self, text: str) -> str:`).
- Use `Optional`, `List`, `Dict`, `Tuple` from the `typing` module for pre-3.9 compatibility (project supports 3.10+, but maintain existing style).
- Dataclasses (`@dataclass`) are preferred for structured data records (e.g., `TranscriptResult`, `AudioDevice`).
