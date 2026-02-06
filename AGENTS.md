### Development Guidelines (EchoFlux Project)

You are contributing to the EchoFlux open-source project.
Please follow these rules strictly.

1. Full File Delivery
   - Always provide the complete content of any file you create or modify.
   - Never send partial snippets or diffs.
   - The file must be ready to replace the existing file directly.

2. File Limit Per Response
   - Provide no more than 1–2 files per response.
   - Wait for explicit confirmation (""OK"") before continuing with additional files.
   - Do not anticipate or pre-generate future files.

3. Code Style & Comments
   - Avoid verbose or AI-style explanatory comments.
   - Only add comments when necessary.
   - Keep comments concise and natural, written as if for a professional dev team.
   - Do not explain obvious logic.

4. Architecture Compliance (MANDATORY)
   - Follow the EchoFlux modular architecture:
     - Engine logic must remain inside `/engine`
     - UI must not contain business logic
     - No direct model loading inside UI layer
   - Respect backend abstraction interfaces (ASRBackend, TranslationBackend, etc.).
   - New components must be pluggable and backend-agnostic.
   - No tight coupling between modules.

5. Configuration & Constants
   - Never hardcode paths, model names, ports, or magic values.
   - Use centralized configuration management.
   - Environment-specific values must be configurable.

6. Logging
   - Use the existing project logging mechanism.
   - Do not use print statements.
   - Logs must be structured and meaningful.
   - Logging levels must be appropriate (debug/info/warning/error).

7. Model & Resource Handling
   - Do not bundle models in code.
   - Models must be loaded dynamically from the configured model directory.
   - Ensure GPU/CPU fallback works correctly.

8. Cross-Platform Compatibility
   - Code must support both Windows and macOS.
   - Avoid OS-specific hacks unless properly abstracted.
   - Any OS-specific implementation must be isolated.

9. Performance Considerations
   - Streaming logic must be non-blocking.
   - Avoid unnecessary memory copies.
   - Keep latency-sensitive paths minimal.
   - No synchronous blocking calls in real-time processing loops.

10. Production Readiness
    - Code must be clean, testable, and maintainable.
    - Functions should be small and focused.
    - Prefer composition over inheritance.
    - Handle errors explicitly and safely.

If any architectural decision is unclear, ask before implementing.
Do not invent undocumented behavior.
