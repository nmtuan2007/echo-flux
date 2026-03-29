"""
engine/llm/assistant.py

Universal LLM assistant for EchoFlux.
Supports any OpenAI-compatible provider (OpenAI, OpenRouter, Ollama, LM Studio, etc.)
All API calls run on a background threading.Thread to avoid blocking the asyncio loop.
"""
import json
import logging
import threading
from typing import Callable, List, Optional

logger = logging.getLogger("echoflux.llm")

SUGGESTION_PROMPT = """\
You are an expert communication strategist and professional coach.

Context:
I am in a live professional conversation (e.g., a meeting, negotiation, or interview).
I need your help to respond to a specific statement just made by the other person.

Here is the recent context of the conversation:
{context}

The specific statement I need help replying to is:
"{target_text}"

Your Task:
Provide 3 distinct, practical ways to respond to or resolve this statement.
- Option 1: Cooperative / Problem-solving (Find common ground)
- Option 2: Clarifying / Probing (Ask a smart follow-up question)
- Option 3: Confident / Assertive (Stand your ground or provide a direct answer)

Keep each response under 2 sentences. Make them natural, conversational, and immediately usable.

Output strictly in this JSON format:
{{
  "options": [
    {{"strategy": "Cooperative", "text": "Your suggested response here..."}},
    {{"strategy": "Clarifying", "text": "Your suggested response here..."}},
    {{"strategy": "Assertive", "text": "Your suggested response here..."}}
  ]
}}
"""

SUMMARY_PROMPT = """\
You are an AI assistant specialized in analyzing professional conversations.
Analyze the following transcript and provide:

1. A concise summary (5-7 bullet points)
2. Key decisions or conclusions
3. Action items (include the responsible person if mentioned)
4. Any risks, blockers, or open questions

Keep the output clear, structured, and professional. Use markdown formatting.

Transcript:
{transcript}
"""


class LLMAssistant:
    """
    Wraps an OpenAI-compatible client and exposes async-friendly,
    thread-based methods for Smart Reply suggestions and meeting summarization.
    """

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
            logger.info(
                "LLMAssistant initialized: model=%s, base_url=%s",
                self._model,
                self._base_url or "default (OpenAI)",
            )
        except ImportError:
            logger.error(
                "openai package not installed. Run: pip install openai>=1.0.0"
            )
            self._client = None
        except Exception as e:
            logger.error("Failed to initialize LLM client: %s", e)
            self._client = None

    def is_available(self) -> bool:
        return self._client is not None

    def request_suggestion(
        self,
        entry_id: str,
        target_text: str,
        context: List[str],
        callback: Callable[[dict], None],
    ):
        """
        Non-blocking: spawns a thread to call the LLM and invoke callback(result_dict).
        result_dict has keys: "entry_id", "options" or "error".
        """
        t = threading.Thread(
            target=self._suggestion_thread,
            args=(entry_id, target_text, context, callback),
            name=f"LLM-Suggestion-{entry_id}",
            daemon=True,
        )
        t.start()

    def request_summary(
        self,
        transcript_text: str,
        chunk_callback: Callable[[str], None],
        done_callback: Callable[[], None],
    ):
        """
        Non-blocking: spawns a thread to stream the summary.
        chunk_callback(text) is called for each streamed chunk.
        done_callback() is called when complete.
        """
        t = threading.Thread(
            target=self._summary_thread,
            args=(transcript_text, chunk_callback, done_callback),
            name="LLM-Summary",
            daemon=True,
        )
        t.start()

    # ── Private thread workers ──────────────────────────────────────────────

    def _suggestion_thread(
        self,
        entry_id: str,
        target_text: str,
        context: List[str],
        callback: Callable[[dict], None],
    ):
        if not self._client:
            callback({"entry_id": entry_id, "error": "LLM client not available. Install openai package."})
            return

        context_str = "\n".join(f"- {s}" for s in context) if context else "(no prior context)"
        prompt = SUGGESTION_PROMPT.format(context=context_str, target_text=target_text)

        try:
            # Try with json_object response format (supported by OpenAI/OpenRouter)
            # Fall back to plain completion for local providers that don't support it
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=600,
                    response_format={"type": "json_object"},
                )
            except Exception:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=600,
                )

            raw = response.choices[0].message.content or ""

            # Extract JSON even if the model wrapped it in markdown code fences
            if "```" in raw:
                import re
                match = re.search(r"```(?:json)?\s*({.*?})\s*```", raw, re.DOTALL)
                if match:
                    raw = match.group(1)
            # Also handle if JSON starts mid-string
            if not raw.strip().startswith("{"):
                import re
                match = re.search(r"{.*}", raw, re.DOTALL)
                if match:
                    raw = match.group(0)

            data = json.loads(raw)
            callback({"entry_id": entry_id, "options": data.get("options", [])})
        except json.JSONDecodeError as e:
            logger.error("LLM returned non-JSON for suggestion: %s", e)
            callback({"entry_id": entry_id, "error": "LLM returned invalid response format."})
        except Exception as e:
            logger.error("Suggestion LLM error: %s", e)
            callback({"entry_id": entry_id, "error": str(e)})

    def _summary_thread(
        self,
        transcript_text: str,
        chunk_callback: Callable[[str], None],
        done_callback: Callable[[], None],
    ):
        if not self._client:
            chunk_callback("**Error:** LLM client not available. Install `openai` package.")
            done_callback()
            return

        prompt = SUMMARY_PROMPT.format(transcript=transcript_text)

        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1200,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    chunk_callback(delta.content)
        except Exception as e:
            logger.error("Summary LLM error: %s", e)
            chunk_callback(f"\n\n**Error:** {e}")
        finally:
            done_callback()
