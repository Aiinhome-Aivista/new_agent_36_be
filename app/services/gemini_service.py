import json
import logging
from typing import Generator, Optional
import requests
import google.generativeai as genai

from app.config import Config
from app.rag.prompts import BASE_GUARDRAIL_SYSTEM_PROMPT

logger = logging.getLogger("DevKnow.LLM")

class LLMService:
    _instance = None
    _gemini_model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.provider = Config.LLM_PROVIDER.lower()
        self._init_provider()

    def _init_provider(self):
        self.provider = Config.LLM_PROVIDER.lower()
        logger.info(f"Initializing LLM Service for provider: [{self.provider.upper()}] (Mode: {Config.LLM_MODE})")

        if self.provider == "gemini":
            if Config.GEMINI_API_KEY and Config.GEMINI_API_KEY.strip():
                try:
                    genai.configure(api_key=Config.GEMINI_API_KEY)
                    generation_config = {
                        "temperature": Config.LLM_TEMPERATURE,
                        "top_p": Config.LLM_TOP_P,
                        "max_output_tokens": Config.LLM_MAX_TOKENS,
                    }
                    self._gemini_model = genai.GenerativeModel(
                        model_name=Config.GEMINI_MODEL,
                        system_instruction=BASE_GUARDRAIL_SYSTEM_PROMPT,
                        generation_config=generation_config
                    )
                    logger.info(f"Gemini client initialized with model: {Config.GEMINI_MODEL}")
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini client: {e}")
                    self._gemini_model = None

    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        """
        Dispatches streaming generation to the active LLM_PROVIDER:
        'gemini' | 'openai' | 'claude' | 'azure_openai' | 'custom'
        """
        provider = Config.LLM_PROVIDER.lower()

        if provider == "gemini":
            yield from self._stream_gemini(prompt)
        elif provider == "openai":
            yield from self._stream_openai(prompt)
        elif provider == "claude":
            yield from self._stream_claude(prompt)
        elif provider == "azure_openai":
            yield from self._stream_azure_openai(prompt)
        elif provider == "custom":
            yield from self._stream_custom_llm(prompt)
        else:
            yield from self._stream_gemini(prompt)

    def _get_graceful_fallback(self, prompt: str) -> str:
        """
        User-friendly response when LLM synthesis is temporarily unavailable.
        If organizational functions were retrieved in the prompt, highlights them cleanly.
        """
        if "### Function #" in prompt or "retrieved organizational database" in prompt.lower():
            return (
                "I have retrieved the relevant organizational function(s) from the knowledge base above. "
                "You can inspect their exact signatures, parameters, and implementations directly in the function cards."
            )
        return (
            "I am currently unable to synthesize an AI response. "
            "Please try submitting your question again shortly."
        )

    # ── 1. Google Gemini Streaming ──────────────────────────────────────────
    def _stream_gemini(self, prompt: str) -> Generator[str, None, None]:
        if not self._gemini_model and Config.GEMINI_API_KEY:
            self._init_provider()

        if not self._gemini_model:
            logger.error("[DevKnow LLM Error] Gemini client is not initialized. Please verify GEMINI_API_KEY in .env.")
            yield self._get_graceful_fallback(prompt)
            return

        try:
            response = self._gemini_model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.exception(f"Error streaming from Gemini: {e}")
            yield self._get_graceful_fallback(prompt)

    # ── 2. OpenAI (ChatGPT / GPT-4o) Streaming ──────────────────────────────
    def _stream_openai(self, prompt: str) -> Generator[str, None, None]:
        if not Config.OPENAI_API_KEY:
            logger.error("[DevKnow LLM Error] OPENAI_API_KEY is not configured in .env.")
            yield self._get_graceful_fallback(prompt)
            return

        base_url = Config.OPENAI_BASE_URL.rstrip('/')
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": Config.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": BASE_GUARDRAIL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": Config.LLM_TEMPERATURE,
            "top_p": Config.LLM_TOP_P,
            "max_tokens": Config.LLM_MAX_TOKENS,
            "stream": True
        }

        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    logger.error(f"[DevKnow OpenAI API Error] HTTP {resp.status_code} - {resp.text}")
                    yield self._get_graceful_fallback(prompt)
                    return
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            data_content = line_str.replace("data: ", "").strip()
                            if data_content == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_content)
                                delta = chunk["choices"][0]["delta"]
                                if "content" in delta and delta["content"]:
                                    yield delta["content"]
                            except Exception:
                                pass
        except Exception as e:
            logger.exception(f"OpenAI streaming error: {e}")
            yield self._get_graceful_fallback(prompt)

    # ── 3. Anthropic Claude Streaming ────────────────────────────────────────
    def _stream_claude(self, prompt: str) -> Generator[str, None, None]:
        if not Config.ANTHROPIC_API_KEY:
            logger.error("[DevKnow LLM Error] ANTHROPIC_API_KEY is not configured in .env.")
            yield self._get_graceful_fallback(prompt)
            return

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": Config.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": Config.ANTHROPIC_MODEL,
            "system": BASE_GUARDRAIL_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(Config.LLM_MAX_TOKENS, 4096),
            "temperature": Config.LLM_TEMPERATURE,
            "stream": True
        }

        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    logger.error(f"[DevKnow Claude API Error] HTTP {resp.status_code} - {resp.text}")
                    yield self._get_graceful_fallback(prompt)
                    return
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            data_content = line_str.replace("data: ", "").strip()
                            try:
                                chunk = json.loads(data_content)
                                if chunk.get("type") == "content_block_delta":
                                    delta = chunk.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        yield delta.get("text", "")
                            except Exception:
                                pass
        except Exception as e:
            logger.exception(f"Claude streaming error: {e}")
            yield self._get_graceful_fallback(prompt)

    # ── 4. Azure OpenAI Streaming ────────────────────────────────────────────
    def _stream_azure_openai(self, prompt: str) -> Generator[str, None, None]:
        if not Config.AZURE_OPENAI_API_KEY or not Config.AZURE_OPENAI_ENDPOINT:
            logger.error("[DevKnow LLM Error] AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT missing in .env.")
            yield self._get_graceful_fallback(prompt)
            return

        endpoint = Config.AZURE_OPENAI_ENDPOINT.rstrip('/')
        url = (
            f"{endpoint}/openai/deployments/{Config.AZURE_OPENAI_DEPLOYMENT}/chat/completions"
            f"?api-version={Config.AZURE_OPENAI_API_VERSION}"
        )
        headers = {
            "api-key": Config.AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [
                {"role": "system", "content": BASE_GUARDRAIL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": Config.LLM_TEMPERATURE,
            "top_p": Config.LLM_TOP_P,
            "max_tokens": Config.LLM_MAX_TOKENS,
            "stream": True
        }

        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    logger.error(f"[DevKnow Azure OpenAI Error] HTTP {resp.status_code} - {resp.text}")
                    yield self._get_graceful_fallback(prompt)
                    return
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            data_content = line_str.replace("data: ", "").strip()
                            if data_content == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_content)
                                delta = chunk["choices"][0]["delta"]
                                if "content" in delta and delta["content"]:
                                    yield delta["content"]
                            except Exception:
                                pass
        except Exception as e:
            logger.exception(f"Azure OpenAI error: {e}")
            yield self._get_graceful_fallback(prompt)

    # ── 5. Local / Custom LLM (Ollama, vLLM, Mistral) Streaming ───────────────
    def _stream_custom_llm(self, prompt: str) -> Generator[str, None, None]:
        url = Config.LLM_API_URL or "http://localhost:11434/api/generate"
        payload = {
            "model": Config.LLM_MODEL,
            "prompt": f"{BASE_GUARDRAIL_SYSTEM_PROMPT}\n\n{prompt}",
            "stream": True,
            "options": {
                "temperature": Config.LLM_TEMPERATURE,
                "top_p": Config.LLM_TOP_P
            }
        }

        try:
            with requests.post(url, json=payload, stream=True, timeout=Config.LLM_TIMEOUT) as resp:
                if resp.status_code != 200:
                    logger.error(f"[DevKnow Custom LLM Error] HTTP {resp.status_code} - {resp.text}")
                    yield self._get_graceful_fallback(prompt)
                    return
                for line in resp.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if "response" in chunk:
                                yield chunk["response"]
                            elif "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except Exception:
                            pass
        except Exception as e:
            logger.exception(f"Custom LLM error on {url}: {e}")
            yield self._get_graceful_fallback(prompt)

    def generate_single(self, prompt: str) -> str:
        """Non-streaming single-shot completion"""
        chunks = list(self.stream_generate(prompt))
        return "".join(chunks)

# Export singleton and alias for backwards-compatibility
llm_service = LLMService.get_instance()
gemini_service = llm_service
