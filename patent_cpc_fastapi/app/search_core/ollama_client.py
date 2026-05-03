"""
OllamaClient — thin wrapper around the Ollama HTTP API.

Mirrors claims_core/ollama_client.py so both modules can share the same
pattern. If you already import OllamaClient from claims_core in your main
app, you can point search_core/__init__.py to that instead and delete this file.
"""

import json
import urllib.request
import urllib.error
from typing import Optional


class OllamaClient:
    """
    Sends chat requests to a locally running Ollama instance.

    Parameters
    ----------
    model_name : str
        The Ollama model tag, e.g. "gpt-oss:120b-cloud" or "llama3:70b".
    base_url : str
        Base URL of the Ollama API. Defaults to http://localhost:11434.
    timeout : int
        Request timeout in seconds. Defaults to 300 (5 minutes) to allow
        for large patent documents.
    """

    def __init__(
        self,
        model_name: str = "gpt-oss:120b-cloud",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Token usage tracking
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "total_duration_ns": 0,
        }

    @property
    def usage(self) -> dict:
        """Return cumulative token usage across all calls."""
        return dict(self._usage)

    def _record_usage(self, response_json: dict):
        """Extract and accumulate token counts from an Ollama response."""
        prompt_tokens = response_json.get("prompt_eval_count", 0)
        completion_tokens = response_json.get("eval_count", 0)
        duration = response_json.get("total_duration", 0)
        self._usage["prompt_tokens"] += prompt_tokens
        self._usage["completion_tokens"] += completion_tokens
        self._usage["total_tokens"] += prompt_tokens + completion_tokens
        self._usage["calls"] += 1
        self._usage["total_duration_ns"] += duration

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> Optional[str]:
        """
        Send a system + user message to the model and return the response text.

        Parameters
        ----------
        system_prompt : str
            The system instruction (search strategy prompt).
        user_message : str
            The user turn containing the patent document content.
        temperature : float
            Low temperature (0.1) for deterministic, factual output.
        max_tokens : int
            Maximum tokens in the response.

        Returns
        -------
        str or None
            The model's response text, or None on failure.

        Raises
        ------
        RuntimeError
            If the API call fails or returns an unexpected response.
        """
        endpoint = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. "
                f"Is Ollama running? Detail: {e}"
            ) from e

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Ollama returned non-JSON response: {raw[:200]}"
            ) from e

        # Record token usage
        self._record_usage(body)

        # Standard Ollama /api/chat response shape
        try:
            return body["message"]["content"]
        except (KeyError, TypeError):
            # Fallback: try flat "response" key (older Ollama versions)
            if "response" in body:
                return body["response"]
            raise RuntimeError(
                f"Unexpected Ollama response structure: {str(body)[:300]}"
            )

    def embeddings(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        """
        Generate a vector embedding for the given text.

        Parameters
        ----------
        text : str
            The input string to embed.
        model : str
            The embedding model to use (default: nomic-embed-text).

        Returns
        -------
        list of float
            The embedding vector.
        """
        endpoint = f"{self.base_url}/api/embeddings"
        payload = {"model": model, "prompt": text}
        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                body = json.loads(response.read().decode("utf-8"))
                return body["embedding"]
        except Exception as e:
            raise RuntimeError(f"Embedding call failed: {e}")
