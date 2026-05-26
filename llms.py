import os
import requests
from typing import Optional


class LLMManager:
    """Manage multiple LLM providers (OpenAI, custom HTTP endpoint, or fallback echo).
    Use environment variables `OPENAI_API_KEY`, `LLM_API_URL`, and `DEFAULT_MODEL`.
    """

    def __init__(self, openai_api_key: Optional[str] = None, llm_api_url: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm_api_url = llm_api_url or os.getenv("LLM_API_URL")
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

    def generate(self, prompt: str, model: Optional[str] = None, provider: Optional[str] = None) -> dict:
        model = model or self.default_model
        # choose provider if not specified
        if provider is None:
            if self.openai_api_key:
                provider = "openai"
            elif self.llm_api_url:
                provider = "custom"
            else:
                provider = "echo"

        if provider == "openai":
            if not self.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY not configured for provider 'openai'")
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 512}
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            j = r.json()
            text = j.get("choices", [{}])[0].get("message", {}).get("content") or j.get("choices", [{}])[0].get("text")
            return {"text": text, "raw": j}

        if provider == "custom":
            if not self.llm_api_url:
                raise RuntimeError("LLM_API_URL not configured for provider 'custom'")
            r = requests.post(self.llm_api_url, json={"prompt": prompt, "model": model}, timeout=30)
            r.raise_for_status()
            return r.json()

        # simple echo fallback
        return {"text": prompt[:100], "raw": {"provider": "echo"}}
