"""
LLM Manager Module
"""

import os
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional


class LLMManager:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        self.azure_api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.save_responses = os.getenv("SAVE_RESPONSES", "true").lower() == "true"

        if self.save_responses:
            self.responses_dir = Path.cwd() / "responses"
            self.responses_dir.mkdir(exist_ok=True)
            print(f"[INFO] Responses will be saved to: {self.responses_dir}")

        print(f"[INFO] Endpoint: {self.azure_endpoint}")
        print(f"[INFO] Deployment: {self.azure_deployment_name}")
        print(f"[INFO] API Version: {self.azure_api_version}")
        print(f"[INFO] API Key configured: {bool(self.azure_api_key)}")
        print(f"[INFO] Save responses: {self.save_responses}")

    def _save_response(self, prompt: str, response: dict, model: str, max_tokens: int) -> None:
        if not self.save_responses:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        prompt_preview = prompt[:50].replace(" ", "_").replace("\n", "").replace("/", "_")
        filename = f"{timestamp}_{model}_{prompt_preview}.json"
        filepath = self.responses_dir / filename

        response_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "response": response.get("text", ""),  # Just the text
            "response_length": len(response.get("text", "")),
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Response saved to: {filepath}")
        except Exception as e:
            print(f"[WARNING] Failed to save response: {e}")

    def generate(self, prompt: str, model: Optional[str] = None, provider: Optional[str] = None, max_tokens: int = 4000) -> dict:
        model = model or self.azure_deployment_name

        if provider is None:
            if self.azure_endpoint and self.azure_api_key and self.azure_deployment_name:
                provider = "azure"
            else:
                raise RuntimeError("No LLM provider configured.")

        if provider == "azure":
            if not (self.azure_endpoint and self.azure_api_key and self.azure_deployment_name):
                raise RuntimeError("Azure OpenAI not fully configured")

            # Truncate prompt if too long (rough estimate)
            max_chars = 10000
            if len(prompt) > max_chars:
                print(f"[WARNING] Prompt length {len(prompt)} chars, truncating to {max_chars}")
                keep_start = int(max_chars * 0.7)
                keep_end = max_chars - keep_start
                prompt = prompt[:keep_start] + "\n...[truncated]...\n" + prompt[-keep_end:]

            base_endpoint = self.azure_endpoint.rstrip("/")
            url = f"{base_endpoint}/openai/deployments/{self.azure_deployment_name}/chat/completions?api-version={self.azure_api_version}"

            print(f"[INFO] Calling Azure OpenAI (prompt: {len(prompt)} chars, max_tokens: {max_tokens})")

            headers = {
                "api-key": self.azure_api_key,
                "Content-Type": "application/json",
            }

            payload = {
                "messages": [
                    {"role": "system", "content": "You are a JSON generator. Respond ONLY with valid JSON. No explanations, no markdown, start directly with { and end with }."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }

            try:
                r = requests.post(url, json=payload, headers=headers, timeout=60)
                r.raise_for_status()
                j = r.json()

                text = None
                if "choices" in j and len(j["choices"]) > 0:
                    choice = j["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        text = choice["message"]["content"]
                    elif "text" in choice:
                        text = choice["text"]

                if not text:
                    print(f"[WARNING] Unexpected response format")
                    text = ""

                print(f"[INFO] Generated {len(text)} chars")
                response = {"text": text, "raw": j}

                self._save_response(prompt, response, model, max_tokens)
                return response

            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                if hasattr(r, 'status_code'):
                    if r.status_code == 400:
                        error_msg = f"Bad request. Prompt length: {len(prompt)} chars. {str(e)}"
                    elif r.status_code == 404:
                        error_msg = f"Deployment '{self.azure_deployment_name}' not found. Check your .env file."
                raise RuntimeError(f"Azure OpenAI API call failed: {error_msg}") from e

        raise RuntimeError(f"Unknown provider: {provider}")

    def list_azure_deployments(self) -> dict:
        if not (self.azure_endpoint and self.azure_api_key):
            raise RuntimeError("Azure OpenAI listing requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")

        base_endpoint = self.azure_endpoint.rstrip("/")
        headers = {"api-key": self.azure_api_key, "Content-Type": "application/json"}
        list_api_version = "2024-02-15-preview"
        url = f"{base_endpoint}/openai/models?api-version={list_api_version}"

        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                j = r.json()
                data = j.get("data", [])
                chat_models = []
                chat_keywords = ["gpt", "chat", "instruct", "turbo", "deepseek"]

                for item in data:
                    if isinstance(item, dict):
                        model_id = item.get("id")
                        if model_id and any(keyword in model_id.lower() for keyword in chat_keywords):
                            chat_models.append({"id": model_id, "chat_completion": True})

                chat_models.sort(key=lambda x: x["id"])
                print(f"\nFound {len(chat_models)} chat-capable models:")
                print("=" * 50)
                for m in chat_models[:20]:
                    print(f"  - {m['id']}")
                if len(chat_models) > 20:
                    print(f"  ... and {len(chat_models) - 20} more")

                return {"route": "/openai/models", "api_version": list_api_version, "items": chat_models, "raw": j}
            else:
                raise RuntimeError(f"Failed to list models: {r.status_code} - {r.text}")
        except Exception as e:
            raise RuntimeError(f"Could not list Azure OpenAI models: {e}") from e