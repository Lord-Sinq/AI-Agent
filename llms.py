"""
LLM Manager Module - Enhanced for Strict JSON Output
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
        self.azure_api_version = os.getenv("AZURE_API_VERSION", "2025-12-11")

        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.save_responses = os.getenv("SAVE_RESPONSES", "true").lower() == "true"

        # Add strict mode flag
        self.strict_json_mode = os.getenv("STRICT_JSON_MODE", "true").lower() == "true"

        if self.save_responses:
            self.responses_dir = Path.cwd() / "responses"
            self.responses_dir.mkdir(exist_ok=True)
            print(f"[INFO] Responses will be saved to: {self.responses_dir}")

        print(f"[INFO] Endpoint: {self.azure_endpoint}")
        print(f"[INFO] Deployment: {self.azure_deployment_name}")
        print(f"[INFO] API Version: {self.azure_api_version}")
        print(f"[INFO] API Key configured: {bool(self.azure_api_key)}")
        print(f"[INFO] Save responses: {self.save_responses}")
        print(f"[INFO] Strict JSON mode: {self.strict_json_mode}")

    def _get_system_prompt(self) -> str:
        """Get the system prompt based on strict mode setting."""
        if self.strict_json_mode:
            return """You are a JSON generator. STRICT RULES:
                1. Respond ONLY with valid JSON
                2. NO text, NO explanations, NO markdown before or after JSON
                3. NO  tags or any XML/HTML
                4. Start immediately with { or [
                5. End immediately with } or ]
                6. Use double quotes for strings
                7. No trailing commas
                8. Escape special characters properly

                Examples of ACCEPTABLE responses:
                {"features": ["age", "salary"], "scale": ["age"]}
                ["option1", "option2"]

                Examples of UNACCEPTABLE responses:
                Here is your JSON: {"key": "value"}
                {"key": "value"} (with extra text)
                Think step by step and then provide... {"key": "value"}"""
        else:
            return "You are a helpful assistant that provides accurate information."

    def _save_response(self, prompt: str, response: dict, model: str, max_tokens: int) -> None:
        if not self.save_responses:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        prompt_preview = prompt[:50].replace(" ", "_").replace("\n", "").replace("/", "_")
        filename = f"{timestamp}_{model}_{prompt_preview}.json"
        filepath = self.responses_dir / filename

        # Also save the raw response text separately for debugging
        raw_text_path = self.responses_dir / f"{timestamp}_raw_response.txt"
        raw_text_path.write_text(response.get("text", ""))

        response_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "response": response.get("text", ""),
            "response_length": len(response.get("text", "")),
            "strict_mode": self.strict_json_mode,
            "raw_response_file": str(raw_text_path)
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Response saved to: {filepath}")
            print(f"[INFO] Raw text saved to: {raw_text_path}")
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

            # Use the strict system prompt
            system_prompt = self._get_system_prompt()

            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,  # Lower for more deterministic output
                "top_p": 0.9,
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

                # Post-process if in strict mode
                if self.strict_json_mode and text:
                    original_length = len(text)
                    text = self._clean_json_response(text)
                    if len(text) != original_length:
                        print(f"[INFO] Cleaned response: {original_length} -> {len(text)} chars")

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

    def _clean_json_response(self, text: str) -> str:
        """Aggressively clean response to extract only valid JSON."""
        # Remove  tags
        if '<think>' in text:
            import re
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Remove common prefixes
        prefixes_to_remove = [
            r'^Here is your JSON:?\s*',
            r'^Here is the JSON:?\s*',
            r'^JSON:?\s*',
            r'^The JSON is:?\s*',
            r'^Output:?\s*',
            r'^Response:?\s*',
            r'^```json\s*',
            r'^```\s*',
            r'^\s*',
        ]

        import re
        for prefix in prefixes_to_remove:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)

        # Remove trailing code blocks
        text = re.sub(r'\s*```\s*$', '', text)

        # Find first { or [
        start = -1
        for i, char in enumerate(text):
            if char in '{[':
                start = i
                break

        # Find last } or ]
        end = -1
        for i in range(len(text) - 1, -1, -1):
            if text[i] in '}]':
                end = i
                break

        if start != -1 and end != -1 and end > start:
            json_candidate = text[start:end+1]
            # Validate it's valid JSON
            try:
                json.loads(json_candidate)
                return json_candidate
            except:
                pass

        # If we couldn't extract valid JSON, return original
        return text

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