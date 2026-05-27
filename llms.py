import os
import requests
from typing import Optional


class LLMManager:
    """Manage LLM providers with priority: Azure OpenAI."""

    def __init__(self, openai_api_key: Optional[str] = None):
        # Azure OpenAI configuration
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        self.azure_api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

        print(f"[INFO] Endpoint: {self.azure_endpoint}")
        print(f"[INFO] Deployment: {self.azure_deployment_name}")
        print(f"[INFO] API Version: {self.azure_api_version}")
        print(f"[INFO] API Key configured: {bool(self.azure_api_key)}")

    def generate(self, prompt: str, model: Optional[str] = None, provider: Optional[str] = None) -> dict:
        model = model or self.default_model

        if provider is None:
            if self.azure_endpoint and self.azure_api_key and self.azure_deployment_name:
                provider = "azure"
            else:
                raise RuntimeError(
                    "No LLM provider configured. Please set:\n"
                    "  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME\n"
                )

        if provider == "azure":
            if not (self.azure_endpoint and self.azure_api_key and self.azure_deployment_name):
                raise RuntimeError("Azure OpenAI not fully configured")

            # Clean up endpoint
            base_endpoint = self.azure_endpoint.rstrip("/")

            # Construct URL for chat completions
            url = f"{base_endpoint}/openai/deployments/{self.azure_deployment_name}/chat/completions?api-version={self.azure_api_version}"

            print("[INFO] Calling Azure OpenAI...")

            headers = {
                "api-key": self.azure_api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.7,
            }

            try:
                r = requests.post(url, json=payload, headers=headers, timeout=30)
                r.raise_for_status()
                j = r.json()
                text = j.get("choices", [{}])[0].get("message", {}).get("content") or j.get("choices", [{}])[0].get(
                    "text"
                )
                return {"text": text, "raw": j}
            except requests.exceptions.RequestException as e:
                if hasattr(r, "status_code") and r.status_code == 404:
                    # Provide helpful suggestions
                    working_models = ["gpt-4o-mini", "gpt-35-turbo", "gpt-4o", "gpt-4"]
                    raise RuntimeError(
                        f"\n❌ Deployment '{self.azure_deployment_name}' not found or doesn't support chat completions.\n\n"
                        f"Try one of these verified chat models in your .env file:\n"
                        f"  AZURE_DEPLOYMENT_NAME=gpt-4o-mini  (Recommended)\n"
                        f"  AZURE_DEPLOYMENT_NAME=gpt-35-turbo (Faster/Cheaper)\n"
                        f"  AZURE_DEPLOYMENT_NAME=gpt-4o       (Most Capable)\n\n"
                        f"Your current deployment: {self.azure_deployment_name}\n"
                        f"Run 'python main.py --list-deployments' to see all available models."
                    )
                raise RuntimeError(f"Azure OpenAI API call failed: {str(e)}")

        raise RuntimeError(f"Unknown provider: {provider}")

    def list_azure_deployments(self) -> dict:
        """List available models that support chat completions."""
        if not (self.azure_endpoint and self.azure_api_key):
            raise RuntimeError("Azure OpenAI listing requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")

        base_endpoint = self.azure_endpoint.rstrip("/")
        headers = {"api-key": self.azure_api_key, "Content-Type": "application/json"}
        list_api_version = "2024-02-15-preview"

        # List all models (this endpoint works for you)
        url = f"{base_endpoint}/openai/models?api-version={list_api_version}"

        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                j = r.json()
                data = j.get("data", [])

                # Filter for chat-capable models
                chat_models = []
                chat_keywords = ["gpt", "chat", "instruct", "turbo", "deepseek"]

                for item in data:
                    if isinstance(item, dict):
                        model_id = item.get("id")
                        if model_id and any(keyword in model_id.lower() for keyword in chat_keywords):
                            chat_models.append({"id": model_id, "chat_completion": True})

                # Sort and display
                chat_models.sort(key=lambda x: x["id"])

                print(f"\n✅ Found {len(chat_models)} chat-capable models:")
                print("=" * 50)
                for model in chat_models[:20]:  # Show first 20
                    print(f"  • {model['id']}")
                if len(chat_models) > 20:
                    print(f"  ... and {len(chat_models) - 20} more")

                return {
                    "route": "/openai/models",
                    "api_version": list_api_version,
                    "items": chat_models,
                    "raw": j,
                }
            else:
                raise RuntimeError(f"Failed to list models: {r.status_code} - {r.text}")
        except Exception as e:
            raise RuntimeError(f"Could not list Azure OpenAI models: {e}")
